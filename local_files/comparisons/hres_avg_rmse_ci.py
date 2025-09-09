# rmse_pipeline_with_ci_and_improvement.py
import os
import glob
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

# ----------------- config (edit as needed) -----------------
LAT_MIN, LAT_MAX = 6, 38
LON_MIN, LON_MAX = 65, 95

ROLLED_DIR = "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/025res"
HRES_PATH = "/Datastorage/divij.khaitan_asp25/forecasts_2025/correct_6hourly_hres_forecasts20250701_20250730.nc"

TARGET_PREFIX = "era_precip025_target_init_"
BASE_PREFIX = "'base_precip025_init_"
FINETUNED_PREFIX = "fine_precip025_init_"  # change if your finetuned prefix differs

TARGET_VAR = "total_precipitation_6hr"  # (assumed) mm / 6h
HRES_VAR = "tp_6h"                         # meters / 6h -> convert to mm

OUT_CSV = "./rmse_by_init_and_lead_hres_base_fine201408onward.csv"
OUT_PNG = "./plots/rmse_levels.png"


LOCATIONS = {
    # Country-wide bounding box
    "india": {"lat_min": 6.0, "lat_max": 38.0, "lon_min": 65.0, "lon_max": 95.0},

    # Major Indian cities (rounded to nearest integer degree ranges)
    "mumbai":   {"lat_min": 18.0, "lat_max": 20.0, "lon_min": 72.0, "lon_max": 74.0},
    "delhi":    {"lat_min": 28.0, "lat_max": 29.0, "lon_min": 76.0, "lon_max": 78.0},
    "kolkata":  {"lat_min": 22.0, "lat_max": 23.0, "lon_min": 88.0, "lon_max": 89.0},
    "chennai":  {"lat_min": 12.0, "lat_max": 13.0, "lon_min": 80.0, "lon_max": 81.0},
    "bengaluru":{"lat_min": 12.0, "lat_max": 13.0, "lon_min": 77.0, "lon_max": 78.0},
    "hyderabad":{"lat_min": 17.0, "lat_max": 18.0, "lon_min": 78.0, "lon_max": 79.0},
    "ahmedabad":{"lat_min": 23.0, "lat_max": 24.0, "lon_min": 72.0, "lon_max": 73.0},
    "pune":     {"lat_min": 18.0, "lat_max": 19.0, "lon_min": 73.0, "lon_max": 74.0},
    "jaipur":   {"lat_min": 26.0, "lat_max": 27.0, "lon_min": 75.0, "lon_max": 76.0},
    "lucknow":  {"lat_min": 26.0, "lat_max": 27.0, "lon_min": 80.0, "lon_max": 81.0},
    "bhopal":   {"lat_min": 23.0, "lat_max": 24.0, "lon_min": 77.0, "lon_max": 78.0},
    "guwahati": {"lat_min": 26.0, "lat_max": 27.0, "lon_min": 91.0, "lon_max": 92.0},
    "srinagar": {"lat_min": 34.0, "lat_max": 35.0, "lon_min": 74.0, "lon_max": 75.0},
    "thiruvananthapuram": {"lat_min": 8.0, "lat_max": 9.0, "lon_min": 76.0, "lon_max": 77.0},
}

# ----------------- utilities (HAC, selection, grids) -----------------

def _autocorr(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < lag + 1:
        return np.nan
    x = x - x.mean()
    denom = np.dot(x, x)
    numer = np.dot(x[:-lag], x[lag:])
    return numer / denom if denom > 0 else np.nan

def inflation_factor_k(series, max_lag=2):
    """
    HAC/Bartlett inflation for serial correlation:
      k = sqrt( 1 + 2 * sum_{h=1..L} (1 - h/n) * rho(h) )
    """
    y = pd.Series(series, dtype=float).dropna().values
    n = len(y)
    if n <= 1:
        return 1.0
    rhos = []
    for h in range(1, min(max_lag, n-1) + 1):
        r = _autocorr(y, h)
        if not np.isfinite(r):
            r = 0.0
        rhos.append((1 - h / n) * r)
    k2 = 1.0 + 2.0 * np.sum(rhos)
    return float(np.sqrt(max(k2, 1.0)))  # never deflate

def _select_bbox(da, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX):
    """Select a lat/lon box, robust to coordinate direction (ascending/descending)."""
    lat = da["lat"]; lon = da["lon"]

    # Handle descending coordinates safely
    lat_start, lat_end = (lat_max, lat_min) if lat[0] > lat[-1] else (lat_min, lat_max)
    lon_start, lon_end = (lon_max, lon_min) if lon[0] > lon[-1] else (lon_min, lon_max)

    return da.sel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end))


def _select_region(da, region_name, locations=LOCATIONS):
    """Select a named region from LOCATIONS."""
    if region_name not in locations:
        raise KeyError(f"Unknown region '{region_name}'. Available: {list(locations.keys())}")
    bbox = locations[region_name]
    return _select_bbox(da, **bbox)


def _to_float_list(arr_like):
    """Materialize possible Dask/xarray arrays to python floats (no .item(), works for vectors)."""
    a = arr_like
    if hasattr(a, "compute"):
        try:
            a = a.compute()
        except Exception:
            pass
    a = np.asarray(a)
    return [float(v) for v in a.ravel()]



def create_regridder(src_da, dst_da):
    if os.path.exists('./hres_era5_regridder025.nc') and abs(dst_da['lat'].values[0] - dst_da['lat'].values[1]) == 0.25:
        print(f"Reusing weights for 0.25 resolution")
        temp_regridder = xe.Regridder(src_da, dst_da, method="bilinear", reuse_weights=True, filename='./hres_era5_regridder025.nc')
    else:
        temp_regridder = xe.Regridder(src_da, dst_da, method="bilinear", reuse_weights=False)
    if not os.path.exists('./hres_era5_regridder025.nc'):
        temp_regridder.to_netcdf('./hres_era5_regridder025.nc')
    return temp_regridder

def regrid_each_step(da_step_lat_lon, regridder, step_dim="step", savepath = None):
    if os.path.exists(savepath):
        print(f"Returning regridded hres with {savepath.split('/')[-1]} ")
        return xr.open_dataset(savepath)
    out_list = []
    for i in range(da_step_lat_lon.sizes[step_dim]):
        out_i = regridder(da_step_lat_lon.isel({step_dim: i}))
        out_list.append(out_i)
    out = xr.concat(out_list, dim=step_dim)
    if step_dim in da_step_lat_lon.coords:
        out = out.assign_coords({step_dim: da_step_lat_lon[step_dim]})

    if not os.path.exists(savepath):
        out.to_netcdf(savepath)

    return out

def _standardize_pred_da(ds, var=TARGET_VAR):
    if var in ds.data_vars:
        da = ds[var]
    else:
        da = ds
    if "batch" in da.dims:
        da = da.squeeze("batch", drop=True)
    if "time" in da.dims:
        da = da.rename({"time": "step"})
    return da

def _compute_lead_hours_from_coord(step_coord, init_dt64):
    vals = step_coord.values
    if np.issubdtype(vals.dtype, np.timedelta64):
        lead_hours = (vals / np.timedelta64(1, "h")).astype(float)
    elif np.issubdtype(vals.dtype, np.datetime64):
        lead_hours = ((vals - init_dt64) / np.timedelta64(1, "h")).astype(float)
    else:
        # fallback assume 6-hour cadence
        lead_hours = np.arange(step_coord.size) * 6.0
    return lead_hours

def _extract_init_date_from_filename(path, prefix):
    base = os.path.basename(path)
    if not base.startswith(prefix):
        return None
    stem = base[len(prefix):].rsplit(".nc", 1)[0]  # 'YYYY-MM-DD 00:00:00'
    return stem.split()[0]                          # 'YYYY-MM-DD'

# ----------------- RMSE computation (multi-init loop) -----------------

def compute_rmse_table(rolled_dir=ROLLED_DIR,
                       hres_path=HRES_PATH,
                       finetuned_prefix=FINETUNED_PREFIX):
    logging.info("Discovering initialization dates from targets...")
    target_paths = sorted(glob.glob(os.path.join(rolled_dir, f"{TARGET_PREFIX}*.nc")))
    if not target_paths:
        raise FileNotFoundError(f"No target files found: {rolled_dir}/{TARGET_PREFIX}*.nc")

    init_dates = sorted({ _extract_init_date_from_filename(p, TARGET_PREFIX)
                          for p in target_paths if _extract_init_date_from_filename(p, TARGET_PREFIX) })
    logging.info(f"Found {len(init_dates)} init dates.")

    logging.info("Opening HRES...")
    hres = xr.open_dataset(hres_path, decode_timedelta=True)
    hres_tp = hres[HRES_VAR]  # (time, step, lat, lon), meters / 6h

    logging.info("Creating regridder (HRES grid -> target grid)...")
    first_target = xr.open_dataset(os.path.join(rolled_dir, f"{TARGET_PREFIX}{init_dates[0]} 00:00:00.nc"), decode_timedelta=True)
    target_grid_da = _standardize_pred_da(first_target, var=TARGET_VAR).isel(step=0)
    regridder = create_regridder(hres_tp.isel(time=0, step=1), target_grid_da)

    rows = []

    for date_str in tqdm(init_dates[25:50], desc="Processing init dates"):
        print(date_str)
        init_iso = f"{date_str}T00:00:00"
        init_dt64 = np.datetime64(init_iso)

        # ground truth
        tgt_path = os.path.join(rolled_dir, f"{TARGET_PREFIX}{date_str} 00:00:00.nc")
        if not os.path.exists(tgt_path):
            logging.warning(f"Missing target for {date_str}, skipping.")
            continue
        target_ds = xr.open_dataset(tgt_path,decode_timedelta=True)
        gt = _standardize_pred_da(target_ds, var=TARGET_VAR)

        # HRES -> mm, drop step 0
        try:
            if hres_tp.attrs.get('units') == 'm':
                print("Units in metres already")
                hres_sel = hres_tp.sel(time=np.datetime64(init_iso)).isel(step=slice(1, None))
            else:
                hres_sel = hres_tp.sel(time=np.datetime64(init_iso)).isel(step=slice(1, None)) * 1000.0
        except Exception as e:
            logging.warning(f"HRES selection failed for {date_str}: {e}")
            continue
        print("Starting Regrid")
        hres_rg = regrid_each_step(hres_sel, regridder, step_dim="step", savepath = '/Datastorage/saptarishi.dhanuka_asp25/hres_2025_regridded_07.nc')
        print("Finished Regrid")

        # Align steps across GT & HRES; drop GT step 0 to match 6h,12h,...
        common_steps = min(hres_rg.sizes["step"], gt.sizes["step"] - 1)
        if common_steps <= 0:
            logging.warning(f"No overlapping steps for {date_str}, skipping.")
            continue
        gt_use = gt.isel(step=slice(1, 1 + common_steps))
        hres_use = hres_rg.isel(step=slice(0, common_steps))

        # Base & Finetuned (optional)
        base_da = None
        base_path = os.path.join(rolled_dir, f"{BASE_PREFIX}{date_str} 00:00:00.nc")
        if os.path.exists(base_path):
            base_ds = xr.open_dataset(base_path,decode_timedelta=True)
            base_da = _standardize_pred_da(base_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        ft_da = None
        finetuned_path = os.path.join(rolled_dir, f"{finetuned_prefix}{date_str} 00:00:00.nc")
        if os.path.exists(finetuned_path):
            ft_ds = xr.open_dataset(finetuned_path,decode_timedelta=True)
            ft_da = _standardize_pred_da(ft_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        # lead hours from GT step coordinate
        lead_hours = _compute_lead_hours_from_coord(gt_use["step"], init_dt64)

        def rmse_over_india(pred_da, truth_da):
            pred_box = _select_bbox(pred_da)
            truth_box = _select_bbox(truth_da)
            diff = pred_box - truth_box
            return np.sqrt((diff ** 2).mean(dim=["lat", "lon"]))

        # HRES rows
        rmse_hres = rmse_over_india(hres_use, gt_use).values
        for lh, val in zip(lead_hours, rmse_hres):
            rows.append({"model": "HRES", "init_date": date_str,
                         "forecast_horizon_hours": float(lh), "rmse": float(val)})

        # Base rows
        if base_da is not None:
            rmse_base = rmse_over_india(base_da, gt_use).values
            for lh, val in zip(lead_hours, rmse_base):
                rows.append({"model": "Graphcast_Base", "init_date": date_str,
                             "forecast_horizon_hours": float(lh), "rmse": float(val)})

        # Finetuned rows
        if ft_da is not None:
            rmse_ft = rmse_over_india(ft_da, gt_use).values
            for lh, val in zip(lead_hours, rmse_ft):
                rows.append({"model": "Graphcast_Finetuned1", "init_date": date_str,
                             "forecast_horizon_hours": float(lh), "rmse": float(val)})

    df = pd.DataFrame(rows).sort_values(["model", "init_date", "forecast_horizon_hours"])
    return df

# ----------------- summarization (RMSE mean + HAC CI) -----------------

def summarize_rmse_by_model(df,
                            lead_col="forecast_horizon_hours",
                            init_col="init_date",
                            model_col="model",
                            rmse_col="rmse",
                            alpha=0.20,
                            max_lag=2):
    """
    One row per (model, lead): mean RMSE and HAC (Bartlett) inflated CI.
    Uses normal approximation on RMSE series across init dates.
    """
    rows = []
    zcrit = stats.norm.ppf(1 - alpha/2.0)

    for (model, lead), g in df.groupby([model_col, lead_col]):
        g = g[[init_col, rmse_col]].dropna().sort_values(init_col)
        vals = pd.to_numeric(g[rmse_col], errors="coerce").dropna().values
        n = len(vals)
        if n < 2:
            rows.append({"model": model, "lead": lead, "n": n,
                         "rmse_hat": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "k": 1.0})
            continue

        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        k = inflation_factor_k(vals - m, max_lag=max_lag)
        se = k * s / np.sqrt(n)

        lo = max(0.0, m - zcrit * se)  # RMSE ≥ 0
        hi = m + zcrit * se

        rows.append({"model": model, "lead": lead, "n": n,
                     "rmse_hat": m, "ci_lo": lo, "ci_hi": hi, "k": k})

    return pd.DataFrame(rows).sort_values(["model", "lead"]).reset_index(drop=True)

# ----------------- plotting (levels + % improvement) -----------------

def plot_rmse_levels(summary_df,
                     baseline_model,
                     models_to_plot=None,
                     custom_colors=None,
                     title="RMSE Comparison and % Improvement",
                     ymin=0, ymax=16,
                     savepath=None,
                     models_rename=None,
                     improvement_ylim=(None, 70)):
    """
    Top: RMSE level curves with shaded CI
    Bottom: % improvement vs baseline (lower RMSE is better)
    """
    if summary_df.empty:
        raise ValueError("summary_df is empty.")

    all_models = list(summary_df["model"].unique())
    models = all_models if models_to_plot is None else [m for m in models_to_plot if m in all_models]
    print(models)
    if baseline_model not in models:
        raise ValueError(f"Baseline model '{baseline_model}' not found.")
    models = [baseline_model] + [m for m in models if m != baseline_model]

    leads = np.sort(summary_df["lead"].unique())

    # colors (baseline black unless overridden)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    cmap = {}
    for i, m in enumerate(models):
        if custom_colors and m in custom_colors:
            cmap[m] = custom_colors[m]
        else:
            cmap[m] = "black" if m == baseline_model else cycle[(i-1) % len(cycle)]

    # figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # --- top: RMSE curves with CIs ---
    for m in models:
        g = summary_df[summary_df["model"] == m].sort_values("lead")
        if g.empty: continue
        x = g["lead"].values
        y = g["rmse_hat"].values
        lo = g["ci_lo"].values
        hi = g["ci_hi"].values

        lw = 3.0 if m == baseline_model else 2.0
        z = 5 if m == baseline_model else 4
        label = models_rename.get(m, m) if models_rename else m
        ax1.plot(x, y, linestyle="-", linewidth=lw, color=cmap[m], label=label, zorder=z)
        ax1.fill_between(x, lo, hi, color=cmap[m], alpha=(0.12 if m == baseline_model else 0.16), linewidth=0)

    ax1.set_title(title, fontsize=20, pad=10)
    ax1.set_ylabel("RMSE per 6 hours", fontsize=18)
    ax1.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
    ax1.legend(title="", ncols=2, fontsize=21)

    if ymin is not None or ymax is not None:
        ax1.set_ylim(bottom=ymin if ymin is not None else ax1.get_ylim()[0],
                     top=ymax if ymax is not None else ax1.get_ylim()[1])

    # --- bottom: % improvement vs baseline ---
    base = summary_df[summary_df["model"] == baseline_model].sort_values("lead")[["lead", "rmse_hat"]]
    base = base.rename(columns={"rmse_hat": "rmse_base"})

    for m in models:
        if m == baseline_model: continue
        g = summary_df[summary_df["model"] == m].sort_values("lead")[["lead", "rmse_hat"]]
        merged = pd.merge(base, g, on="lead", how="inner")
        # improvement = positive is better (lower RMSE than baseline)
        denom = merged["rmse_base"].replace(0, np.nan)
        improvement = 100.0 * (merged["rmse_base"] - merged["rmse_hat"]) / denom
        ax2.plot(merged["lead"].values, improvement.values, linestyle="-", linewidth=2.0, color=cmap[m],
                 label=models_rename.get(m, m) if models_rename else m)

    ax2.axhline(0, color="black", linewidth=1, linestyle="--")
    ax2.set_xlabel("Lead time (hours)", fontsize=18)
    ax2.set_ylabel("% Improvement", fontsize=18)
    ax2.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
    ax2.legend(fontsize=18)

    if improvement_ylim is not None:
        lo, hi = improvement_ylim
        ax2.set_ylim(bottom=lo if lo is not None else ax2.get_ylim()[0],
                     top=hi if hi is not None else ax2.get_ylim()[1])

    # ticks every 24h on both
    step = 24 if (leads.max() - leads.min() >= 96) else max(6, int(np.median(np.diff(leads))) if len(leads) > 1 else 6)
    ax1.set_xticks(np.arange(leads.min()-12, leads.max() + 1, 24))
    ax2.set_xticks(np.arange(leads.min()-12, leads.max() + 1, 24))

    ax1.tick_params(axis="both", labelsize=18)
    ax2.tick_params(axis="both", labelsize=18)

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    # plt.show()



# ---- NEW: average target precipitation per lead (month >= 8) ----

def compute_avg_target_precip_by_lead(
    rolled_dir=ROLLED_DIR,
    target_prefix=TARGET_PREFIX,
    target_var=TARGET_VAR,
    min_month=8,                     # month >= 8 (August onward)
    drop_first_step=True,            # keep consistent with RMSE/lead alignment
    lat_min=LAT_MIN, lat_max=LAT_MAX,
    lon_min=LON_MIN, lon_max=LON_MAX
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - lead_hours (float)
      - mean_precip_mm_6h (float): India-box mean of target precipitation, averaged across init dates
      - n_inits (int): number of init dates contributing to that lead
    Scans all 'target_init_YYYY-MM-DD 00:00:00.nc' with month >= min_month.
    """
    import glob
    year = 2025

    all_targets = sorted(glob.glob(os.path.join(rolled_dir, f"{target_prefix}*.nc")))
    # print(all_targets)
    if not all_targets:
        raise FileNotFoundError(f"No target files found under {rolled_dir}/{target_prefix}*.nc")

    recs = []
    for p in all_targets:
        date_str = _extract_init_date_from_filename(p, target_prefix)
        if not date_str:
            continue
        try:
            y, m, d = map(int, date_str.split("-"))
        except Exception:
            continue
        if (m < min_month and y != year) or (y !=year):
            # print(m, y)
            continue

        init_iso = f"{date_str}T00:00:00"
        print(init_iso)
        init_dt64 = np.datetime64(init_iso)

        ds = xr.open_dataset(p,decode_timedelta=True)
        da = _standardize_pred_da(ds, var=target_var)
        # (optional) drop step 0 to align with 6h, 12h, ...
        if drop_first_step and da.sizes.get("step", 0) > 0:
            da = da.isel(step=slice(1, None))

        # India box mean per step
        da_box = _select_bbox(da, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        # print(da_box.lat)
        # print(da_box.lon)
        step_mean = da_box.mean(dim=["lat", "lon"])  # mm / 6h assumed
        # lead hours
        lead_hours = _compute_lead_hours_from_coord(step_mean["step"], init_dt64)
        # store
        df_i = pd.DataFrame({
            "init_date": date_str,
            "lead_hours": lead_hours.astype(float),
            "precip_mm_6h": step_mean.values.astype(float),
        })
        recs.append(df_i)

    if not recs:
        raise RuntimeError(f"No eligible target files (month >= min_month) found for year {year}")

    df_all = pd.concat(recs, ignore_index=True)
    # align leads (just in case of float drift)
    df_all["lead_hours"] = np.rint(df_all["lead_hours"]).astype(float)

    agg = (df_all
           .groupby("lead_hours", as_index=False)
           .agg(mean_precip_mm_6h=("precip_mm_6h", "mean"),
                n_inits=("init_date", "nunique")))
    agg = agg.sort_values("lead_hours").reset_index(drop=True)
    return agg


# ---- NEW: RMSE + CI plot with a secondary axis for avg precip ----

def plot_rmse_levels_with_precip(
    df_rmse,                                   # tidy per-init RMSE table: model, init_date, forecast_horizon_hours, rmse
    rolled_dir=ROLLED_DIR,                     # where target files live
    baseline_model="base",
    models_to_plot=None,
    custom_colors=None,
    models_rename=None,
    title="RMSE (with HAC CIs) + Avg Precip vs Lead",
    alpha=0.20,
    max_lag=0.1,
    min_month_for_precip=8,
    savepath=None,
    improvement_ylim=(None, 70),
    ymin=None, ymax=None, lead_max_hours = None, lead_col = None
):
    """
    Recreates the two-panel RMSE figure (levels + % improvement) and overlays
    avg target precipitation (month >= min_month_for_precip) on a twin y-axis in the *top* panel.
    """


    ns_to_hours = lambda s: int(s.split()[0]) // (3600 * 1_000_000_000)

    if 'forecast_horizon_hours' in df_rmse.columns:
        lead_col = 'forecast_horizon_hours'
    
    try:
        df_rmse[lead_col] = df_rmse[lead_col].apply(ns_to_hours)
    except AttributeError as ae:
        pass

    # --- summarize RMSE with HAC-inflated CI ---
    rmse_summary = summarize_rmse_by_model(
        df_rmse,
        lead_col=lead_col,
        init_col="init_date",
        model_col="model",
        rmse_col="rmse",
        alpha=alpha,
        max_lag=max_lag
    )

    if rmse_summary.empty:
        raise ValueError("RMSE summary is empty.")

    # --- compute average precip vs lead (targets, month >= 8) ---
    precip_df = compute_avg_target_precip_by_lead(
        rolled_dir=rolled_dir,
        target_prefix=TARGET_PREFIX,
        target_var=TARGET_VAR,
        min_month=min_month_for_precip,
        drop_first_step=True,  # to match RMSE alignment
        lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX
    )
    print(precip_df)

    print("Setting up models")

    if lead_max_hours is not None:
        rmse_summary = rmse_summary[rmse_summary["lead"] <= lead_max_hours].copy()
        precip_df = precip_df[precip_df["lead_hours"] <= lead_max_hours].copy()
    print(precip_df)


    # --- set up models to plot (keep same ordering style) ---
    all_models = list(rmse_summary["model"].unique())
    models = all_models if models_to_plot is None else [m for m in models_to_plot]
    print(models, all_models, models_to_plot)
    if baseline_model not in models:
        raise ValueError(f"Baseline model '{baseline_model}' not found in RMSE summary.")
    models = [baseline_model] + [m for m in models if m != baseline_model]

    leads = np.sort(rmse_summary["lead"].unique())
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    cmap = {}
    for i, m in enumerate(models):
        if custom_colors and m in custom_colors:
            cmap[m] = custom_colors[m]
        else:
            cmap[m] = "black" if m == baseline_model else cycle[(i-1) % len(cycle)]

    # --- figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # ===== top: RMSE curves with HAC CIs =====
    for m in models:
        g = rmse_summary[rmse_summary["model"] == m].sort_values("lead")
        if g.empty: continue
        x = g["lead"].values
        y = g["rmse_hat"].values
        lo = g["ci_lo"].values
        hi = g["ci_hi"].values
        lw = 3.0 if m == baseline_model else 2.0
        z = 5 if m == baseline_model else 4
        label = models_rename.get(m, m) if models_rename else m
        ax1.plot(x, y, linestyle="-", linewidth=lw, color=cmap[m], label=label, zorder=z)
        ax1.fill_between(x, lo, hi, color=cmap[m], alpha=(0.12 if m == baseline_model else 0.16), linewidth=0)

    ax1.set_title(title, fontsize=20, pad=10)
    ax1.set_ylabel("RMSE", fontsize=18)
    ax1.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.4)

    if ymin is not None or ymax is not None:
        ax1.set_ylim(bottom=ymin if ymin is not None else ax1.get_ylim()[0],
                     top=ymax if ymax is not None else ax1.get_ylim()[1])

    # ----- twin y-axis: Average precipitation vs lead -----
    ax1b = ax1.twinx()
    ax1b.plot(
        precip_df["lead_hours"].values,
        precip_df["mean_precip_mm_6h"].values,
        linestyle="--",
        linewidth=4.2,
        color="darkgreen",
        label="Average Precip India",
        alpha=0.9

    )
    lo_ppt = 0
    hi_ppt = None
    ax1b.set_ylim(bottom=lo_ppt if lo_ppt is not None else ax1b.get_ylim()[0],
                     top=hi_ppt if hi_ppt is not None else ax1b.get_ylim()[1])
    
    ax1b.set_ylabel("Avg precip (mm / 6h)", fontsize=20, color="darkgreen")
    ax1b.tick_params(axis="y", labelcolor="darkgreen", labelsize=18)

    # combine legends (RMSE lines + precip)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, title="", ncols=2, fontsize=21, loc="upper left")

    # ===== bottom: % improvement vs baseline =====
    base = rmse_summary[rmse_summary["model"] == baseline_model].sort_values("lead")[["lead", "rmse_hat"]]
    base = base.rename(columns={"rmse_hat": "rmse_base"})
    for m in models:
        if m == baseline_model: continue
        g = rmse_summary[rmse_summary["model"] == m].sort_values("lead")[["lead", "rmse_hat"]]
        merged = pd.merge(base, g, on="lead", how="inner")
        denom = merged["rmse_base"].replace(0, np.nan)
        improvement = 100.0 * (merged["rmse_base"] - merged["rmse_hat"]) / denom
        ax2.plot(merged["lead"].values, improvement.values, linestyle="-", linewidth=2.0,
                 color=cmap[m], label=(models_rename.get(m, m) if models_rename else m))

    ax2.axhline(0, color="black", linewidth=1, linestyle="--")
    ax2.set_xlabel("Lead time (hours)", fontsize=18)
    ax2.set_ylabel("% Improvement", fontsize=18)
    ax2.grid(True, axis="x", linestyle="--", linewidth=0.9, alpha=0.4)
    ax2.legend(fontsize=18)

    # ticks every 24h on both panels
    print(f"Leads: {leads}")
    if leads[0].dtype == 'str' and 'nano' in leads[0]:
        ns_to_hours = lambda s: int(s.split()[0]) // (3600 * 1_000_000_000)
        leads = list(map(ns_to_hours, leads))
        leads = np.array(leads)


    ax1.set_xticks(np.arange(leads.min()-12, leads.max() + 1, 24))
    ax2.set_xticks(np.arange(leads.min()-12, leads.max() + 1, 24))

    ax1.tick_params(axis="both", labelsize=18)
    ax2.tick_params(axis="both", labelsize=18)

    if improvement_ylim is not None:
        lo, hi = improvement_ylim
        ax2.set_ylim(bottom=lo if lo is not None else ax2.get_ylim()[0],
                     top=hi if hi is not None else ax2.get_ylim()[1])

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    # return both tables if you want to reuse them
    return rmse_summary, precip_df

# ----------------- main -----------------

def main():

    OUT_PNG = "./plots/rmse_levels_precip6h.png"
    parser = argparse.ArgumentParser(description="Compute RMSE CSV and plot mean RMSE + % improvement with HAC CIs.")
    parser.add_argument("--input_csv", default=None)  # if provided, skip RMSE computation
    parser.add_argument("--rolled_dir", default=ROLLED_DIR)
    parser.add_argument("--hres_path", default=HRES_PATH)
    parser.add_argument("--finetuned_prefix", default=FINETUNED_PREFIX)
    parser.add_argument("--out_csv", default=OUT_CSV)
    parser.add_argument("--out_png", default=OUT_PNG)
    parser.add_argument("--baseline_model", default="base")  # choose "HRES" or "Graphcast_Base"
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max_lag", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    savepath = args.input_csv
    if savepath and os.path.exists(savepath):
        df = pd.read_csv(savepath)
        print("Reading existing")
    else:
        df = compute_rmse_table(rolled_dir=args.rolled_dir,
                            hres_path=args.hres_path,
                            finetuned_prefix=args.finetuned_prefix)
        df.to_csv(args.out_csv, index=False)
        print(f"Saved RMSE table -> {args.out_csv}")

    # 2) Summarize & plot

    df_rmse = df

    # Optional: colors & rename (edit as desired)
    custom_colors = {
        "HRES": "tab:purple",
        "base": "orange",
        "finetuned": "tab:blue",
        "fine": "tab:blue",
        "fine_val": "tab:cyan",
        "fine_6h": "tab:cyan",
    }
    models_rename = {
        "HRES": "HRES",
        "base": "Base",
        "finetuned": "Finetuned",
        "fine": "Finetuned",
        "fine_val": "Finetuned-Val",
        "fine_6h": "Finetuned-6h",
    }

    plot_path = args.out_png
    rmse_summary, precip_by_lead = plot_rmse_levels_with_precip(
        df_rmse,
        rolled_dir=ROLLED_DIR,
        baseline_model="Graphcast_Finetuned1",
        models_to_plot=None,
        custom_colors=custom_colors,
        models_rename=models_rename,
        title="RMSE Comparison and % Improvement + Avg Precip",
        alpha=0.20,
        max_lag=0,
        min_month_for_precip=7,
        savepath=plot_path,
        improvement_ylim=(None, 80),
        ymin=None, ymax=None, lead_col='forecast_horizon_hours'
    )



    # rmse_summary = summarize_rmse_by_model(
    #     df,
    #     lead_col="lead_hours",
    #     init_col="init_date",
    #     model_col="model",
    #     rmse_col="rmse",
    #     alpha=args.alpha,
    #     max_lag=args.max_lag
    # )



    # # choose models to display
    # models_to_plot = ["base", "finetuned"]

    # plot_rmse_levels(
    #     rmse_summary,
    #     baseline_model=args.baseline_model,
    #     models_to_plot=models_to_plot,
    #     custom_colors=custom_colors,
    #     title="RMSE Comparison and % Improvement (HAC-inflated CIs)",
    #     ymin=0, ymax=14,
    #     savepath=args.out_png,
    #     models_rename=models_rename,
    #     improvement_ylim=(None, 80),
    # )
    print(f"Saved plot -> {args.out_png}")

if __name__ == "__main__":
    main()



"""
python hres_avg_rmse_ci.py --input_csv skill_score_India_2025-08-3113-25-44.csv --out_png ./plots/rmse_levels_precip_cached.png


"""