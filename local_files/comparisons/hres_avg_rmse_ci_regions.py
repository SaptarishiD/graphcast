import os
import glob
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy import stats

# ----------------- config (edit as needed) -----------------
# Define a dictionary of locations and their corresponding coordinates
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

ROLLED_DIR = "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds"
HRES_PATH = "/Datastorage/divij.khaitan_asp25/forecasts_2014/hres_forecasts_20140601_20140930.nc"

TARGET_PREFIX = "target_init_"
BASE_PREFIX = "base_init_"
FINETUNED_PREFIX = "finetuned_init_"

TARGET_VAR = "total_precipitation_6hr"
HRES_VAR = "tp"

OUT_CSV = "./rmse_by_init_and_lead_hres_base_fine201408onward.csv"
OUT_PNG = "./plots/rmse_levels.png"
OUT_DIR = "./results"  # Directory to save the output NetCDF files

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
    return float(np.sqrt(max(k2, 1.0)))

def _select_bbox(da, lat_min, lat_max, lon_min, lon_max):
    lat = da["lat"]; lon = da["lon"]
    lat_sel = lat.where((lat >= lat_min) & (lat <= lat_max), drop=True)
    lon_sel = lon.where((lon >= lon_min) & (lon <= lon_max), drop=True)
    return da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

def create_regridder(src_da, dst_da):
    return xe.Regridder(src_da, dst_da, method="bilinear", reuse_weights=False)

def regrid_each_step(da_step_lat_lon, regridder, step_dim="step"):
    out_list = []
    for i in range(da_step_lat_lon.sizes[step_dim]):
        out_i = regridder(da_step_lat_lon.isel({step_dim: i}))
        out_list.append(out_i)
    out = xr.concat(out_list, dim=step_dim)
    if step_dim in da_step_lat_lon.coords:
        out = out.assign_coords({step_dim: da_step_lat_lon[step_dim]})
    return out

def _standardize_pred_da(ds, var=TARGET_VAR):
    da = ds[var]
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
        lead_hours = np.arange(step_coord.size) * 6.0
    return lead_hours

def _extract_init_date_from_filename(path, prefix):
    base = os.path.basename(path)
    if not base.startswith(prefix):
        return None
    stem = base[len(prefix):].rsplit(".nc", 1)[0]
    return stem.split()[0]

# ----------------- RMSE computation (multi-init loop) -----------------

def compute_rmse_table(rolled_dir=ROLLED_DIR,
                       hres_path=HRES_PATH,
                       finetuned_prefix=FINETUNED_PREFIX,
                       locations=LOCATIONS):
    logging.info("Discovering initialization dates from targets...")
    target_paths = sorted(glob.glob(os.path.join(rolled_dir, f"{TARGET_PREFIX}*.nc")))
    if not target_paths:
        raise FileNotFoundError(f"No target files found: {rolled_dir}/{TARGET_PREFIX}*.nc")

    init_dates = sorted({_extract_init_date_from_filename(p, TARGET_PREFIX)
                          for p in target_paths if _extract_init_date_from_filename(p, TARGET_PREFIX)})
    logging.info(f"Found {len(init_dates)} init dates.")

    logging.info("Opening HRES...")
    hres = xr.open_dataset(hres_path, decode_timedelta=True)
    hres_tp = hres[HRES_VAR]

    logging.info("Creating regridder (HRES grid -> target grid)...")
    first_target = xr.open_dataset(os.path.join(rolled_dir, f"{TARGET_PREFIX}{init_dates[0]} 00:00:00.nc"), decode_timedelta=True)
    target_grid_da = _standardize_pred_da(first_target, var=TARGET_VAR).isel(step=0)
    regridder = create_regridder(hres_tp.isel(time=0, step=1), target_grid_da)

    rows = []
    print(init_dates)

    for date_str in tqdm(init_dates, desc="Processing init dates"):
        init_iso = f"{date_str}T00:00:00"
        init_dt64 = np.datetime64(init_iso)

        tgt_path = os.path.join(rolled_dir, f"{TARGET_PREFIX}{date_str} 00:00:00.nc")
        if not os.path.exists(tgt_path):
            logging.warning(f"Missing target for {date_str}, skipping.")
            continue
        target_ds = xr.open_dataset(tgt_path, decode_timedelta=True)
        gt = _standardize_pred_da(target_ds, var=TARGET_VAR)

        try:
            hres_sel = hres_tp.sel(time=np.datetime64(init_iso)).isel(step=slice(1, None)) * 1000.0
        except Exception as e:
            logging.warning(f"HRES selection failed for {date_str}: {e}")
            continue
        hres_rg = regrid_each_step(hres_sel, regridder, step_dim="step")

        common_steps = min(hres_rg.sizes["step"], gt.sizes["step"] - 1)
        if common_steps <= 0:
            logging.warning(f"No overlapping steps for {date_str}, skipping.")
            continue
        gt_use = gt.isel(step=slice(1, 1 + common_steps))
        hres_use = hres_rg.isel(step=slice(0, common_steps))

        base_da = None
        base_path = os.path.join(rolled_dir, f"{BASE_PREFIX}{date_str} 00:00:00.nc")
        if os.path.exists(base_path):
            base_ds = xr.open_dataset(base_path, decode_timedelta=True)
            base_da = _standardize_pred_da(base_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        ft_da = None
        finetuned_path = os.path.join(rolled_dir, f"{finetuned_prefix}{date_str} 00:00:00.nc")
        if os.path.exists(finetuned_path):
            ft_ds = xr.open_dataset(finetuned_path, decode_timedelta=True)
            ft_da = _standardize_pred_da(ft_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        lead_hours = _compute_lead_hours_from_coord(gt_use["step"], init_dt64)

        for region_name, region_coords in locations.items():
            lat_min, lat_max = region_coords["lat_min"], region_coords["lat_max"]
            lon_min, lon_max = region_coords["lon_min"], region_coords["lon_max"]

            def rmse_and_values(pred_da, truth_da):
                pred_box = _select_bbox(pred_da, lat_min, lat_max, lon_min, lon_max)
                truth_box = _select_bbox(truth_da, lat_min, lat_max, lon_min, lon_max)
                diff = pred_box - truth_box
                rmse = np.sqrt((diff ** 2).mean(dim=["lat", "lon"]))
                return rmse, pred_box, truth_box

            # HRES
            rmse_hres, pred_hres, actual_precip = rmse_and_values(hres_use, gt_use)
            for lh, val, pred, actual in zip(lead_hours, rmse_hres.values, pred_hres.values, actual_precip.values):
                rows.append({"model": "HRES", "init_date": date_str,
                             "forecast_horizon_hours": float(lh), "rmse": float(val),
                             "predicted_precip": pred.tolist(), "actual_precip": actual.tolist(), "region": region_name})

            # Base
            if base_da is not None:
                rmse_base, pred_base, _ = rmse_and_values(base_da, gt_use)
                for lh, val, pred, actual in zip(lead_hours, rmse_base.values, pred_base.values, actual_precip.values):
                    rows.append({"model": "Graphcast_Base", "init_date": date_str,
                                 "forecast_horizon_hours": float(lh), "rmse": float(val),
                                 "predicted_precip": pred.tolist(), "actual_precip": actual.tolist(), "region": region_name})

            # Finetuned
            if ft_da is not None:
                rmse_ft, pred_ft, _ = rmse_and_values(ft_da, gt_use)
                for lh, val, pred, actual in zip(lead_hours, rmse_ft.values, pred_ft.values, actual_precip.values):
                    rows.append({"model": "Graphcast_Finetuned1", "init_date": date_str,
                                 "forecast_horizon_hours": float(lh), "rmse": float(val),
                                 "predicted_precip": pred.tolist(), "actual_precip": actual.tolist(), "region": region_name})

    df = pd.DataFrame(rows).sort_values(["model", "init_date", "forecast_horizon_hours", "region"])
    return df


def rollout_forecasts_for_dates(initialization_dates, locations=LOCATIONS, rolled_dir=ROLLED_DIR, hres_path=HRES_PATH, finetuned_prefix=FINETUNED_PREFIX, out_dir=OUT_DIR):
    """
    Rolls out forecasts for a list of particular initialization dates for each of the mentioned regions and saves the data.
    """
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Rolling out forecasts for {len(initialization_dates)} dates and {len(locations)} locations.")

    hres = xr.open_dataset(hres_path, decode_timedelta=True)
    hres_tp = hres[HRES_VAR]

    first_target_path = glob.glob(os.path.join(rolled_dir, f"{TARGET_PREFIX}*.nc"))[0]
    first_target = xr.open_dataset(first_target_path, decode_timedelta=True)
    target_grid_da = _standardize_pred_da(first_target, var=TARGET_VAR).isel(step=0)
    regridder = create_regridder(hres_tp.isel(time=0, step=1), target_grid_da)

    for date_str in tqdm(initialization_dates, desc="Processing init dates"):
        init_iso = f"{date_str}T00:00:00"
        init_dt64 = np.datetime64(init_iso)

        tgt_path = os.path.join(rolled_dir, f"{TARGET_PREFIX}{date_str} 00:00:00.nc")
        if not os.path.exists(tgt_path):
            logging.warning(f"Missing target for {date_str}, skipping.")
            continue
        target_ds = xr.open_dataset(tgt_path, decode_timedelta=True)
        gt = _standardize_pred_da(target_ds, var=TARGET_VAR)

        try:
            hres_sel = hres_tp.sel(time=np.datetime64(init_iso)).isel(step=slice(1, None)) * 1000.0
        except Exception as e:
            logging.warning(f"HRES selection failed for {date_str}: {e}")
            continue
        hres_rg = regrid_each_step(hres_sel, regridder, step_dim="step")

        common_steps = min(hres_rg.sizes["step"], gt.sizes["step"] - 1)
        if common_steps <= 0:
            logging.warning(f"No overlapping steps for {date_str}, skipping.")
            continue
        gt_use = gt.isel(step=slice(1, 1 + common_steps))
        hres_use = hres_rg.isel(step=slice(0, common_steps))

        base_da = None
        base_path = os.path.join(rolled_dir, f"{BASE_PREFIX}{date_str} 00:00:00.nc")
        if os.path.exists(base_path):
            base_ds = xr.open_dataset(base_path, decode_timedelta=True)
            base_da = _standardize_pred_da(base_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        ft_da = None
        finetuned_path = os.path.join(rolled_dir, f"{finetuned_prefix}{date_str} 00:00:00.nc")
        if os.path.exists(finetuned_path):
            ft_ds = xr.open_dataset(finetuned_path, decode_timedelta=True)
            ft_da = _standardize_pred_da(ft_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        for region_name, region_coords in locations.items():
            lat_min, lat_max = region_coords["lat_min"], region_coords["lat_max"]
            lon_min, lon_max = region_coords["lon_min"], region_coords["lon_max"]

            gt_region = _select_bbox(gt_use, lat_min, lat_max, lon_min, lon_max)
            hres_region = _select_bbox(hres_use, lat_min, lat_max, lon_min, lon_max)
            if base_da is not None:
                base_region = _select_bbox(base_da, lat_min, lat_max, lon_min, lon_max)
            if ft_da is not None:
                ft_region = _select_bbox(ft_da, lat_min, lat_max, lon_min, lon_max)

            output_ds = xr.Dataset({
                'actual_precip': gt_region,
                'predicted_precip_hres': hres_region,
            })
            if base_da is not None:
                output_ds['predicted_precip_base'] = base_region
            if ft_da is not None:
                output_ds['predicted_precip_finetuned'] = ft_region

            output_filename = os.path.join(out_dir, f"forecast_{date_str}_{region_name}.nc")
            output_ds.to_netcdf(output_filename)
            logging.info(f"Saved forecast data to {output_filename}")


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

        lo = max(0.0, m - zcrit * se)
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

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    cmap = {}
    for i, m in enumerate(models):
        if custom_colors and m in custom_colors:
            cmap[m] = custom_colors[m]
        else:
            cmap[m] = "black" if m == baseline_model else cycle[(i-1) % len(cycle)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

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
    ax1.set_ylabel("RMSE (mm / 6h)", fontsize=18)
    ax1.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
    ax1.legend(title="", ncols=2, fontsize=21)

    if ymin is not None or ymax is not None:
        ax1.set_ylim(bottom=ymin if ymin is not None else ax1.get_ylim()[0],
                     top=ymax if ymax is not None else ax1.get_ylim()[1])

    base = summary_df[summary_df["model"] == baseline_model].sort_values("lead")[["lead", "rmse_hat"]]
    base = base.rename(columns={"rmse_hat": "rmse_base"})

    for m in models:
        if m == baseline_model: continue
        g = summary_df[summary_df["model"] == m].sort_values("lead")[["lead", "rmse_hat"]]
        merged = pd.merge(base, g, on="lead", how="inner")
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

    step = 24 if (leads.max() - leads.min() >= 96) else max(6, int(np.median(np.diff(leads))) if len(leads) > 1 else 6)
    ax1.set_xticks(np.arange(leads.min()-12, leads.max() + 1, 24))
    ax2.set_xticks(np.arange(leads.min()-12, leads.max() + 1, 24))

    ax1.tick_params(axis="both", labelsize=18)
    ax2.tick_params(axis="both", labelsize=18)

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")


# ---- NEW: average target precipitation per lead (month >= 8) ----
def compute_avg_target_precip_by_lead(
    rolled_dir=ROLLED_DIR,
    target_prefix=TARGET_PREFIX,
    target_var=TARGET_VAR,
    min_month=8,
    drop_first_step=True,
    lat_min=6, lat_max=38,
    lon_min=65, lon_max=95
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - lead_hours (float)
      - mean_precip_mm_6h (float): India-box mean of target precipitation, averaged across init dates
      - n_inits (int): number of init dates contributing to that lead
    """
    import glob

    all_targets = sorted(glob.glob(os.path.join(rolled_dir, f"{target_prefix}*.nc")))
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
        if (m < min_month and y != 2014) or (y !=2014):
            continue

        init_iso = f"{date_str}T00:00:00"
        print(init_iso)
        init_dt64 = np.datetime64(init_iso)

        ds = xr.open_dataset(p,decode_timedelta=True)
        da = _standardize_pred_da(ds, var=target_var)
        if drop_first_step and da.sizes.get("step", 0) > 0:
            da = da.isel(step=slice(1, None))

        da_box = _select_bbox(da, lat_min, lat_max, lon_min, lon_max)
        step_mean = da_box.mean(dim=["lat", "lon"])
        lead_hours = _compute_lead_hours_from_coord(step_mean["step"], init_dt64)
        df_i = pd.DataFrame({
            "init_date": date_str,
            "lead_hours": lead_hours.astype(float),
            "precip_mm_6h": step_mean.values.astype(float),
        })
        recs.append(df_i)

    if not recs:
        raise RuntimeError("No eligible target files (month >= min_month) found.")

    df_all = pd.concat(recs, ignore_index=True)
    df_all["lead_hours"] = np.rint(df_all["lead_hours"]).astype(float)

    agg = (df_all
           .groupby("lead_hours", as_index=False)
           .agg(mean_precip_mm_6h=("precip_mm_6h", "mean"),
                n_inits=("init_date", "nunique")))
    agg = agg.sort_values("lead_hours").reset_index(drop=True)
    return agg


# ---- NEW: RMSE + CI plot with a secondary axis for avg precip ----
def plot_rmse_levels_with_precip(
    df_rmse,
    rolled_dir=ROLLED_DIR,
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
    ymin=None, ymax=None, lead_max_hours = None, lead_col = None,
    region="india"
):
    """
    Recreates the two-panel RMSE figure and overlays avg target precipitation on a twin y-axis in the *top* panel.
    """
    # ns_to_hours = lambda s: int(s.split()[0]) // (3600 * 1_000_000_000)

    if df_rmse[lead_col].dtype == object:
        ns_to_hours = lambda s: int(s.split()[0]) // (3600 * 1_000_000_000)
        df_rmse[lead_col] = df_rmse[lead_col].apply(ns_to_hours)

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

    region_coords = LOCATIONS.get(region)
    if not region_coords:
        raise ValueError(f"Region '{region}' not found in LOCATIONS dictionary.")

    precip_df = compute_avg_target_precip_by_lead(
        rolled_dir=rolled_dir,
        target_prefix=TARGET_PREFIX,
        target_var=TARGET_VAR,
        min_month=min_month_for_precip,
        drop_first_step=True,
        lat_min=region_coords["lat_min"], lat_max=region_coords["lat_max"],
        lon_min=region_coords["lon_min"], lon_max=region_coords["lon_max"]
    )
    print(precip_df)

    print("Setting up models")

    if lead_max_hours is not None:
        rmse_summary = rmse_summary[rmse_summary["lead"] <= lead_max_hours].copy()
        precip_df = precip_df[precip_df["lead_hours"] <= lead_max_hours].copy()
    print(precip_df)
    print(rmse_summary)

    all_models = list(rmse_summary["model"].unique())
    models = all_models if models_to_plot is None else [m for m in models_to_plot if m in all_models]
    print(models)
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

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
    ax1.set_ylabel("RMSE (mm / 6h)", fontsize=18)
    ax1.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.4)

    if ymin is not None or ymax is not None:
        ax1.set_ylim(bottom=ymin if ymin is not None else ax1.get_ylim()[0],
                     top=ymax if ymax is not None else ax1.get_ylim()[1])

    ax1b = ax1.twinx()
    ax1b.plot(
        precip_df["lead_hours"].values,
        precip_df["mean_precip_mm_6h"].values,
        linestyle="--",
        linewidth=4.2,
        color="darkgreen",
        label=f"Average Precip {region.capitalize()}",
        alpha=0.9

    )
    lo_ppt = 0
    hi_ppt = None
    ax1b.set_ylim(bottom=lo_ppt if lo_ppt is not None else ax1b.get_ylim()[0],
                     top=hi_ppt if hi_ppt is not None else ax1b.get_ylim()[1])

    ax1b.set_ylabel("Avg precip (mm / 6h)", fontsize=20, color="darkgreen")
    ax1b.tick_params(axis="y", labelcolor="darkgreen", labelsize=18)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, title="", ncols=2, fontsize=21, loc="upper left")

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

    if isinstance(leads[0], str) and 'nano' in leads[0]:
        ns_to_hours = lambda s: int(s.split()[0]) // (3600 * 1_000_000_000)
        leads = list(map(ns_to_hours, leads))
        leads = np.array(leads)

    ax1.set_xticks(np.arange(min(leads)-12, max(leads) + 1, 24))
    ax2.set_xticks(np.arange(min(leads)-12, max(leads) + 1, 24))

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
    return rmse_summary, precip_df

# ----------------- main -----------------

def main():

    OUT_PNG = "./plots/rmse_levels_precip.png"
    parser = argparse.ArgumentParser(description="Compute RMSE CSV and plot mean RMSE + % improvement with HAC CIs.")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--rolled_dir", default=ROLLED_DIR)
    parser.add_argument("--hres_path", default=HRES_PATH)
    parser.add_argument("--finetuned_prefix", default=FINETUNED_PREFIX)
    parser.add_argument("--out_csv", default=OUT_CSV)
    parser.add_argument("--out_png", default=OUT_PNG)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--baseline_model", default="base")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max_lag", type=int, default=2)
    parser.add_argument("--rollout_dates", nargs='+', help="List of initialization dates for forecast rollout (YYYY-MM-DD)")
    parser.add_argument("--region", default="india", help="Region to plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.rollout_dates:
        rollout_forecasts_for_dates(args.rollout_dates,
                                    locations=LOCATIONS,
                                    rolled_dir=args.rolled_dir,
                                    hres_path=args.hres_path,
                                    finetuned_prefix=args.finetuned_prefix,
                                    out_dir=args.out_dir)
        return

    savepath = args.input_csv
    if savepath and os.path.exists(savepath):
        df = pd.read_csv(savepath)
        print("Reading existing CSV")
    else:
        df = compute_rmse_table(rolled_dir=args.rolled_dir,
                            hres_path=args.hres_path,
                            finetuned_prefix=args.finetuned_prefix,
                            locations=LOCATIONS)
        df.to_csv(args.out_csv, index=False)
        print(f"Saved RMSE table -> {args.out_csv}")

    df_rmse = df[df["region"] == args.region]

    custom_colors = {
        "HRES": "tab:purple",
        "base": "orange",
        "finetuned": "tab:blue",
        "fine": "tab:blue",
        "fine_val": "tab:cyan",
    }
    models_rename = {
        "HRES": "HRES",
        "base": "Base",
        "finetuned": "Finetuned",
        "fine": "Finetuned",
        "fine_val": "Finetuned-Val",
    }

    plot_path = args.out_png.replace('.png', f'_{args.region}.png')
    rmse_summary, precip_by_lead = plot_rmse_levels_with_precip(
        df_rmse,
        rolled_dir=ROLLED_DIR,
        baseline_model="Graphcast_Base",
        models_to_plot=["base", "fine", "fine_val"],
        custom_colors=custom_colors,
        models_rename=models_rename,
        title=f"RMSE Comparison and % Improvement + Avg Precip ({args.region.capitalize()})",
        alpha=0.20,
        max_lag=0,
        min_month_for_precip=8,
        savepath=plot_path,
        improvement_ylim=(None, 80),
        ymin=None, ymax=None, lead_col='forecast_horizon_hours',
        region=args.region
    )

    print(f"Saved plot -> {plot_path}")

if __name__ == "__main__":
    main()