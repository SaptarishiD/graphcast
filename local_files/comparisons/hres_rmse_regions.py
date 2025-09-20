#!/usr/bin/env python3
# regional_rmse_and_ts.py

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

# --- shapefile / masking deps ---
import geopandas as gpd
import rioxarray  # noqa: F401 (needed for .rio accessor)
from rasterio.features import geometry_mask
from shapely.geometry import mapping

# ----------------- config (edit as needed) -----------------
LAT_MIN, LAT_MAX = 6, 38
LON_MIN, LON_MAX = 65, 95

ROLLED_DIR = "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds"
HRES_PATH = "/Datastorage/divij.khaitan_asp25/forecasts_2014/hres_forecasts_20140601_20140930.nc"

year = 2014
TARGET_PREFIX = f"target_init_"
BASE_PREFIX = f"base_init_"
FINETUNED_PREFIX = f"fine_init_"  # change if your finetuned prefix differs

TARGET_VAR = "total_precipitation_6hr"  # (assumed) mm / 6h
HRES_VAR = "tp"                         # meters / 6h -> convert to mm

# Default output paths
OUT_CSV_RMSE_REG = "./rmse_by_init_lead_by_region.csv"
OUT_CSV_TS = "./avg_rain_by_date_lead_by_region.csv"
OUT_DIR_TS_PLOTS = "./plots/avg_rain_timeseries"
OUT_DIR_RMSE_PLOTS = "./plots/rmse_by_region"

# Regions list as requested
REGIONS_LIST = ['Central_Northeast', 'Hilly_Regions', 'Northeast', 'Northwest', 'South_Peninsular', 'West_Central']

# ----------------- utilities (selection, grids, etc.) -----------------

def _select_bbox(da, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX):
    """Quick bbox crop (assumes 'lat','lon' exist)."""
    if "latitude" in da.coords or "longitude" in da.coords:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    lat = da["lat"]; lon = da["lon"]
    lat_sel = lat.where((lat >= lat_min) & (lat <= lat_max), drop=True)
    lon_sel = lon.where((lon >= lon_min) & (lon <= lon_max), drop=True)
    return da.sel(lat=lat_sel, lon=lon_sel)

def create_regridder(src_da, dst_da):
    """xesmf regridder: src (lat,lon) -> dst (lat,lon)."""
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
    """Make (step, lat, lon) with coord 'step'."""
    da = ds[var]
    if "batch" in da.dims:
        da = da.squeeze("batch", drop=True)
    if "time" in da.dims:
        da = da.rename({"time": "step"})
    # ensure lat/lon names
    if "latitude" in da.coords or "longitude" in da.coords:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
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

def _nearest_step_index(lead_hours_array, desired_hours, tol_hours=3.1):
    idx = int(np.argmin(np.abs(lead_hours_array - desired_hours)))
    if np.abs(lead_hours_array[idx] - desired_hours) > tol_hours:
        return None
    return idx

# ----------------- YOUR shapefile-based regional mask -----------------
# (verbatim pattern; used as-is inside helper that builds a boolean mask)
def mask_dbase_regions(dbase, region_name='West_Central'):
    if 'lon' in dbase.coords:
        dbase = dbase.rename({'lon': 'longitude', 'lat': 'latitude'})
    indiashp_path_regions = "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog"

    shapefile_path = indiashp_path_regions
    regions = ['Central_Northeast', 'Hilly_Regions', 'Northeast', 'Northwest', 'South_Peninsular', 'West_Central']
    region_shp = gpd.read_file(shapefile_path, layer=region_name).to_crs(epsg=4326)

    if (dbase.longitude > 180).any():
        dbase = dbase.assign_coords(longitude=(((dbase.longitude + 180) % 360) - 180))
        dbase = dbase.sortby("longitude")

    dbase = dbase.rio.write_crs("EPSG:4326", inplace=True)

    mask_region = geometry_mask(
        [mapping(region_shp['geometry'].iloc[0])],     # List of geometries
        out_shape=(len(dbase.latitude), len(dbase.longitude)),
        transform=dbase.rio.transform(),
        invert=True,                           # Areas *inside* geometry == True
        all_touched=False
    )

    mask_xr = xr.DataArray(mask_region, coords={"latitude": dbase.latitude, "longitude": dbase.longitude}, dims=("latitude", "longitude"))

    dbase_masked = xr.where(mask_xr, dbase, 0)
    dbase_masked = dbase_masked.assign_coords(longitude=((dbase_masked.longitude + 360) % 360))
    dbase_masked = dbase_masked.sortby("longitude")

    return dbase_masked

# ----------------- helpers to build boolean region masks on target grid -----------------

def _region_bool_mask_on_grid(lat, lon, region_name):
    """
    Build a boolean (lat,lon) mask on the given grid using your shapefile function.
    We pass a 2D ones array to get the mask, then convert back to ('lat','lon') and -180..180.
    """
    tmpl = xr.DataArray(np.ones((lat.size, lon.size), dtype=float),
                        coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    # rename to latitude/longitude for your function
    tmpl_ll = tmpl.rename({"lat": "latitude", "lon": "longitude"})
    masked = mask_dbase_regions(tmpl_ll, region_name=region_name)  # 0 outside, 1 inside
    # back to lat/lon and unwrap longitudes to [-180, 180]
    masked = masked.rename({"latitude": "lat", "longitude": "lon"})
    masked = masked.assign_coords(lon=(((masked["lon"] + 180) % 360) - 180)).sortby("lon")
    # align to original tmpl grid (exact match expected)
    masked = masked.reindex(lat=tmpl.lat, lon=tmpl.lon)
    return xr.where(masked > 0.5, True, False)

def _regional_mean_2d(da2d, bool_mask):
    """Mean over region for a single 2D slice (lat,lon)."""
    return da2d.where(bool_mask).mean(dim=["lat", "lon"], skipna=True)

# ----------------- RMSE by region -----------------

def compute_rmse_table_by_region(rolled_dir=ROLLED_DIR,
                                 hres_path=HRES_PATH,
                                 finetuned_prefix=FINETUNED_PREFIX,
                                 regions=REGIONS_LIST,
                                 out_csv=OUT_CSV_RMSE_REG):
    """
    Writes tidy CSV with columns:
      region, model, init_date, lead_hours, rmse
    """
    logging.info("Discovering initialization dates (per-region RMSE)...")
    target_paths = sorted(glob.glob(os.path.join(rolled_dir, f"{TARGET_PREFIX}*.nc")))
    if not target_paths:
        raise FileNotFoundError(f"No target files found: {rolled_dir}/{TARGET_PREFIX}*.nc")
    init_dates = sorted({ _extract_init_date_from_filename(p, TARGET_PREFIX)
                          for p in target_paths if _extract_init_date_from_filename(p, TARGET_PREFIX) })
    logging.info(f"Found {len(init_dates)} init dates.")

    logging.info("Opening HRES & building regridder...")
    hres = xr.open_dataset(hres_path, decode_timedelta=True)
    hres_tp = hres[HRES_VAR]  # meters / 6h

    first_target = xr.open_dataset(os.path.join(rolled_dir, f"{TARGET_PREFIX}{init_dates[0]} 00:00:00.nc"),
                                   decode_timedelta=True)
    gt0 = _standardize_pred_da(first_target, var=TARGET_VAR)
    target_grid_da = gt0.isel(step=0)  # (lat,lon)
    regridder = create_regridder(hres_tp.isel(time=0, step=1), target_grid_da)

    # Precompute region boolean masks on target grid
    region_masks = {r: _region_bool_mask_on_grid(target_grid_da.lat, target_grid_da.lon, r) for r in regions}

    rows = []

    for date_str in tqdm(init_dates, desc="RMSE by region"):

        y, m, _ = map(int, date_str.split("-"))
        if not (y == 2014 and m >= 8):
                print("Skipping not year 2014 and month 8 onward")
                continue
        init_iso = f"{date_str}T00:00:00"
        init_dt64 = np.datetime64(init_iso)

        tgt_path = os.path.join(rolled_dir, f"{TARGET_PREFIX}{date_str} 00:00:00.nc")
        if not os.path.exists(tgt_path):
            continue
        target_ds = xr.open_dataset(tgt_path, decode_timedelta=True)
        gt = _standardize_pred_da(target_ds, var=TARGET_VAR)   # (step, lat, lon)

        # HRES -> mm & align; drop step 0
        try:
            hres_sel = hres_tp.sel(time=np.datetime64(init_iso)).isel(step=slice(1, None)) * 1000.0
        except Exception as e:
            logging.warning(f"HRES selection failed for {date_str}: {e}")
            continue
        hres_rg = regrid_each_step(hres_sel, regridder, step_dim="step")

        # Align steps across GT & HRES; drop GT step 0 to match indices
        common_steps = min(hres_rg.sizes["step"], gt.sizes["step"] - 1)
        if common_steps <= 0:
            continue
        gt_use   = gt.isel(step=slice(1, 1 + common_steps))
        hres_use = hres_rg.isel(step=slice(0, common_steps))

        # Base & Finetuned (optional)
        base_da = None
        base_path = os.path.join(rolled_dir, f"{BASE_PREFIX}{date_str} 00:00:00.nc")
        if os.path.exists(base_path):
            base_ds = xr.open_dataset(base_path, decode_timedelta=True)
            base_da = _standardize_pred_da(base_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        ft_da = None
        finetuned_path = os.path.join(rolled_dir, f"{FINETUNED_PREFIX}{date_str} 00:00:00.nc")
        if os.path.exists(finetuned_path):
            ft_ds = xr.open_dataset(finetuned_path, decode_timedelta=True)
            ft_da = _standardize_pred_da(ft_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        lead_hours = _compute_lead_hours_from_coord(gt_use["step"], init_dt64)

        # For each region, compute RMSE over region (mask)
        for region, mask in region_masks.items():
            # (step, lat, lon) -> apply mask via .where
            def region_rmse(pred, truth):
                diff2 = (pred - truth) ** 2
                # mean over region only
                mse = diff2.where(mask).mean(dim=["lat", "lon"], skipna=True)
                return np.sqrt(mse).astype(float)

            # HRES
            rmse_hres = region_rmse(hres_use, gt_use).values
            for lh, val in zip(lead_hours, rmse_hres):
                rows.append({"region": region, "model": "HRES", "init_date": date_str,
                             "lead_hours": float(lh), "rmse": float(val)})

            # Base
            if base_da is not None:
                rmse_base = region_rmse(base_da, gt_use).values
                for lh, val in zip(lead_hours, rmse_base):
                    rows.append({"region": region, "model": "Graphcast_Base", "init_date": date_str,
                                 "lead_hours": float(lh), "rmse": float(val)})

            # Finetuned
            if ft_da is not None:
                rmse_ft = region_rmse(ft_da, gt_use).values
                for lh, val in zip(lead_hours, rmse_ft):
                    rows.append({"region": region, "model": "Graphcast_Finetuned1", "init_date": date_str,
                                 "lead_hours": float(lh), "rmse": float(val)})

    df = pd.DataFrame(rows).sort_values(["region","model","init_date","lead_hours"])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved per-region RMSE -> {out_csv}")
    return df

# ----------------- Regionwise avg rainfall time series @ 24h/72h -----------------

def compute_regionwise_avg_rain_timeseries(
    rolled_dir=ROLLED_DIR,
    hres_path=HRES_PATH,
    finetuned_prefix=FINETUNED_PREFIX,
    regions=REGIONS_LIST,
    desired_leads=(24.0, 72.0),
    out_csv=OUT_CSV_TS
):
    """
    Writes CSV with columns:
      region, init_date, lead_hours, truth_mm_6h, hres_mm_6h, base_mm_6h, finetuned_mm_6h
    Each value is a spatial **mean** over the region at the nearest step to desired lead.
    NOTE: If you want daily accumulations (sum of 6-hourly slices over 24h window),
          replace the single-step extraction with a sum over 4 consecutive steps.
    """
    logging.info("Discovering initialization dates (regionwise TS)...")
    target_paths = sorted(glob.glob(os.path.join(rolled_dir, f"{TARGET_PREFIX}*.nc")))
    if not target_paths:
        raise FileNotFoundError(f"No target files found: {rolled_dir}/{TARGET_PREFIX}*.nc")
    init_dates = sorted({ _extract_init_date_from_filename(p, TARGET_PREFIX)
                          for p in target_paths if _extract_init_date_from_filename(p, TARGET_PREFIX) })
    logging.info(f"Found {len(init_dates)} init dates.")

    logging.info("Opening HRES & building regridder (TS)...")
    hres = xr.open_dataset(hres_path, decode_timedelta=True)
    hres_tp = hres[HRES_VAR]  # meters / 6h

    first_target = xr.open_dataset(os.path.join(rolled_dir, f"{TARGET_PREFIX}{init_dates[0]} 00:00:00.nc"),
                                   decode_timedelta=True)
    gt0 = _standardize_pred_da(first_target, var=TARGET_VAR)
    target_grid_da = gt0.isel(step=0)
    regridder = create_regridder(hres_tp.isel(time=0, step=1), target_grid_da)

    # Precompute region masks on this grid
    region_masks = {r: _region_bool_mask_on_grid(target_grid_da.lat, target_grid_da.lon, r) for r in regions}

    rows = []

    for date_str in tqdm(init_dates, desc="Avg rainfall TS (24/72h)"):
        init_iso = f"{date_str}T00:00:00"
        init_dt64 = np.datetime64(init_iso)

        tgt_path = os.path.join(rolled_dir, f"{TARGET_PREFIX}{date_str} 00:00:00.nc")
        if not os.path.exists(tgt_path):
            continue
        target_ds = xr.open_dataset(tgt_path, decode_timedelta=True)
        gt = _standardize_pred_da(target_ds, var=TARGET_VAR)   # (step, lat, lon)

        # HRES -> mm & align; drop step 0
        try:
            hres_sel = hres_tp.sel(time=np.datetime64(init_iso)).isel(step=slice(1, None)) * 1000.0
        except Exception as e:
            logging.warning(f"HRES selection failed for {date_str}: {e}")
            continue
        hres_rg = regrid_each_step(hres_sel, regridder, step_dim="step")

        # Align steps
        common_steps = min(hres_rg.sizes["step"], gt.sizes["step"] - 1)
        if common_steps <= 0:
            continue
        gt_use   = gt.isel(step=slice(1, 1 + common_steps))
        hres_use = hres_rg.isel(step=slice(0, common_steps))

        # Base & Finetuned (optional)
        base_da = None
        base_path = os.path.join(rolled_dir, f"{BASE_PREFIX}{date_str} 00:00:00.nc")
        if os.path.exists(base_path):
            base_ds = xr.open_dataset(base_path, decode_timedelta=True)
            base_da = _standardize_pred_da(base_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        ft_da = None
        finetuned_path = os.path.join(rolled_dir, f"{FINETUNED_PREFIX}{date_str} 00:00:00.nc")
        if os.path.exists(finetuned_path):
            ft_ds = xr.open_dataset(finetuned_path, decode_timedelta=True)
            ft_da = _standardize_pred_da(ft_ds, var=TARGET_VAR).isel(step=slice(1, 1 + common_steps))

        lead_hours = _compute_lead_hours_from_coord(gt_use["step"], init_dt64).astype(float)

        # For each desired lead, take nearest step and compute regional means
        for region, mask in region_masks.items():
            for desired in desired_leads:
                idx = _nearest_step_index(lead_hours, desired, tol_hours=3.1)
                if idx is None:
                    continue
                truth_mean = float(_regional_mean_2d(gt_use.isel(step=idx), mask))
                hres_mean  = float(_regional_mean_2d(hres_use.isel(step=idx), mask))
                base_mean = np.nan
                if base_da is not None:
                    base_mean = float(_regional_mean_2d(base_da.isel(step=idx), mask))
                finetuned_mean = np.nan
                if ft_da is not None:
                    finetuned_mean = float(_regional_mean_2d(ft_da.isel(step=idx), mask))

                rows.append({
                    "region": region,
                    "init_date": date_str,
                    "lead_hours": float(lead_hours[idx]),
                    "truth_mm_6h": truth_mean,
                    "hres_mm_6h": hres_mean,
                    "base_mm_6h": base_mean,
                    "finetuned_mm_6h": finetuned_mean,
                })

    df = pd.DataFrame(rows).sort_values(["region","lead_hours","init_date"])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved avg rain time series -> {out_csv}")
    return df

# ----------------- Plotting: truth vs predictions per region & lead -----------------

def plot_avg_rainfall_timeseries(
    csv_path=OUT_CSV_TS,
    out_dir=OUT_DIR_TS_PLOTS,
    regions_to_plot=None,            # None -> all regions in CSV
    leads_to_plot=(24.0, 72.0),
    custom_colors=None               # {"Truth":"black","HRES":"tab:purple","Graphcast_Base":"orange","Graphcast_Finetuned1":"tab:blue"}
):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Timeseries CSV is empty.")
    df["init_date"] = pd.to_datetime(df["init_date"])
    regions = sorted(df["region"].unique()) if regions_to_plot is None else [r for r in regions_to_plot if r in set(df["region"])]
    os.makedirs(out_dir, exist_ok=True)

    if custom_colors is None:
        custom_colors = {"Truth":"black","HRES":"tab:purple","Graphcast_Base":"orange","Graphcast_Finetuned1":"tab:blue"}

    for region in regions:
        sub_r = df[df["region"] == region].copy()
        for lead in leads_to_plot:
            # pick nearest recorded lead (robust to 24 vs 24.0)
            lead_vals = np.sort(sub_r["lead_hours"].unique())
            if lead_vals.size == 0:
                continue
            lead_idx = int(np.argmin(np.abs(lead_vals - lead)))
            lead_sel = float(lead_vals[lead_idx])
            g = sub_r[np.isclose(sub_r["lead_hours"], lead_sel)].sort_values("init_date")

            if g.empty:
                continue

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(g["init_date"], g["truth_mm_6h"], label="Truth", color=custom_colors.get("Truth","black"), linewidth=3.0)
            ax.plot(g["init_date"], g["hres_mm_6h"], label="HRES", color=custom_colors.get("HRES","tab:purple"), linewidth=2.4)
            if "base_mm_6h" in g:
                ax.plot(g["init_date"], g["base_mm_6h"], label="Graphcast_Base", color=custom_colors.get("Graphcast_Base","orange"), linewidth=2.4)
            if "finetuned_mm_6h" in g:
                ax.plot(g["init_date"], g["finetuned_mm_6h"], label="Graphcast_Finetuned1", color=custom_colors.get("Graphcast_Finetuned1","tab:blue"), linewidth=2.4)

            ax.set_title(f"{region} • Total Precip (spatial mean) • Lead {int(round(lead_sel))} h", fontsize=18, pad=10)
            ax.set_xlabel("Init Date", fontsize=16)
            ax.set_ylabel("Precip (mm / 6h)", fontsize=16)
            ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.4)
            ax.legend(fontsize=14, ncols=2)
            fig.autofmt_xdate(rotation=30)
            plt.tight_layout()

            fname = f"avg_rain_ts_{region.replace(' ', '_')}_lead{int(round(lead_sel))}h.png"
            savepath = os.path.join(out_dir, fname)
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
            plt.close(fig)

# ----------------- (Optional) quick RMSE plot per region (no CI, simple lines) -----------------

def plot_rmse_by_region_simple(
    csv_path=OUT_CSV_RMSE_REG,
    out_dir=OUT_DIR_RMSE_PLOTS,
    regions_to_plot=None,
    models=("Graphcast_Base","Graphcast_Finetuned1"),
    models_rename=None
):
    df = pd.read_csv(csv_path)
    df["init_date"] = pd.to_datetime(df["init_date"])
    print("making dirs")
    os.makedirs(out_dir, exist_ok=True)
    regions = sorted(df["region"].unique()) if regions_to_plot is None else [r for r in regions_to_plot if r in set(df["region"])]

    for region in regions:
        sub = df[df["region"] == region]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        for m in models:
            label = models_rename.get(m, m) if models_rename else m
            g = sub[sub["model"] == m].groupby("lead_hours", as_index=False)["rmse"].mean().sort_values("lead_hours")
            if g.empty: continue
            ax.plot(g["lead_hours"], g["rmse"]*1000, label=label, linewidth=2.2)
        ax.set_title(f"RMSE by Lead • {region}", fontsize=18, pad=10)
        ax.set_xlabel("Lead (hours)", fontsize=16)
        ax.set_ylabel("RMSE (mm / 6h)", fontsize=16)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.1)
        ax.legend(fontsize=14, ncols=2)
        # print(g['lead_hours'])
        ax.set_xticks(np.arange(g["lead_hours"].min()-12, g["lead_hours"].max()+1, 24))
        ymin = 1.1
        ymax = None
        if (ymin is not None) or (ymax is not None):
            # ymin = None
            ax.set_ylim(
                bottom=ymin if ymin is not None else ax.get_ylim()[0]+1,
                top=ymax if ymax is not None else ax.get_ylim()[1]+1.0,
            )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rmse_{region.replace(' ','_')}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

# ----------------- main -----------------

def main():
    parser = argparse.ArgumentParser(description="RMSE per region (shapefile masks) + total precip time series at 1/3 days.")
    parser.add_argument("--rolled_dir", default=ROLLED_DIR)
    parser.add_argument("--hres_path", default=HRES_PATH)
    parser.add_argument("--finetuned_prefix", default=FINETUNED_PREFIX)
    parser.add_argument("--regions", default=",".join(REGIONS_LIST), help="Comma-separated region names (must match shapefile layers).")
    parser.add_argument("--rmse_csv", default=OUT_CSV_RMSE_REG)
    parser.add_argument("--ts_csv", default=OUT_CSV_TS)
    parser.add_argument("--ts_plot_dir", default=OUT_DIR_TS_PLOTS)
    parser.add_argument("--rmse_plot_dir", default=OUT_DIR_RMSE_PLOTS)
    parser.add_argument("--ts_leads", default="24,72", help="Comma-separated lead hours for time series (e.g., '24,72').")
    parser.add_argument("--do_rmse_plots", action="store_true", help="Also emit simple RMSE plots per region.")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    leads = [float(x.strip()) for x in args.ts_leads.split(",") if x.strip()]

    just_plot_rmse = True


        # 3) (Optional) quick RMSE overview plots per region
    if args.do_rmse_plots and just_plot_rmse:
        print("Existing")
        if os.path.exists(args.rmse_csv):
            plot_rmse_by_region_simple(
                csv_path=args.rmse_csv,
                out_dir=args.rmse_plot_dir,
                regions_to_plot=regions,
                models_rename={"Graphcast_Base":"Base", "Graphcast_Finetuned1":"Finetuned"}
            )
        return


    # 1) RMSE per region
    compute_rmse_table_by_region(
        rolled_dir=args.rolled_dir,
        hres_path=args.hres_path,
        finetuned_prefix=args.finetuned_prefix,
        regions=regions,
        out_csv=args.rmse_csv
    )

    # 2) Regionwise avg rainfall TS (1/3 day leads) + plots
    compute_regionwise_avg_rain_timeseries(
        rolled_dir=args.rolled_dir,
        hres_path=args.hres_path,
        finetuned_prefix=args.finetuned_prefix,
        regions=regions,
        desired_leads=leads,
        out_csv=args.ts_csv
    )

    plot_avg_rainfall_timeseries(
        csv_path=args.ts_csv,
        out_dir=args.ts_plot_dir,
        regions_to_plot=regions,
        leads_to_plot=leads,
        custom_colors={"Truth":"black","HRES":"tab:purple","Graphcast_Base":"blue","Graphcast_Finetuned1":"tab:orange"}
    )

    # 3) (Optional) quick RMSE overview plots per region
    if args.do_rmse_plots:
        plot_rmse_by_region_simple(
            csv_path=args.rmse_csv,
            out_dir=args.rmse_plot_dir,
            regions_to_plot=regions
        )

    logging.info("Done.")

if __name__ == "__main__":
    main()
