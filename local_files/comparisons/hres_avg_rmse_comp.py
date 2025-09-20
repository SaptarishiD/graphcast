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

# ----------------- config -----------------
LAT_MIN, LAT_MAX = 6, 38
LON_MIN, LON_MAX = 65, 95

ROLLED_DIR = "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds"
HRES_PATH = "/Datastorage/divij.khaitan_asp25/forecasts_2014/hres_forecasts_20140601_20140831.nc"

TARGET_PREFIX = "target_init_"
BASE_PREFIX = "base_init_"
FINETUNED_PREFIX = "fine_init_"  # change if your finetuned prefix differs
TARGET_VAR = "total_precipitation_6hr"  # in mm (assumed)
HRES_VAR = "tp"                         # in meters, will convert to mm

OUT_CSV = "./rmse_by_init_and_lead.csv"
OUT_PNG = "./mean_rmse_by_lead.png"

# ------------------------------------------

def _select_bbox(da, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX):
    """Robust lat/lon subsetting regardless of ascending/descending coords."""
    lat = da["lat"]
    lon = da["lon"]
    lat_sel = lat.where((lat >= lat_min) & (lat <= lat_max), drop=True)
    lon_sel = lon.where((lon >= lon_min) & (lon <= lon_max), drop=True)
    return da.sel(lat=lat_sel, lon=lon_sel)

def create_regridder(src_da, dst_da):
    """Create an xESMF regridder from src_da grid to dst_da grid."""
    # Expect 2D lat/lon or 1D; xESMF can infer from DataArray
    regridder = xe.Regridder(src_da, dst_da, method="bilinear", reuse_weights=False)
    return regridder

def regrid_each_step(da_step_lat_lon, regridder, step_dim="step"):
    """Apply xESMF regridder over all steps."""
    out_list = []
    for i in range(da_step_lat_lon.sizes[step_dim]):
        out_i = regridder(da_step_lat_lon.isel({step_dim: i}))
        out_list.append(out_i)
    out = xr.concat(out_list, dim=step_dim)
    # carry step coordinate if present
    if step_dim in da_step_lat_lon.coords:
        out = out.assign_coords({step_dim: da_step_lat_lon[step_dim]})
    return out

def _standardize_pred_da(ds, var=TARGET_VAR):
    """
    Extracts the precipitation DA from a predictions/target file.
    - Drops 'batch' if present.
    - Renames 'time' -> 'step' for alignment with HRES.
    """
    da = ds[var]
    for dim in ["batch"]:
        if dim in da.dims:
            da = da.squeeze(dim, drop=True)
    if "time" in da.dims:
        da = da.rename({"time": "step"})
    return da

def _compute_lead_hours_from_coord(step_coord, init_dt64):
    """
    Convert a 'step' coordinate into lead hours.
    Handles timedelta64 and datetime64 coordinates.
    Falls back to 6-hour spacing if neither.
    """
    vals = step_coord.values
    if np.issubdtype(vals.dtype, np.timedelta64):
        lead_hours = (vals / np.timedelta64(1, "h")).astype(float)
    elif np.issubdtype(vals.dtype, np.datetime64):
        lead_hours = ((vals - init_dt64) / np.timedelta64(1, "h")).astype(float)
    else:
        # Fallback: assume 6h cadence
        lead_hours = np.arange(step_coord.size) * 6.0
    return lead_hours

def _extract_init_date_from_filename(path, prefix):
    """
    Expect filenames like '{prefix}{YYYY-MM-DD} 00:00:00.nc'.
    Returns 'YYYY-MM-DD'.
    """
    base = os.path.basename(path)
    if not base.startswith(prefix):
        return None
    stem = base[len(prefix):].rsplit(".nc", 1)[0]  # e.g., '2014-08-01 00:00:00'
    date_str = stem.split()[0]                     # '2014-08-01'
    return date_str

def main(
    rolled_dir=ROLLED_DIR,
    hres_path=HRES_PATH,
    out_csv=OUT_CSV,
    out_png=OUT_PNG,
    finetuned_prefix=FINETUNED_PREFIX,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- discover all init dates from target files ---
    target_paths = sorted(glob.glob(os.path.join(rolled_dir, f"{TARGET_PREFIX}*.nc")))
    if not target_paths:
        raise FileNotFoundError(f"No target files found matching {rolled_dir}/{TARGET_PREFIX}*.nc")

    init_dates = []
    for p in target_paths:
        d = _extract_init_date_from_filename(p, TARGET_PREFIX)
        if d is not None:
            init_dates.append(d)
    init_dates = sorted(set(init_dates))

    logging.info(f"Found {len(init_dates)} initialization dates.")

    # --- open HRES once ---
    logging.info("Opening HRES dataset...")
    hres = xr.open_dataset(hres_path, decode_timedelta=True)
    hres_tp = hres[HRES_VAR]  # (time, step, lat, lon), meters per 6h

    # --- build regridder once using the first target grid ---
    logging.info("Creating regridder from HRES grid to target grid...")
    # Open first target file to define destination grid
    first_target = xr.open_dataset(
        os.path.join(rolled_dir, f"{TARGET_PREFIX}{init_dates[0]} 00:00:00.nc"), decode_timedelta=True
    )
    target_grid_da = _standardize_pred_da(first_target, var=TARGET_VAR).isel(step=0)
    regridder = create_regridder(hres_tp.isel(time=0, step=1), target_grid_da)

    rows = []

    for date_str in tqdm(init_dates, desc="Processing init dates"):
        init_iso = f"{date_str}T00:00:00"
        init_dt64 = np.datetime64(init_iso)

        # --- load target (ground truth) ---
        tgt_path = os.path.join(rolled_dir, f"{TARGET_PREFIX}{date_str} 00:00:00.nc")
        if not os.path.exists(tgt_path):
            logging.warning(f"Missing target file for {date_str}, skipping.")
            continue
        target_ds = xr.open_dataset(tgt_path, decode_timedelta=True)
        gt = _standardize_pred_da(target_ds, var=TARGET_VAR)

        # HRES for this init date: select and drop step 0 (often analysis/zero accum)
        try:
            hres_sel = hres_tp.sel(time=np.datetime64(init_iso)).isel(step=slice(1, None)) * 1000.0  # m->mm
        except Exception as e:
            logging.warning(f"HRES selection failed for {date_str}: {e}. Skipping this date.")
            continue

        # Regrid HRES to target grid step-by-step
        hres_rg = regrid_each_step(hres_sel, regridder, step_dim="step")

        # Standardize GT to 'step' and (optionally) align steps by slicing
        # Keep steps that exist in both to avoid mismatched lengths
        common_steps = min(hres_rg.sizes["step"], gt.sizes["step"])
        gt_use = gt.isel(step=slice(1, 1 + common_steps))  # drop its first step to match HRES starting at 6h
        hres_use = hres_rg.isel(step=slice(0, common_steps))

        # --- base model ---
        base_path = os.path.join(rolled_dir, f"{BASE_PREFIX}{date_str} 00:00:00.nc")
        if os.path.exists(base_path):
            base_ds = xr.open_dataset(base_path, decode_timedelta=True)
            base_da = _standardize_pred_da(base_ds, var=TARGET_VAR)
            base_da = base_da.isel(step=slice(1, 1 + common_steps))
        else:
            base_da = None
            logging.warning(f"Missing base file for {date_str}.")

        # --- finetuned model ---
        finetuned_path = os.path.join(rolled_dir, f"{finetuned_prefix}{date_str} 00:00:00.nc")
        if os.path.exists(finetuned_path):
            ft_ds = xr.open_dataset(finetuned_path, decode_timedelta=True)
            ft_da = _standardize_pred_da(ft_ds, var=TARGET_VAR)
            ft_da = ft_da.isel(step=slice(1, 1 + common_steps))
        else:
            ft_da = None
            logging.warning(f"Missing finetuned file for {date_str}.")

        # --- compute lead hours from GT step coord (datetime) ---
        lead_hours = _compute_lead_hours_from_coord(gt_use["step"], init_dt64)

        # --- compute RMSE over India for each model ---
        def rmse_over_india(pred_da, truth_da):
            pred_box = _select_bbox(pred_da)
            truth_box = _select_bbox(truth_da)
            diff = pred_box - truth_box
            rmse_da = np.sqrt((diff ** 2).mean(dim=["lat", "lon"]))
            return rmse_da

        # HRES
        rmse_hres = rmse_over_india(hres_use, gt_use).values  # shape (step,)
        for lh, val in zip(lead_hours, rmse_hres):
            rows.append({"model": "HRES", "init_date": date_str, "lead_hours": float(lh), "rmse": float(val)})

        # Base
        if base_da is not None:
            rmse_base = rmse_over_india(base_da, gt_use).values
            for lh, val in zip(lead_hours, rmse_base):
                rows.append({"model": "base", "init_date": date_str, "lead_hours": float(lh), "rmse": float(val)})

        # Finetuned
        if ft_da is not None:
            rmse_ft = rmse_over_india(ft_da, gt_use).values
            for lh, val in zip(lead_hours, rmse_ft):
                rows.append({"model": "finetuned", "init_date": date_str, "lead_hours": float(lh), "rmse": float(val)})

    # --- assemble DataFrame and save ---
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No RMSE rows were computed. Check inputs / file patterns.")
    df.sort_values(["model", "init_date", "lead_hours"], inplace=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved RMSE table to {out_csv}")

    # --- mean RMSE by lead time per model & plot ---
    df_mean = df.groupby(["model", "lead_hours"], as_index=False)["rmse"].mean()
    plt.figure(figsize=(8, 5))
    for model, sub in df_mean.groupby("model"):
        if model == 'HRES':
            continue
        sub = sub.sort_values("lead_hours")
        plt.plot(sub["lead_hours"], sub["rmse"], marker="o", label=model)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("RMSE (mm / 6h)")
    plt.title("Mean RMSE vs Lead Time (averaged over initialization dates)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"Saved plot to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RMSE per lead time across inits for HRES/base/finetuned.")
    parser.add_argument("--rolled_dir", default=ROLLED_DIR, help="Directory containing target/base/finetuned files.")
    parser.add_argument("--hres_path", default=HRES_PATH, help="Path to HRES forecasts netCDF.")
    parser.add_argument("--out_csv", default=OUT_CSV, help="Where to save the tidy RMSE CSV.")
    parser.add_argument("--out_png", default=OUT_PNG, help="Where to save the plot.")
    parser.add_argument("--finetuned_prefix", default=FINETUNED_PREFIX, help="Filename prefix for finetuned files.")
    args = parser.parse_args()
    plotonly=True
    if plotonly:
        df = pd.read_csv('rmse_by_init_and_lead_hres_base_fine201408onward.csv')
            # --- mean RMSE by lead time per model & plot ---
        df_mean = df.groupby(["model", "lead_hours"], as_index=False)["rmse"].mean()
        plt.figure(figsize=(8, 5))
        for model, sub in df_mean.groupby("model"):
            if model == 'HRES':
                continue
            sub = sub.sort_values("lead_hours")
            plt.plot(sub["lead_hours"], sub["rmse"], marker="o", label=model)
        plt.xlabel("Lead time (hours)")
        plt.ylabel("RMSE (mm / 6h)")
        plt.title("Mean RMSE vs Lead Time (averaged over initialization dates)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('test_rmse_hres.png', dpi=220)
        print(f"Saved plot to {'test_rmse_hres.png'}")
        exit()

    main(
        rolled_dir=args.rolled_dir,
        hres_path=args.hres_path,
        out_csv=args.out_csv,
        out_png=args.out_png,
        finetuned_prefix=args.finetuned_prefix,
    )
