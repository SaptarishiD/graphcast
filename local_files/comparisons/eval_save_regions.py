#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Region-wise RMSE computation, summarization & plotting for precipitation forecasts.

- Discovers initialization dates from target NetCDFs
- Regrids HRES to target grid
- Computes RMSE per model vs ground truth by lead hours
- Adds 'region' column (from LOCATIONS BBoxes)
- Summarizes (per-horizon means + 95% CI, overall means, integrated RMSE)
- Plots per-region lines and a faceted grid

Requirements: xarray, numpy, pandas, matplotlib, xesmf, tqdm, scipy (optional)
"""

import os
import re
import glob
import math
import logging
import argparse
from typing import Iterable, Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------- Configuration (edit as needed) --------------------

# File/glob naming
TARGET_PREFIX   = "target_"          # files like: target_YYYY-MM-DD 00:00:00.nc
BASE_PREFIX     = "base_"            # files like: base_YYYY-MM-DD 00:00:00.nc
FINETUNED_PREFIX_DEFAULT = "fine_"  # prefix for finetuned files
ROLLED_DIR = "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds"
HRES_PATH = "/Datastorage/divij.khaitan_asp25/forecasts_2014/hres_forecasts_20140601_20140930.nc"

# Variable names
HRES_VAR   = "tp"                    # ERA5/HRES total precipitation (meters per 6h)
TARGET_VAR = "total_precipitation_6hr"  # target variable in your rolled target files

# Default India-wide bbox for helpers (used only if region not provided)
LAT_MIN, LAT_MAX = 6.0, 38.0
LON_MIN, LON_MAX = 65.0, 95.0

# Plot output directory
PLOT_DIR_DEFAULT = "./plots/rmse_regions"

# -------------------- Regions dictionary --------------------

LOCATIONS = {
    "india": {"lat_min": 6.0,  "lat_max": 38.0, "lon_min": 65.0, "lon_max": 95.0},

    "mumbai":    {"lat_min": 18.0, "lat_max": 20.0, "lon_min": 72.0, "lon_max": 74.0},
    "delhi":     {"lat_min": 28.0, "lat_max": 29.0, "lon_min": 76.0, "lon_max": 78.0},
    "kolkata":   {"lat_min": 22.0, "lat_max": 23.0, "lon_min": 88.0, "lon_max": 89.0},
    "chennai":   {"lat_min": 12.0, "lat_max": 13.0, "lon_min": 80.0, "lon_max": 81.0},
    "bengaluru": {"lat_min": 12.0, "lat_max": 13.0, "lon_min": 77.0, "lon_max": 78.0},
    "hyderabad": {"lat_min": 17.0, "lat_max": 18.0, "lon_min": 78.0, "lon_max": 79.0},
    "ahmedabad": {"lat_min": 23.0, "lat_max": 24.0, "lon_min": 72.0, "lon_max": 73.0},
    "pune":      {"lat_min": 18.0, "lat_max": 19.0, "lon_min": 73.0, "lon_max": 74.0},
    "jaipur":    {"lat_min": 26.0, "lat_max": 27.0, "lon_min": 75.0, "lon_max": 76.0},
    "lucknow":   {"lat_min": 26.0, "lat_max": 27.0, "lon_min": 80.0, "lon_max": 81.0},
    "bhopal":    {"lat_min": 23.0, "lat_max": 24.0, "lon_min": 77.0, "lon_max": 78.0},
    "guwahati":  {"lat_min": 26.0, "lat_max": 27.0, "lon_min": 91.0, "lon_max": 92.0},
    "srinagar":  {"lat_min": 34.0, "lat_max": 35.0, "lon_min": 74.0, "lon_max": 75.0},
    "thiruvananthapuram": {"lat_min": 8.0, "lat_max": 9.0, "lon_min": 76.0, "lon_max": 77.0},
}

# -------------------- Utils --------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)

def _extract_init_date_from_filename(path: str, prefix: str) -> Optional[str]:
    base = os.path.basename(path)
    if not base.startswith(prefix):
        return None
    stem = base[len(prefix):].rsplit(".nc", 1)[0]  # 'YYYY-MM-DD 00:00:00'
    return stem.split()[0]   

def _ensure_lat_lon_coords(da: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """
    Normalize lat/lon coordinate names to 'lat' and 'lon'.
    """
    rename_map = {}
    for cand, std in (("latitude", "lat"), ("Longitude", "lon"), ("longitude", "lon"), ("Latitude", "lat")):
        if cand in da.coords and std not in da.coords:
            rename_map[cand] = std
        if cand in da.dims and std not in da.dims:
            rename_map[cand] = std
    return da.rename(rename_map) if rename_map else da

def _standardize_pred_da(ds: xr.Dataset, var: str) -> xr.DataArray:
    """
    Return DataArray with dims at least ('step','lat','lon').
    If a 'time' or 'base_time' dimension exists (single init per file), drop it.
    """
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset variables: {list(ds.data_vars)}")
    da = ds[var]
    da = _ensure_lat_lon_coords(da)

    # If both 'time' and 'base_time' exist, prefer 'base_time'
    time_like_dims = [d for d in ("base_time") if d in da.dims]
    if time_like_dims:
        tdim = time_like_dims[0]
        if da.sizes[tdim] == 1:
            da = da.isel({tdim: 0})
        else:
            # If multiple times, assume first corresponds to init in filename
            da = da.isel({tdim: 0})

    # Ensure step dimension exists
    print(da)
    print(da.dims)
    if "step" not in da.dims:
        da = da.rename({"time": "step"}) if "time" in da.dims else None
        # raise ValueError("Expected 'step' dimension in data array for forecast leads.")

    # Enforce order (step, lat, lon) if present
    dims = list(da.dims)
    for req in ("step", "lat", "lon"):
        if req not in dims:
            raise ValueError(f"Expected '{req}' dimension in data array; got dims {dims}")
    return da.transpose("step", "lat", "lon")

def create_regridder(src_da: xr.DataArray, target_da: xr.DataArray, method: str = "bilinear") -> xe.Regridder:
    """
    Create xESMF regridder from src_da grid to target_da grid.
    """
    src_grid = _ensure_lat_lon_coords(src_da)
    tgt_grid = _ensure_lat_lon_coords(target_da)
    regridder = xe.Regridder(src_grid, tgt_grid, method=method, reuse_weights=True)
    return regridder

def regrid_each_step(src_da: xr.DataArray, regridder: xe.Regridder, step_dim: str = "step") -> xr.DataArray:
    """
    Apply regridder to each step independently. Returns DataArray on target grid with same 'step' coordinate.
    """
    src_da = _ensure_lat_lon_coords(src_da)
    steps = src_da[step_dim]
    out_list = []
    for i in range(src_da.sizes[step_dim]):
        s = src_da.isel({step_dim: i})
        r = regridder(s)  # -> lat/lon like target grid
        out_list.append(r.expand_dims({step_dim: [steps.values[i]]}))
    out = xr.concat(out_list, dim=step_dim)
    return out

def _compute_lead_hours_from_coord(step_coord: xr.DataArray, init_dt64: np.datetime64) -> List[float]:
    """
    Convert xarray 'step' coordinate (timedelta64 or similar) to hours (float).
    """
    # If already numeric hours, return those
    vals = step_coord.values
    if np.issubdtype(vals.dtype, np.number):
        return [float(v) for v in vals]

    # If timedelta64
    if np.issubdtype(vals.dtype, np.timedelta64):
        hours = (vals / np.timedelta64(1, "h")).astype(float)
        return [float(h) for h in hours]

    # If datetime-like (rare for 'step'), compute difference from init
    try:
        hours = ((vals - init_dt64) / np.timedelta64(1, "h")).astype(float)
        return [float(h) for h in hours]
    except Exception:
        raise ValueError(f"Unrecognized 'step' coordinate dtype: {vals.dtype}")

def _to_float_list(arr_like) -> List[float]:
    """
    Materialize possible Dask/xarray arrays to python floats (avoids .item on Dask scalars).
    """
    a = arr_like
    if hasattr(a, "compute"):
        try:
            a = a.compute()
        except Exception:
            pass
    a = np.asarray(a)
    return [float(v) for v in a.ravel()]

# -------------------- Region selection --------------------

def _select_bbox(da: xr.DataArray, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX) -> xr.DataArray:
    da = _ensure_lat_lon_coords(da)
    lat, lon = da["lat"], da["lon"]
    # Handle descending coords safely
    lat_start, lat_end = (lat_max, lat_min) if lat[0] > lat[-1] else (lat_min, lat_max)
    lon_start, lon_end = (lon_max, lon_min) if lon[0] > lon[-1] else (lon_min, lon_max)
    return da.sel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end))

def _select_region(da: xr.DataArray, region_name: str, locations: Dict[str, Dict[str, float]] = LOCATIONS) -> xr.DataArray:
    if region_name not in locations:
        raise KeyError(f"Unknown region '{region_name}'. Available: {list(locations.keys())}")
    bbox = locations[region_name]
    return _select_bbox(da, **bbox)

# -------------------- Core RMSE table w/ region column --------------------

def compute_rmse_table(
    rolled_dir: str,
    hres_path: str,
    finetuned_prefix: str = FINETUNED_PREFIX_DEFAULT,
    regions: Iterable[str] | str = ("india",),
    target_prefix: str = TARGET_PREFIX,
    base_prefix: str = BASE_PREFIX,
    hres_var: str = HRES_VAR,
    target_var: str = TARGET_VAR,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      ['model', 'region', 'init_date', 'forecast_horizon_hours', 'rmse']

    Parameters
    ----------
    rolled_dir : str
        Directory containing target/base/finetuned NetCDFs.
    hres_path : str
        Path to HRES/ERA5 dataset (with time, step, lat, lon).
    finetuned_prefix : str
        Prefix for finetuned rolled files.
    regions : list[str] | str
        Region keys from LOCATIONS (e.g., "india", "mumbai", ...).
    target_prefix, base_prefix : str
        File prefixes for target/base files.
    hres_var, target_var : str
        Variable names in HRES and target datasets.
    """
    # Normalize regions
    if isinstance(regions, str):
        regions = (regions,)
    for r in regions:
        if r not in LOCATIONS:
            raise KeyError(f"Unknown region '{r}'. Available: {list(LOCATIONS.keys())}")

    logging.info("Discovering initialization dates from targets...")
    target_paths = sorted(glob.glob(os.path.join(rolled_dir, f"{target_prefix}*.nc")))
    if not target_paths:
        raise FileNotFoundError(f"No target files found: {rolled_dir}/{target_prefix}*.nc")

    init_dates = sorted({
        _extract_init_date_from_filename(p, target_prefix)
        for p in target_paths
        if _extract_init_date_from_filename(p, target_prefix)
    })
    print(f"Found {len(init_dates)} init dates.")
    # print(target_paths)


    logging.info("Opening HRES...")
    hres = xr.open_dataset(hres_path, decode_timedelta=True)
    if hres_var not in hres:
        raise KeyError(f"HRES variable '{hres_var}' not in dataset: {list(hres.data_vars)}")
    hres_tp = hres[hres_var]  # (time, step, lat, lon), meters / 6h

    logging.info("Creating regridder (HRES grid -> target grid)...")
    first_target = xr.open_dataset(
        os.path.join(rolled_dir, f"{target_prefix}{init_dates[0]} 00:00:00.nc"),
        decode_timedelta=True
    )
    target_grid_da = _standardize_pred_da(first_target, var=target_var).isel(step=0)
    regridder = create_regridder(hres_tp.isel(time=0, step=1), target_grid_da)

    rows = []

    for date_str in tqdm(init_dates, desc="Processing init dates"):
        init_iso = f"{date_str}T00:00:00"
        init_dt64 = np.datetime64(init_iso)

        # Ground truth
        tgt_path = os.path.join(rolled_dir, f"{target_prefix}{date_str} 00:00:00.nc")
        if not os.path.exists(tgt_path):
            logging.warning(f"Missing target for {date_str}, skipping.")
            continue
        target_ds = xr.open_dataset(tgt_path, decode_timedelta=True)
        gt = _standardize_pred_da(target_ds, var=target_var)

        # HRES -> mm, drop step 0
        try:
            hres_sel = hres_tp.sel(time=np.datetime64(init_iso)).isel(step=slice(1, None)) * 1000.0
        except Exception as e:
            logging.warning(f"HRES selection failed for {date_str}: {e}")
            continue
        hres_rg = regrid_each_step(hres_sel, regridder, step_dim="step")

        # Align steps across GT & HRES; drop GT step 0 to match 6h,12h,...
        common_steps = min(hres_rg.sizes["step"], gt.sizes["step"] - 1)
        if common_steps <= 0:
            logging.warning(f"No overlapping steps for {date_str}, skipping.")
            continue
        gt_use    = gt.isel(step=slice(1, 1 + common_steps))
        hres_use  = hres_rg.isel(step=slice(0, common_steps))

        # Base & Finetuned (optional)
        base_da = None
        base_path = os.path.join(rolled_dir, f"{base_prefix}{date_str} 00:00:00.nc")
        if os.path.exists(base_path):
            base_ds = xr.open_dataset(base_path, decode_timedelta=True)
            base_da = _standardize_pred_da(base_ds, var=target_var).isel(step=slice(1, 1 + common_steps))

        ft_da = None
        finetuned_path = os.path.join(rolled_dir, f"{finetuned_prefix}{date_str} 00:00:00.nc")
        if os.path.exists(finetuned_path):
            ft_ds = xr.open_dataset(finetuned_path, decode_timedelta=True)
            ft_da = _standardize_pred_da(ft_ds, var=target_var).isel(step=slice(1, 1 + common_steps))

        # lead hours from GT step coordinate
        lead_hours = _compute_lead_hours_from_coord(gt_use["step"], init_dt64)

        def rmse_over_region(pred_da: xr.DataArray, truth_da: xr.DataArray, region_key: str) -> xr.DataArray:
            pred_box  = _select_region(pred_da, region_key, LOCATIONS)
            truth_box = _select_region(truth_da, region_key, LOCATIONS)
            diff = pred_box - truth_box
            return np.sqrt((diff ** 2).mean(dim=["lat", "lon"]))

        # Compute & append rows per region
        for region_key in regions:
            # HRES
            rmse_hres = rmse_over_region(hres_use, gt_use, region_key)
            for lh, val in zip(lead_hours, _to_float_list(rmse_hres.values)):
                rows.append({
                    "model": "HRES",
                    "region": region_key,
                    "init_date": date_str,
                    "forecast_horizon_hours": float(lh),
                    "rmse": float(val),
                })

            # Base
            if base_da is not None:
                rmse_base = rmse_over_region(base_da, gt_use, region_key)
                for lh, val in zip(lead_hours, _to_float_list(rmse_base.values)):
                    rows.append({
                        "model": "Graphcast_Base",
                        "region": region_key,
                        "init_date": date_str,
                        "forecast_horizon_hours": float(lh),
                        "rmse": float(val),
                    })

            # Finetuned
            if ft_da is not None:
                rmse_ft = rmse_over_region(ft_da, gt_use, region_key)
                for lh, val in zip(lead_hours, _to_float_list(rmse_ft.values)):
                    rows.append({
                        "model": "Graphcast_Finetuned1",
                        "region": region_key,
                        "init_date": date_str,
                        "forecast_horizon_hours": float(lh),
                        "rmse": float(val),
                    })

    df = pd.DataFrame(rows).sort_values(["model", "region", "init_date", "forecast_horizon_hours"])
    return df

# -------------------- Summaries --------------------

def _validate_region_column(df: pd.DataFrame):
    if "region" not in df.columns:
        raise ValueError("DataFrame must include a 'region' column. "
                         "Re-run compute_rmse_table(...) with regions specified.")

def summarize_rmse_regionwise(
    df: pd.DataFrame,
    regions: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    add_improvement_vs: Optional[str] = None,
    save_csv_dir: Optional[str] = PLOT_DIR_DEFAULT
) -> Dict[str, pd.DataFrame]:
    """
    Build tidy summaries:
      - per_horizon: mean/std/sem/ci95 by (region, model, forecast_horizon_hours)
      - overall    : mean/median/min/max by (region, model)
      - integrated : trapezoidal integral across lead horizon by (region, model)
      - If add_improvement_vs is set, adds improve_abs / improve_pct vs that baseline.
    """
    _validate_region_column(df)
    regions = set(_as_list(regions) or df["region"].unique())
    models  = set(_as_list(models)  or df["model"].unique())

    sub = df[df["region"].isin(regions) & df["model"].isin(models)].copy()
    if sub.empty:
        raise ValueError("No rows match requested regions/models.")

    per_horizon = (
        sub.groupby(["region", "model", "forecast_horizon_hours"], as_index=False)
           .agg(mean_rmse=("rmse", "mean"),
                std_rmse=("rmse", "std"),
                n=("rmse", "count"))
    )
    per_horizon["sem_rmse"] = per_horizon["std_rmse"] / per_horizon["n"].clip(lower=1)**0.5
    per_horizon["ci95"] = 1.96 * per_horizon["sem_rmse"]

    overall = (
        sub.groupby(["region", "model"], as_index=False)
           .agg(mean_rmse=("rmse", "mean"),
                median_rmse=("rmse", "median"),
                min_rmse=("rmse", "min"),
                max_rmse=("rmse", "max"),
                n=("rmse", "count"))
    )

    # Integrated RMSE vs horizon (trapezoid)
    integr_rows = []
    for (reg, mod), g in per_horizon.groupby(["region", "model"]):
        g_sorted = g.sort_values("forecast_horizon_hours")
        x = g_sorted["forecast_horizon_hours"].values
        y = g_sorted["mean_rmse"].values
        auc = np.trapz(y, x) if len(x) >= 2 else np.nan
        integr_rows.append({"region": reg, "model": mod, "integrated_rmse": auc})
    integrated = pd.DataFrame(integr_rows)

    if add_improvement_vs is not None:
        # Overall improvements vs baseline region-wise
        baseline_overall = overall[overall["model"] == add_improvement_vs] \
            .set_index(["region", "model"])["mean_rmse"]
        base_by_region = baseline_overall.groupby(level=0).first()

        def _add_imp_level(df_in: pd.DataFrame, val_col: str) -> pd.DataFrame:
            out = df_in.copy()
            out["_base"] = out["region"].map(base_by_region)
            out["improve_abs"] = out["_base"] - out[val_col]
            out["improve_pct"] = (out["improve_abs"] / out["_base"]) * 100.0
            return out.drop(columns=["_base"])

        overall = _add_imp_level(overall, "mean_rmse")
        per_horizon = _add_imp_level(per_horizon, "mean_rmse")

        # Integrated improvements vs baseline's integrated (if present)
        base_integr = integrated[integrated["model"] == add_improvement_vs] \
            .set_index(["region", "model"])["integrated_rmse"]
        base_integr_by_region = base_integr.groupby(level=0).first()
        integrated["_base"] = integrated["region"].map(base_integr_by_region)
        integrated["improve_abs"] = integrated["_base"] - integrated["integrated_rmse"]
        integrated["improve_pct"] = (integrated["improve_abs"] / integrated["_base"]) * 100.0
        integrated.drop(columns=["_base"], inplace=True)

    out = {"per_horizon": per_horizon, "overall": overall, "integrated": integrated}

    if save_csv_dir:
        _ensure_dir(save_csv_dir)
        per_horizon.to_csv(os.path.join(save_csv_dir, "rmse_per_horizon.csv"), index=False)
        overall.to_csv(os.path.join(save_csv_dir, "rmse_overall.csv"), index=False)
        integrated.to_csv(os.path.join(save_csv_dir, "rmse_integrated.csv"), index=False)

    return out

# -------------------- Plotting --------------------

def plot_rmse_by_region(
    per_horizon_df: pd.DataFrame,
    output_dir: str = PLOT_DIR_DEFAULT,
    regions: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    smooth_window: int = 1,
    show_ci: bool = True,
    dpi: int = 160
):
    """
    For each region, plots RMSE vs lead hours for each model (mean across inits),
    with optional 95% CI shaded bands.
    Saves one PNG per region: rmse_{region}.png
    """
    _ensure_dir(output_dir)

    ph = per_horizon_df.copy()
    if regions:
        regions = set(_as_list(regions))
        ph = ph[ph["region"].isin(regions)]
    if models:
        models = set(_as_list(models))
        ph = ph[ph["model"].isin(models)]
    if ph.empty:
        raise ValueError("No rows to plot after filtering.")

    for reg, g in ph.groupby("region"):
        plt.figure(figsize=(8.5, 5.0), dpi=dpi)
        for mod, gm in g.groupby("model"):
            gm = gm.sort_values("forecast_horizon_hours")
            x = gm["forecast_horizon_hours"].values
            y = gm["mean_rmse"].values

            if smooth_window > 1 and len(y) >= smooth_window:
                y = pd.Series(y).rolling(smooth_window, min_periods=1, center=True).mean().values

            plt.plot(x, y, label=mod, linewidth=2)

            if show_ci and "ci95" in gm.columns and gm["n"].max() > 1:
                ci = gm["ci95"].values
                if smooth_window > 1 and len(ci) >= smooth_window:
                    ci = pd.Series(ci).rolling(smooth_window, min_periods=1, center=True).mean().values
                plt.fill_between(x, y - ci, y + ci, alpha=0.2)

        plt.title(f"RMSE vs Lead Time — {reg}")
        plt.xlabel("Forecast Horizon (hours)")
        plt.ylabel("RMSE")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"rmse_{reg}.png")
        plt.savefig(out_path)
        plt.close()

def plot_rmse_regions_grid(
    per_horizon_df: pd.DataFrame,
    output_path: str = os.path.join(PLOT_DIR_DEFAULT, "rmse_regions_grid.png"),
    max_cols: int = 3,
    models: Optional[Iterable[str]] = None,
    smooth_window: int = 1,
    show_ci: bool = False,
    dpi: int = 180
):
    """
    Faceted grid: one subplot per region, lines = models.
    """
    ph = per_horizon_df.copy()
    regions = sorted(ph["region"].unique())
    if models:
        models = set(_as_list(models))
        ph = ph[ph["model"].isin(models)]
    if ph.empty:
        raise ValueError("No rows to plot after filtering.")

    n = len(regions)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8*ncols, 3.6*nrows), dpi=dpi, squeeze=False)

    for idx, reg in enumerate(regions):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        g = ph[ph["region"] == reg]
        for mod, gm in g.groupby("model"):
            gm = gm.sort_values("forecast_horizon_hours")
            x = gm["forecast_horizon_hours"].values
            y = gm["mean_rmse"].values
            if smooth_window > 1 and len(y) >= smooth_window:
                y = pd.Series(y).rolling(smooth_window, min_periods=1, center=True).mean().values
            ax.plot(x, y, label=mod, linewidth=1.8)

            if show_ci and "ci95" in gm.columns and gm["n"].max() > 1:
                ci = gm["ci95"].values
                if smooth_window > 1 and len(ci) >= smooth_window:
                    ci = pd.Series(ci).rolling(smooth_window, min_periods=1, center=True).mean().values
                ax.fill_between(x, y - ci, y + ci, alpha=0.15)

        ax.set_title(reg)
        ax.set_xlabel("Hours")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)

    # Remove any empty axes
    for j in range(n, nrows*ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r][c])

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def summarize_and_plot_regions(
    df_with_region: pd.DataFrame,
    out_dir: str = PLOT_DIR_DEFAULT,
    baseline: Optional[str] = "HRES",
    regions: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    smooth_window: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper:
      1) build summaries (per-horizon, overall, integrated) with improvements vs baseline
      2) write CSVs
      3) save per-region plots + faceted grid
    """
    _ensure_dir(out_dir)
    summaries = summarize_rmse_regionwise(
        df_with_region,
        regions=regions,
        models=models,
        add_improvement_vs=baseline,
        save_csv_dir=out_dir
    )
    plot_rmse_by_region(
        summaries["per_horizon"],
        output_dir=out_dir,
        regions=regions,
        models=models,
        smooth_window=smooth_window,
        show_ci=True
    )
    grid_path = os.path.join(out_dir, "rmse_regions_grid.png")
    plot_rmse_regions_grid(
        summaries["per_horizon"],
        output_path=grid_path,
        models=models,
        smooth_window=smooth_window,
        show_ci=False
    )
    return summaries

# -------------------- CLI --------------------

def _build_argparser():
    p = argparse.ArgumentParser(description="Compute region-wise RMSE, summarize, and plot.")
    p.add_argument("--rolled_dir", default=ROLLED_DIR, help="Directory with rolled target/base/finetuned NetCDFs.")
    p.add_argument("--hres_path",  default=HRES_PATH, help="Path to HRES/ERA5 NetCDF with (time, step, lat/lon).")
    p.add_argument("--finetuned_prefix", default=FINETUNED_PREFIX_DEFAULT, help="Prefix for finetuned files.")
    p.add_argument("--regions", nargs="+", default=["india"], help="Region keys from LOCATIONS.")
    p.add_argument("--target_prefix", default=TARGET_PREFIX, help="Target file prefix.")
    p.add_argument("--base_prefix",   default=BASE_PREFIX,   help="Base   file prefix.")
    p.add_argument("--hres_var",      default=HRES_VAR, help="HRES variable name (e.g., 'tp').")
    p.add_argument("--target_var",    default=TARGET_VAR, help="Target variable name (e.g., 'total_precipitation_6hr').")
    p.add_argument("--out_dir",       default=PLOT_DIR_DEFAULT, help="Directory to write plots & CSVs.")
    p.add_argument("--baseline",      default="HRES", help="Baseline model for improvement calc.")
    p.add_argument("--models",        nargs="*", default=None, help="Optional subset of models to include.")
    p.add_argument("--smooth_window", type=int, default=1, help="Rolling mean window over lead hours.")
    p.add_argument("--loglevel",      default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p

def main():
    ap = _build_argparser()
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))

    df = compute_rmse_table(
        rolled_dir=args.rolled_dir,
        hres_path=args.hres_path,
        finetuned_prefix=args.finetuned_prefix,
        regions=args.regions,
        target_prefix=args.target_prefix,
        base_prefix=args.base_prefix,
        hres_var=args.hres_var,
        target_var=args.target_var,
    )
    summaries = summarize_and_plot_regions(
        df_with_region=df,
        out_dir=args.out_dir,
        baseline=args.baseline,
        regions=args.regions,
        models=args.models,
        smooth_window=args.smooth_window
    )

    # Print a short textual summary
    print("\n=== Overall mean RMSE by region/model ===")
    print(summaries["overall"].sort_values(["region","mean_rmse"]).to_string(index=False))

    print("\n=== Integrated RMSE (area under RMSE vs hours) by region/model ===")
    print(summaries["integrated"].sort_values(["region","integrated_rmse"]).to_string(index=False))

if __name__ == "__main__":
    main()



"""
python region_rmse.py \
  --rolled_dir /path/to/rolled_nc_dir \
  --hres_path  /path/to/hres_or_era5.nc \
  --regions india mumbai delhi kolkata chennai \
  --out_dir ./plots/rmse_regions \
  --baseline HRES \
  --smooth_window 1 \
  --loglevel INFO
"""
