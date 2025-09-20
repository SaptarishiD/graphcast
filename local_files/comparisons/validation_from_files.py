#!/usr/bin/env python3
import os
import re
import glob
import math
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ----------------------- config (edit as needed) -----------------------
INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 38.0
INDIA_LON_MIN, INDIA_LON_MAX = 65.0, 95.0
PRECIP_VAR = "total_precipitation_6hr"   # if Dataset, we take this variable
# ----------------------------------------------------------------------

_DATE_RE = re.compile(r".*init[_ ](\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.nc$")
_TARGET_DATE_RE = re.compile(
    r"^target_init_6h_precip_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.nc$"
)

def _parse_init_date(path: str) -> Optional[pd.Timestamp]:
    m = _DATE_RE.match(os.path.basename(path))
    if not m:
        return None
    # Filename carries a space between date and time (e.g., "2014-08-23 00:00:00")
    return pd.to_datetime(m.group(1), utc=False)


def _parse_target_init_date(path_or_name: str) -> Optional[pd.Timestamp]:
    """Parse init timestamp from a target filename like:
       'target_init_6h_precip_2014-08-22 00:00:00.nc'."""
    name = os.path.basename(path_or_name)
    m = _TARGET_DATE_RE.match(name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), utc=False)


def _try_open_as_dataarray(path: str):
    """
    Try to open as DataArray; if that fails, open as Dataset.
    If Dataset: return Dataset.
    If DataArray: return DataArray.
    """
    try:
        da = xr.open_dataarray(path, decode_timedelta=True)
        return da
    except Exception:
        return xr.open_dataset(path, decode_timedelta=True)

def _extract_precip(obj, var_name=PRECIP_VAR) -> xr.DataArray:
    """
    Accepts either an xarray.Dataset or xarray.DataArray.
    - If Dataset: returns dataset[var_name] if present, otherwise:
        - if exactly one data_var, return it
        - else raise with a clear error
    - If DataArray: returns it verbatim
    """
    if isinstance(obj, xr.DataArray):
        return obj
    if isinstance(obj, xr.Dataset):
        if var_name in obj.data_vars:
            return obj[var_name]
        if len(obj.data_vars) == 1:
            only = list(obj.data_vars)[0]
            warnings.warn(f"'{var_name}' not found; using the only var present: '{only}'")
            return obj[only]
        raise ValueError(
            f"Dataset has multiple variables but '{var_name}' not found. "
            f"Available: {list(obj.data_vars)}"
        )
    raise TypeError(f"Unsupported xarray type: {type(obj)}")

def _lat_name(da: xr.DataArray) -> str:
    for k in ["lat", "latitude", "y"]:
        if k in da.dims or k in da.coords:
            return k
    raise KeyError("Latitude dim/coord not found (tried 'lat','latitude','y')")

def _lon_name(da: xr.DataArray) -> str:
    for k in ["lon", "longitude", "x"]:
        if k in da.dims or k in da.coords:
            return k
    raise KeyError("Longitude dim/coord not found (tried 'lon','longitude','x')")

def _select_bbox(da: xr.DataArray,
                 lat_range=(INDIA_LAT_MIN, INDIA_LAT_MAX),
                 lon_range=(INDIA_LON_MIN, INDIA_LON_MAX)) -> xr.DataArray:
    latn = _lat_name(da); lonn = _lon_name(da)
    lat_vals = da[latn].values
    lon_vals = da[lonn].values

    # Respect coordinate ordering for slicing
    lat_slice = slice(lat_range[0], lat_range[1]) if lat_vals[0] <= lat_vals[-1] else slice(lat_range[1], lat_range[0])
    lon_slice = slice(lon_range[0], lon_range[1]) if lon_vals[0] <= lon_vals[-1] else slice(lon_range[1], lon_range[0])

    return da.sel({latn: lat_slice, lonn: lon_slice})

def _ensure_time(da: xr.DataArray) -> xr.DataArray:
    """
    Try to guarantee a 'time' coordinate for alignment.
    Strategies:
      1) If 'time' already present -> keep.
      2) If 'time' and 'step' present -> compute time = time + step (broadcast).
      3) Else, if 'step' present in both pred & target later, we align on 'step' instead.
    """
    if "time" in da.coords:
        return da

    has_time = "time" in da.coords or "time" in da.dims
    has_step = "step" in da.coords or "step" in da.dims

    if has_time and has_step:
        # Broadcast: many forecast files use time (init) and step (lead)
        time_coord = da["time"]
        step_coord = da["step"]
        # xr automatically broadcasts when adding
        vt = time_coord + step_coord
        da = da.assign_coords(time=vt)
        return da

    # Fall back; caller may align on 'step' later
    return da

def _align_pred_target(pred: xr.DataArray, tgt: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, str]:
    """
    Align along 'time' if possible, otherwise along 'step'.
    Returns (pred_aligned, targ_aligned, align_key) where align_key in {'time','step'}.
    """
    p = _ensure_time(pred)
    t = _ensure_time(tgt)
    # print(p.coords)
    # print('----')
    # print(p.dims)
    # print('----')
    # print(t.dims)
    # print('----')
    # print(t.coords)
    # print('----')

    if "time" in p.coords and "time" in t.coords:
        common = np.intersect1d(p["time"].values,
                                t["time"].values)
        if len(common) == 0:
            raise ValueError("No overlapping time between prediction and target.")
        p = p.sel(time=common)
        t = t.sel(time=common)
        return p, t, "time"

    # else try step
    if "step" in p.dims or "step" in p.coords:
        if "step" in t.dims or "step" in t.coords:
            common = np.intersect1d(p["step"].values, t["step"].values)
            if len(common) == 0:
                raise ValueError("No overlapping step between prediction and target.")
            p = p.sel(step=common)
            t = t.sel(step=common)
            return p, t, "step"

    raise ValueError("Could not align pred & target: need shared 'time' or 'step'.")

def _spatial_rmse(pred: xr.DataArray, tgt: xr.DataArray) -> xr.DataArray:
    """
    sqrt(mean((pred - tgt)^2)) over spatial dims only.
    Keeps any remaining dims (e.g., 'time' or 'step').
    """
    latn = _lat_name(pred); lonn = _lon_name(pred)
    diff2 = (pred - tgt) ** 2
    return np.sqrt(diff2.mean(dim=[latn, lonn], skipna=True))

def _discover_files(data_dir: str) -> Dict[str, Dict[pd.Timestamp, str]]:
    """
    Returns a dict:
      {'target': {init_time: path, ...},
       'base':   {init_time: path, ...},
       'fine':   {init_time: path, ...}}
    """
    out = {"target": {}, "base": {}, "fine": {}}

    patterns = {
        "target": os.path.join(data_dir, "target_init_6h_precip_*.nc"),
        "base":   os.path.join(data_dir, "base_init_*.nc"),
        "fine":   os.path.join(data_dir, "fine_val_init_*.nc"),
    }

    for key, pat in patterns.items():
        for p in glob.glob(pat):
            if key == "target":
                ts = _parse_target_init_date(p)
                # print(ts)
            else:
                ts = _parse_init_date(p)  # handles base/fine: *_init_YYYY-mm-dd HH:MM:SS.nc
            if ts is not None:
                out[key][ts] = p

    return out


def _filter_by_date(d: Dict[pd.Timestamp, str],
                    start: Optional[pd.Timestamp],
                    end: Optional[pd.Timestamp]) -> Dict[pd.Timestamp, str]:
    if start is not None:
        d = {k: v for k, v in d.items() if k >= start}
    if end is not None:
        d = {k: v for k, v in d.items() if k <= end}
    return dict(sorted(d.items()))



def _lead_hours_index(align_key, rmse_da, init_time):
    """
    Return lead hours as ints. Works whether the coordinate is valid_time (datetime64)
    or step (timedelta64 / raw ns ints).
    """
    # Pick the axis to align on
    if align_key in rmse_da.coords:
        v = rmse_da[align_key].values
    elif "valid_time" in rmse_da.coords:
        v = rmse_da["valid_time"].values
    elif "step" in rmse_da.coords:
        v = rmse_da["step"].values
    else:
        raise KeyError(
            f"No alignable coord found: tried '{align_key}', 'valid_time', 'step'"
        )

    v = np.asarray(v)

    # Normalize init_time to numpy datetime64[ns]
    if isinstance(init_time, np.datetime64):
        init64 = init_time.astype("datetime64[ns]")
    elif isinstance(init_time, pd.Timestamp):
        init64 = np.datetime64(init_time.to_datetime64())
    else:
        # handles str, python datetime, etc.
        init64 = np.datetime64(pd.to_datetime(init_time), "ns")

    if np.issubdtype(v.dtype, np.datetime64):
        # v are absolute valid times
        lead_h = (v.astype("datetime64[ns])") - init64).astype("timedelta64[h]").astype(int)
    else:
        # v are durations (timedelta) or raw ns ints
        if not np.issubdtype(v.dtype, np.timedelta64):
            v = v.astype("timedelta64[ns]")
        lead_h = v.astype("timedelta64[h]").astype(int)

    return lead_h.astype(float)

def validate_from_files(
    data_dir: str,
    outdir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    models=("base", "fine"),
    var_name: str = PRECIP_VAR,
):
    """
    Scan data_dir for target/base/fine files, filter by init-date range,
    compute per-lead RMSE over India, and write CSVs + plots.

    Outputs:
      - {outdir}/rmse_base_by_init_and_step.csv
      - {outdir}/rmse_fine_by_init_and_step.csv
      - {outdir}/rmse_model_summary.csv
      - {outdir}/plots/rmse_{YYYY-MM-DD_HHMMSS}.png (per init, Base vs Fine)
      - {outdir}/plots/rmse_average_across_inits.png
    """
    os.makedirs(outdir, exist_ok=True)
    plotdir = os.path.join(outdir, "plots")
    os.makedirs(plotdir, exist_ok=True)

    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None

    files = _discover_files(data_dir)
    targets = _filter_by_date(files["target"], start_ts, end_ts)
    # print(f"Targets: {files}")

    # Only evaluate init dates where the target exists and model file exists
    per_model_records: Dict[str, List[Dict]] = {m: [] for m in models}

    for init_time, targ_path in targets.items():
        # print(init_time)
        # For each chosen model, check if we have a matching init
        for model in models:
            model_path = files[model].get(init_time)
            if model_path is None:
                # No matching forecast for this init — skip quietly
                continue

            # Load arrays (DataArray or Dataset); extract precip var if needed
            targ_raw = _try_open_as_dataarray(targ_path)
            pred_raw = _try_open_as_dataarray(model_path)
            target_da = _extract_precip(targ_raw, var_name=var_name)
            pred_da = _extract_precip(pred_raw, var_name=var_name)

            # Spatial subset over India
            target_da = _select_bbox(target_da)
            pred_da = _select_bbox(pred_da)

            # Align along time (preferred) or step
            pred_al, targ_al, align_key = _align_pred_target(pred_da, target_da)

            # Compute spatial RMSE (keeps time/step dimension)
            rmse_da = _spatial_rmse(pred_al, targ_al)

            # Turn into rows
            lead_hours = _lead_hours_index(align_key, rmse_da, init_time)
            rmse_vals = rmse_da.values.reshape(-1)

            # Store records
            for lh, rv in zip(lead_hours, rmse_vals):
                per_model_records[model].append(
                    {
                        "init_time": init_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lead_hours": float(lh),
                        "rmse": float(rv),
                        "align_key": align_key,
                    }
                )

            # Per-init plot (Base vs Fine): we accumulate both models first,
            # so we only draw after loop if both are available. Here we’ll collect into a dict.
        # end for model
    # end for init

    # Write per-model CSVs and make plots
    summaries = []
    per_init_series: Dict[pd.Timestamp, Dict[str, pd.DataFrame]] = {}

    for model in models:
        rows = per_model_records[model]
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(["init_time", "lead_hours"])
        csv_path = os.path.join(outdir, f"rmse_{model}_by_init_and_step.csv")
        df.to_csv(csv_path, index=False)

        # Track average across inits (by lead_hours)
        avg = df.groupby("lead_hours", as_index=False)["rmse"].mean()
        avg["model"] = model
        summaries.append(avg)

        # Stash per-init data for later “Base vs Fine” plots
        for init_str, g in df.groupby("init_time"):
            init_ts = pd.to_datetime(init_str)
            per_init_series.setdefault(init_ts, {})[model] = g[["lead_hours", "rmse"]].reset_index(drop=True)

    # Summary CSV
    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        summary_df.to_csv(os.path.join(outdir, "rmse_model_summary.csv"), index=False)

    # --- Plot: average across inits ---
    if summaries:
        plt.figure(figsize=(8, 5), dpi=150)
        for model in models:
            sub = summary_df[summary_df["model"] == model]
            if not sub.empty:
                plt.plot(sub["lead_hours"], sub["rmse"], marker="o", label=model.upper())
        plt.xlabel("Lead (hours)")
        plt.ylabel("RMSE over India (mm/6h)")
        plt.title("Precip RMSE vs Lead — average across inits")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plotdir, "rmse_average_across_inits.png"))
        plt.close()

    # --- Plots: per-init (Base vs Fine) ---
    # Only draw when we have at least one model for that init; if both, plot both.
    from tqdm import tqdm
    for init_ts, model_map in tqdm(sorted(per_init_series.items())):
        plt.figure(figsize=(8, 5), dpi=150)
        plotted = False
        for model in models:
            if model in model_map:
                dfm = model_map[model].sort_values("lead_hours")
                plt.plot(dfm["lead_hours"], dfm["rmse"], marker="o", label=model.upper())
                plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Lead (hours)")
        plt.ylabel("RMSE over India (mm/6h)")
        plt.title(f"Precip RMSE vs Lead — {init_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fname = f"rmse_{init_ts.strftime('%Y-%m-%d_%H%M%S')}.png"
        plt.savefig(os.path.join(plotdir, fname))
        plt.close()

    print(f"Done. Outputs in: {outdir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validate base/fine forecasts against target files.")
    ap.add_argument("--data_dir", required=True, help="Directory with target/base/fine files")
    ap.add_argument("--outdir", required=True, help="Where to write CSVs and plots")
    ap.add_argument("--start_date", default=None, help="e.g. '2014-08-01 00:00:00'")
    ap.add_argument("--end_date",   default=None, help="e.g. '2015-09-30 00:00:00'")
    ap.add_argument("--models", default="base,fine", help="Comma list among {base,fine}")
    ap.add_argument("--var_name", default=PRECIP_VAR, help="Variable to extract when files are Datasets")
    args = ap.parse_args()

    models = tuple([m.strip() for m in args.models.split(",") if m.strip()])
    validate_from_files(
        data_dir=args.data_dir,
        outdir=args.outdir,
        start_date=args.start_date,
        end_date=args.end_date,
        models=models,
        var_name=args.var_name,
    )
