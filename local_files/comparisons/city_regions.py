from tqdm.auto import tqdm
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob
import re
from typing import Dict, Tuple, List, Optional

# ----------------------------
# Helpers
# ----------------------------

def _extract_init_from_stem(stem: str) -> str:
    """
    Extract init datetime string from a filename stem.
    Expected forms include:
      * "*init_YYYY-MM-DD HH:MM:SS"
      * "*init_YYYY-MM-DD_HH:MM:SS"
      * "target_init_..." / "fine_init_..." / "base_init_..." etc.
    Returns the canonical "YYYY-MM-DD HH:MM:SS" string.
    """
    # 1) Prefer explicit "...init_<datetime>" pattern
    m = re.search(r'init_(\d{4}-\d{2}-\d{2}[ _]\d{2}:\d{2}:\d{2})', stem)
    if m:
        dt = m.group(1).replace('_', ' ')
        return dt

    # 2) Fallback: any datetime in the stem
    m2 = re.search(r'(\d{4}-\d{2}-\d{2}[ _]\d{2}:\d{2}:\d{2})', stem)
    if m2:
        return m2.group(1).replace('_', ' ')

    raise ValueError(f"Could not extract init datetime from filename stem: {stem}")


def _lat_slice(da: xr.DataArray, lat_min: float, lat_max: float):
    """Return a slice object that respects ascending or descending latitude coordinates."""
    lat_vals = da["lat"].values
    if lat_vals[0] <= lat_vals[-1]:
        return slice(lat_min, lat_max)
    else:
        return slice(lat_max, lat_min)


def _lon_slice(da: xr.DataArray, lon_min: float, lon_max: float):
    """Return a slice object for longitudes (assumes ascending order)."""
    return slice(lon_min, lon_max)


def _select_region(da: xr.DataArray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.DataArray:
    """
    Select a lat/lon bbox from a DataArray with dims (..., lat, lon).
    Handles 0/360 wrap for longitudes if lon_min > lon_max (e.g., 350..360 U 0..20).
    """
    # Latitude slice respecting order
    lat_sel = _lat_slice(da, lat_min, lat_max)

    # Longitude(s)
    if lon_min <= lon_max:
        return da.sel(lat=lat_sel, lon=_lon_slice(da, lon_min, lon_max))
    else:
        # wrap: [lon_min..360] U [0..lon_max]
        part1 = da.sel(lat=lat_sel, lon=_lon_slice(da, lon_min, 360))
        part2 = da.sel(lat=lat_sel, lon=_lon_slice(da, 0, lon_max))
        return xr.concat([part1, part2], dim="lon")


def _ensure_dims_and_batch(forecast_da: xr.DataArray, target_da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Ensure both arrays have dims (time, batch, lat, lon). If 'batch' is missing, add it.
    Also transpose target to match forecast order when needed.
    """
    # Ensure 'batch' exists
    if "batch" not in forecast_da.dims:
        forecast_da = forecast_da.expand_dims({"batch": [0]}, axis=1)
    if "batch" not in target_da.dims:
        target_da = target_da.expand_dims({"batch": [0]}, axis=1)

    # Ensure consistent dim order
    desired = ("time", "batch", "lat", "lon")
    if forecast_da.dims != desired:
        forecast_da = forecast_da.transpose(*[d for d in desired if d in forecast_da.dims])
    if target_da.dims != desired:
        target_da = target_da.transpose(*[d for d in desired if d in target_da.dims])

    return forecast_da, target_da


def _lead_hours_from_time_coord(time_values) -> np.ndarray:
    """Convert a sequence of time-like deltas to hours as float."""
    return np.array([pd.Timedelta(t).total_seconds() / 3600.0 for t in time_values], dtype=float)

# ----------------------------
# Core computation
# ----------------------------

def compute_regional_rmse(
    forecast_file: str,
    target_file: str,
    regions_dict: Dict[str, Tuple[float, float, float, float]],
    model_name: str = "model",
    precip_var: str = "total_precipitation_6hr",
) -> pd.DataFrame:
    """
    Compute RMSE for precipitation over specified regions for all lead times.

    Returns a DataFrame with columns: init_date, lead_time, model, region, rmse
    """
    init_date_str = _extract_init_from_stem(Path(forecast_file).stem)
    init_date = pd.to_datetime(init_date_str)

    # Open lazily; do computation via xarray to support Dask-backed files too.
    with xr.open_dataset(forecast_file, decode_timedelta=True) as fds, xr.open_dataset(target_file, decode_timedelta=True) as tds:
        if precip_var not in fds or precip_var not in tds:
            raise KeyError(f"'{precip_var}' not found in one of the datasets")

        forecast_da = fds[precip_var]
        target_da = tds[precip_var]

        # Ensure dims & add missing 'batch'
        forecast_da, target_da = _ensure_dims_and_batch(forecast_da, target_da)

        # Vectorized time axis handling
        lead_hours = _lead_hours_from_time_coord(forecast_da["time"].values)

        rows = []
        # Compute region-wise RMSE across spatial dims for every lead time (vectorized)
        for region_name, (lat_min, lat_max, lon_min, lon_max) in regions_dict.items():
            f_reg = _select_region(forecast_da, lat_min, lat_max, lon_min, lon_max).isel(batch=0)
            t_reg = _select_region(target_da,   lat_min, lat_max, lon_min, lon_max).isel(batch=0)

            # Align just in case coords differ slightly
            f_reg, t_reg = xr.align(f_reg, t_reg, join="inner")

            # mean over lat/lon -> one value per time
            # NOTE: keep computations in xarray; compute at the end
            rmse_t = ((f_reg - t_reg) ** 2).mean(dim=("lat", "lon")) ** 0.5
            rmse_vals = rmse_t.compute().values  # shape: (time,)

            # Assemble rows for all lead times
            for lh, rv in zip(lead_hours, rmse_vals):
                rows.append(
                    {
                        "init_date": init_date,
                        "lead_time": float(lh),
                        "model": model_name,
                        "region": region_name,
                        "rmse": float(rv),
                    }
                )

    return pd.DataFrame(rows)


def process_all_forecasts_for_model(
    forecast_dir: str,
    target_dir: str,
    regions_dict: Dict[str, Tuple[float, float, float, float]],
    model_name: str,
    forecast_pattern: str = "*init_*.nc",
    target_prefix: str = "target_init_",
    precip_var: str = "total_precipitation_6hr",
    restrict_init_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process all forecast files for a single model and compute regional RMSE.

    restrict_init_dates (optional): list of init datetime strings in the form
      "YYYY-MM-DD HH:MM:SS". If provided, only those inits are processed.
    """
    forecast_files = sorted(glob.glob(str(Path(forecast_dir) / forecast_pattern)))
    if not forecast_files:
        raise ValueError(f"No forecast files found in {forecast_dir} with pattern {forecast_pattern}")

    all_results = []

    for fpath in forecast_files:
        stem = Path(fpath).stem
        try:
            init_dt = _extract_init_from_stem(stem)
        except ValueError:
            print(f"[{model_name}] Warning: Could not parse init from '{stem}', skipping.")
            continue

        if restrict_init_dates is not None and init_dt not in restrict_init_dates:
            continue

        target_file = Path(target_dir) / f"{target_prefix}{init_dt}.nc"
        if not target_file.exists():
            print(f"[{model_name}] Warning: Missing target for init '{init_dt}': {target_file.name}, skipping.")
            continue

        print(f"[{model_name}] Processing init: {init_dt}")
        try:
            df = compute_regional_rmse(
                forecast_file=fpath,
                target_file=str(target_file),
                regions_dict=regions_dict,
                model_name=model_name,
                precip_var=precip_var,
            )
            all_results.append(df)
        except Exception as e:
            print(f"[{model_name}] Error at init {init_dt}: {e}")
            continue

    if not all_results:
        return pd.DataFrame(columns=["init_date", "lead_time", "model", "region", "rmse"])

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.sort_values(["init_date", "lead_time", "region"], inplace=True)
    return final_df


def process_multiple_models(
    models: List[Dict[str, str]],
    target_dir: str,
    regions_dict: Dict[str, Tuple[float, float, float, float]],
    output_combined_file: str,
    per_model_outdir: Optional[str] = None,
    target_prefix: str = "target_init_",
    precip_var: str = "total_precipitation_6hr",
    restrict_init_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process forecasts for many models and save results.

    models: list of dicts, each with:
        {
          "name": "base_model",
          "forecast_dir": "/path/to/forecasts",
          "forecast_pattern": "base_init_2014*.nc"  # optional; defaults to "*init_*.nc"
        }

    Saves:
      - Per-model CSVs if per_model_outdir is provided
      - Combined CSV at output_combined_file

    Returns combined DataFrame.
    """
    per_model_dfs = []
    per_model_outdir = Path(per_model_outdir) if per_model_outdir else None
    if per_model_outdir:
        per_model_outdir.mkdir(parents=True, exist_ok=True)

    for m in tqdm(models, desc="Models"):
        name = m["name"]
        fdir = m["forecast_dir"]
        fpat = m.get("forecast_pattern", "*init_*.nc")
        print(f"\n=== Model: {name} ===")
        df = process_all_forecasts_for_model(
            forecast_dir=fdir,
            target_dir=target_dir,
            regions_dict=regions_dict,
            model_name=name,
            forecast_pattern=fpat,
            target_prefix=target_prefix,
            precip_var=precip_var,
            restrict_init_dates=restrict_init_dates,
        )
        if not df.empty and per_model_outdir:
            out_path = per_model_outdir / f"rmse_regions_{name}.csv"
            df.to_csv(out_path, index=False)
            print(f"[{name}] Saved per-model results -> {out_path}")
        per_model_dfs.append(df)

    if per_model_dfs:
        combined = pd.concat(per_model_dfs, ignore_index=True)
        combined.sort_values(["model", "init_date", "lead_time", "region"], inplace=True)
        Path(output_combined_file).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_combined_file, index=False)
        print(f"\nSaved combined results -> {output_combined_file}")
        return combined

    print("\nNo results to save for any model.")
    return pd.DataFrame(columns=["init_date", "lead_time", "model", "region", "rmse"])


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Define regions: (lat_min, lat_max, lon_min, lon_max)
    regions = {
        "North_America": (20, 70, 230, 300),   # 0-360 longitudes
        "Europe":        (35, 70, 350, 40),    # wraps 350..360 U 0..40
        "East_Asia":     (20, 50, 100, 145),
        "South_America": (-55, 15, 280, 330),
        "Africa":        (-35, 35, 340, 55),   # wraps
        "Australia":     (-45, -10, 110, 155),
        "Arctic":        (70, 90, 0, 360),     # all longitudes
        "Tropics":       (-20, 20, 0, 360),

        # Major Indian cities (rounded to nearest integer degree ranges)
        "mumbai":        (18, 20, 72, 74),
        "delhi":         (28, 29, 76, 78),
        "kolkata":       (22, 23, 88, 89),
        "chennai":       (12, 13, 80, 81),
        "bengaluru":     (12, 13, 77, 78),
        "hyderabad":     (17, 18, 78, 79),
        "ahmedabad":     (23, 24, 72, 73),
        "pune":          (18, 19, 73, 74),
        "jaipur":        (26, 27, 75, 76),
        "lucknow":       (26, 27, 80, 81),
        "bhopal":        (23, 24, 77, 78),
        "guwahati":      (26, 27, 91, 92),
        "srinagar":      (34, 35, 74, 75),
        "thiruvananthapuram": (8, 9, 76, 77),
    }

    # List as many models as you want here
    models_to_eval = [
        {
            "name": "base_model",
            "forecast_dir": "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/",
            "forecast_pattern": "base_init_2014*.nc",
        },
        {
            "name": "finetuned_model",
            "forecast_dir": "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/",
            "forecast_pattern": "fine_init_2014*.nc",
        },
        # Add more models as needed...
    ]

    # If desired, restrict to explicit init dates (strings exactly like filenames):
    # e.g., ["2014-09-14 00:00:00", "2014-09-16 00:00:00"]
    restrict_inits = None

    combined_df = process_multiple_models(
        models=models_to_eval,
        target_dir="/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds",
        regions_dict=regions,
        output_combined_file="rmse_regions_ALL_MODELS_cities.csv",
        per_model_outdir="per_model_results",
        target_prefix="target_init_",
        precip_var="total_precipitation_6hr",
        restrict_init_dates=restrict_inits,
    )

    if not combined_df.empty:
        print("\nSummary by model & region (mean RMSE):")
        print(combined_df.groupby(["model", "region"])["rmse"].mean().sort_values())

        print("\nSummary by model & lead_time (mean RMSE):")
        print(combined_df.groupby(["model", "lead_time"])["rmse"].mean().sort_values())

        print("\nSample rows:")
        print(combined_df.head(10))
    
    print(f"Saved combined results to 'rmse_regions_ALL_MODELS_cities.csv'")
