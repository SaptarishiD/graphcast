# <eval_acc.py>
import xarray as xr
import matplotlib.pyplot as plt


pred_ft = xr.open_dataset('/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/main_eval_7_days_fine_010824.nc')
pred_base = xr.open_dataset('/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/main_eval_7_days_base_010824.nc')
targs = xr.open_dataset('/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/main_eval_7_days_targets_010824.nc')
clim_ppt = xr.open_zarr('/Datastorage/saptarishi.dhanuka_asp25/era5_data/precip_climatology.zarr')


import xarray as xr
import numpy as np
import pandas as pd

def compute_precipitation_acc_debugged(
    predictions_finetuned: xr.Dataset,
    targets: xr.Dataset,
    climatology_precip: xr.DataArray
) -> xr.DataArray:
    """
    Computes a latitude-weighted Anomaly Correlation Coefficient (ACC) for precipitation.

    This function is debugged to correctly use the .dt accessor for pandas
    datetime properties. It handles timedelta conversion, aligns dataset
    structures, interpolates climatology, and applies latitude weighting.

    Args:
        predictions_finetuned: An xarray Dataset containing the predicted 'total_precipitation_6hr'.
        targets: An xarray Dataset containing the target 'total_precipitation_6hr'.
        climatology_precip: An xarray DataArray of precipitation climatology.

    Returns:
        An xarray DataArray containing the ACC value for each forecast time step.
    """
    # 1. Select the variable and align data structures
    pred_pr = predictions_finetuned['total_precipitation_6hr'].squeeze('batch')
    targ_pr = targets['total_precipitation_6hr'].squeeze('batch').transpose('time', 'lat', 'lon')

    # 2. Create absolute time coordinates
    start_time = pd.to_datetime('2024-08-01 00:00:00')
    # Convert the 'time' coordinate (timedelta) to a pandas Series for easy addition
    time_deltas = pred_pr['time'].to_pandas()
    valid_times = start_time + time_deltas

    # 3. Correctly derive dayofyear and hour using the .dt accessor
    # This is the fix for the AttributeError.
    dayofyear_vals = xr.DataArray(valid_times.dt.dayofyear, coords={'time': pred_pr.time})
    hour_vals = xr.DataArray(valid_times.dt.hour, coords={'time': pred_pr.time})

    # 4. Prepare climatology: rename and interpolate
    climatology_renamed = climatology_precip.rename({'latitude': 'lat', 'longitude': 'lon'})
    climatology_interp = climatology_renamed.sel(
        dayofyear=dayofyear_vals,
        hour=hour_vals
    ).interp_like(pred_pr)

    # 5. Compute anomalies
    pred_anomaly = pred_pr - climatology_interp
    targ_anomaly = targ_pr - climatology_interp

    # 6. Compute latitude weights
    weights = np.cos(np.deg2rad(pred_anomaly['lat']))
    weights.name = "weights"

    # 7. Calculate the weighted Anomaly Correlation Coefficient
    pred_anomaly_mean = pred_anomaly.weighted(weights).mean(dim=['lat', 'lon'])
    targ_anomaly_mean = targ_anomaly.weighted(weights).mean(dim=['lat', 'lon'])

    pred_anomaly_dev = pred_anomaly - pred_anomaly_mean
    targ_anomaly_dev = targ_anomaly - targ_anomaly_mean

    numerator = (weights * pred_anomaly_dev * targ_anomaly_dev).sum(dim=['lat', 'lon'])
    denominator_pred_sq = (weights * pred_anomaly_dev**2).sum(dim=['lat', 'lon'])
    denominator_targ_sq = (weights * targ_anomaly_dev**2).sum(dim=['lat', 'lon'])
    
    acc = numerator / np.sqrt(denominator_pred_sq * denominator_targ_sq)
    # acc.name = "anomaly_correlation_coefficient"

    return acc

acc_base = compute_precipitation_acc_debugged(pred_base.sel(lat=slice(6, 37), lon=slice(65, 95)), targs.sel(lat=slice(6, 37), lon=slice(65, 95)), clim_ppt.sel(latitude=slice(37, 6), longitude=slice(65, 95)))

acc = compute_precipitation_acc_debugged(pred_ft.sel(lat=slice(6, 37), lon=slice(65, 95)), targs.sel(lat=slice(6, 37), lon=slice(65, 95)), clim_ppt.sel(latitude=slice(37, 6), longitude=slice(65, 95)))





import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def plot_acc_india_precip_multi(acc_datasets, 
                                precip_var_candidates=("total_precipitation_6hr", "total_precipitation", "tp"),
                                title="India precip ACC (multiple inits/models)",
                                show_skill_lines=True):
    """
    Plot ACC vs lead time for multiple datasets on the same graph.

    Parameters
    ----------
    acc_datasets : list of (str, xr.Dataset)
        List of (label, ACC dataset) tuples.
    precip_var_candidates : tuple of str
        Candidate variable names for precip ACC.
    title : str
        Plot title.
    show_skill_lines : bool
        Whether to draw reference skill lines at ACC=0.6 and ACC=0.5.
    """

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for label, acc_ds in acc_datasets:
        # 1) pick precip var
        acc_var = None
        for cand in precip_var_candidates:
            if cand == 'total_precipitation_6hr':
                acc_var = cand
                break
        if acc_var is None:
            raise KeyError(f"Could not find a precip ACC variable in {list(acc_ds.data_vars)} for '{label}'")

        acc = acc_ds[acc_var]

        # drop/mean over unwanted dims
        for d in ("level", "batch"):
            if d in acc.dims:
                acc = acc.mean(d)

        # 2) lead axis in hours (time is timedelta64)
        lead_hours = (acc["time"].astype("timedelta64[h]").astype(int)).values
        lead_days = lead_hours / 24.0

        # 3) plot each dataset
        ax.plot(lead_hours, acc.values, marker="o", linewidth=1.5, label=label)

        # Quick stats
        below = np.where(np.asarray(acc.values) < 0.6)[0]
        if below.size:
            first_below_idx = below[0]
            print(f"[{label}] First time ACC < 0.6 at lead = {lead_hours[first_below_idx]} h "
                  f"(day {lead_days[first_below_idx]:.1f}).")
        else:
            print(f"[{label}] ACC stays ≥ 0.6 over the shown lead times.")

    # Add reference lines and day separators
    if show_skill_lines:
        ax.axhline(0.6, linestyle="--", linewidth=1.0, color="gray", label="ACC = 0.6")
        ax.axhline(0.5, linestyle=":", linewidth=1.0, color="gray", label="ACC = 0.5")

    max_lead_days = int(lead_days.max())
    for d in range(max_lead_days + 1):
        ax.axvline(d * 24, color="0.85", linewidth=0.8, zorder=0)

    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("ACC")
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"acc_india_precip_multi_{title.replace(' ', '_').lower()}.png", dpi=300)

plot_acc_india_precip_multi([
    ("Model Base", acc_base),
    ("Model Fine", acc)
], precip_var_candidates=("total_precipitation_6hr", "total_precipitation", "tp"),
title="India Precipitation ACC (Base vs Fine-tuned Model)", show_skill_lines=True)
# </eval_acc.py>