# plot_with_imerg_targets.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import xarray as xr
from scipy import stats
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

def _maybe_fix_lon(ds, lon_min=65.0, lon_max=95.0):
    """If dataset lon is 0..360 but bbox is in -180..180 (or vice versa), fix coords."""
    lons = ds['lon'].values
    if lons.min() >= 0 and lon_min < 0:
        # convert ds lon to -180..180
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    if lons.max() > 180 and lon_max <= 180:
        # ds is 0..360 but bbox is given as 65..95 -> OK, nothing to do
        pass
    return ds

def load_target_precip_from_accumulated(accumulated_nc_path,
                                       init_dates,
                                       varname='precipitation',
                                       india_lat_min=6.0, india_lat_max=38.0,
                                       india_lon_min=65.0, india_lon_max=95.0,
                                       start_month=8,
                                       lead_hours_per_step=6,
                                       n_steps=28,
                                       time_tolerance=pd.Timedelta('6h'),
                                       verbose=True):
    """
    Robust extractor: for each init_date (string or Timestamp) returns an array
    of length n_steps with summed precipitation over the India bbox for
    times init + 6h, init + 12h, ..., init + n_steps*6h.

    Only init dates with month >= start_month are processed.
    """
    ds = xr.open_dataset(accumulated_nc_path)
    if varname not in ds:
        raise KeyError(f"variable '{varname}' not found in {accumulated_nc_path}. Found: {list(ds.data_vars)}")

    # fix lon convention if needed
    ds = _maybe_fix_lon(ds, lon_min=india_lon_min, lon_max=india_lon_max)

    # Subset once to India bbox (this yields a smaller dataset in memory)
    try:
        ds_india = ds[[varname]].sel(lat=slice(india_lat_min, india_lat_max),
                                      lon=slice(india_lon_min, india_lon_max))
    except Exception as e:
        # If slice ordering is reversed, try reversing
        if verbose:
            print("Warning: spatial slice raised:", e, "trying reversed lat/lon order.")
        ds_india = ds[[varname]].sel(lat=slice(india_lat_max, india_lat_min),
                                      lon=slice(india_lon_max, india_lon_min))

    times = pd.to_datetime(ds_india['time'].values)
    if verbose:
        print(f"Accumulated dataset time range: {times.min()} to {times.max()}  (n={len(times)})")
        print(f"Processing {len(init_dates)} unique init dates (will filter month>={start_month})")

    offsets = [pd.Timedelta(hours=lead_hours_per_step * (i + 1)) for i in range(n_steps)]
    target_data = {}
    processed = 0
    total_valid_counts = 0

    for init in init_dates:
        try:
            init_ts = pd.to_datetime(init)
        except Exception:
            if verbose:
                print(f"Skipping unparsable init_date: {init}")
            continue

        if init_ts.month < start_month:
            if verbose:
                print(f"Skipping init {init_ts} because month {init_ts.month} < {start_month}")
            continue

        processed += 1
        vals = []
        valid_count = 0
        for off in offsets:
            desired_time = init_ts + off
            # try exact match
            sel = None
            try:
                # if the exact numpy datetime64 exists, .sel will find it; else will raise KeyError
                sel = ds_india[varname].sel(time=np.datetime64(desired_time))
                used_time = np.datetime64(desired_time)
                used_method = 'exact'
            except (KeyError, IndexError, ValueError):
                # fallback to nearest with tolerance
                try:
                    sel = ds_india[varname].sel(time=desired_time, method='nearest', tolerance=time_tolerance)
                    # find which time was selected
                    nearest_time = pd.to_datetime(sel['time'].values)
                    used_time = nearest_time
                    used_method = 'nearest'
                except Exception:
                    sel = None
                    used_time = None
                    used_method = 'none'

            if sel is None:
                if verbose:
                    print(f"Init {init_ts} + {off} -> NO match within {time_tolerance}")
                vals.append(np.nan)
                continue

            # reduce to scalar spatial sum
            try:
                ssum = sel.sum(dim=['lat', 'lon']).values
            except Exception:
                try:
                    ssum = sel.squeeze().sum(dim=['lat', 'lon']).values
                except Exception as e:
                    if verbose:
                        print(f"Summation error for {init_ts}+{off}: {e}")
                    ssum = np.nan

            # Convert to scalar float
            ssum_scalar = np.nan
            if isinstance(ssum, np.ndarray):
                if ssum.size == 1:
                    ssum_scalar = float(ssum.flatten()[0])
                else:
                    ssum_scalar = float(np.nanmean(ssum))
            elif np.isscalar(ssum):
                ssum_scalar = float(ssum)
            else:
                ssum_scalar = np.nan

            if np.isfinite(ssum_scalar):
                valid_count += 1
            vals.append(ssum_scalar)

            if verbose:
                print(f"Init {init_ts} +{int(off.total_seconds()/3600)}h -> used_time={used_time} method={used_method} value={ssum_scalar}")

        target_data[str(init_ts)] = np.array(vals, dtype=float)
        total_valid_counts += valid_count
        if verbose:
            print(f"Init {init_ts}: valid steps {valid_count}/{len(offsets)}")

    ds.close()
    if verbose:
        print(f"Processed {processed} init dates. Total valid (non-NaN) step counts across all inits: {total_valid_counts}")
    return target_data


def create_target_dataframe(target_data):
    """Same as before but with a defensive check and explicit dtype handling."""
    rows = []
    for init_date, precip_values in target_data.items():
        if precip_values is None or len(precip_values) == 0:
            continue
        for i, precip_val in enumerate(precip_values):
            horizon_hours = (i + 1) * 6
            horizon_ns = int(horizon_hours * 3600 * 1e9)
            rows.append({
                'init_date': init_date,
                'forecast_horizon': f'{horizon_ns} nanoseconds',
                'model': 'Ground_Truth',
                'region': 'India',
                'precipitation': (np.nan if np.isnan(precip_val) else float(precip_val))
            })
    result = pd.DataFrame(rows)
    result.to_csv('test_target.csv')
    return pd.DataFrame(rows)


# ---------------------- The plotting function (modified) -----------------------
def plot_forecast_with_ground_truth(csv_path: str,
                                    accumulated_nc_path: str,
                                    output_path: str,
                                    k_factor: float = 2.0,
                                    baseline_model: str = "Graphcast_Base",
                                    models_to_plot: list = None,
                                    model_rename: dict = None,
                                    use_saved_data: bool = True):
    # 1. Load forecast CSV
    print(f"Loading forecast CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # figure out horizon -> lead_time_hours
    def parse_horizon_to_hours(horizon_str):
        try:
            return pd.to_timedelta(horizon_str).total_seconds() / 3600
        except Exception:
            return np.nan

    if 'forecast_horizon_times' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_times'].apply(parse_horizon_to_hours)
    elif 'forecast_horizon_days' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_days'] * 24
    elif 'forecast_horizon' in df.columns:
        # your format: '123456789 nanoseconds'
        df['lead_time_hours'] = pd.to_timedelta(df['forecast_horizon']).dt.total_seconds() / 3600
    elif 'forecast_horizon_hours' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_hours']
    else:
        raise RuntimeError("Could not find forecast horizon column in CSV.")

    df.dropna(subset=['lead_time_hours'], inplace=True)
    if 'region' not in df.columns:
        df['region'] = 'India'
    has_region = 'region' in df.columns

    # Filter to only include models in models_to_plot
    if models_to_plot:
        df = df[df['model'].isin(models_to_plot)]
        print(f"Filtered to models: {models_to_plot}")

    # 2. Load ground truth from the accumulated IMERG dataset, but only for init dates month >= 8
    unique_init_dates = df['init_date'].unique()
    print(f"Loading ground truth from {accumulated_nc_path} for {len(unique_init_dates)} unique inits (August onward only)...")

    if not os.path.exists('test_target.csv'):
        target_data = load_target_precip_from_accumulated(
            accumulated_nc_path,
            unique_init_dates,
            varname='precipitation',
            india_lat_min=6.0, india_lat_max=38.0,
            india_lon_min=65.0, india_lon_max=95.0,
            start_month=8,
            lead_hours_per_step=6,
            n_steps=28,
            time_tolerance=pd.Timedelta('3h')
        )
    else:
        target_data = None
        target_df = pd.read_csv('test_target.csv')
        target_df['lead_time_hours'] = pd.to_timedelta(target_df['forecast_horizon']).dt.total_seconds() / 3600
        print(f"Loaded ground truth for initializations (after month>=8 filter).")

    if target_data:
        if not os.path.exists('test_target.csv'):
            target_df = create_target_dataframe(target_data)
        else:
            target_df = pd.read_csv('test_target.csv')
        target_df['lead_time_hours'] = pd.to_timedelta(target_df['forecast_horizon']).dt.total_seconds() / 3600
        print(f"Loaded ground truth for {len(target_data)} initializations (after month>=8 filter).")
    else:
        print("Warning: No target ground truth extracted from accumulated dataset.")
        target_df = pd.DataFrame()

    # 3. summary stats for forecast models (uses mse column)
    group_cols = ['model'] + (['region'] if has_region else []) + ['lead_time_hours']
    summary_df = (
        df.groupby(group_cols)['mse']
          .agg(['mean', 'std', 'count'])
          .reset_index()
          .rename(columns={'mean': 'mse_mean', 'std': 'mse_std', 'count': 'n_samples'})
    )
    summary_df['mse_std'].fillna(0, inplace=True)

    # confidence intervals
    z_score = 1.96
    summary_df['ci_half_width_standard'] = z_score * (summary_df['mse_std'] / np.sqrt(summary_df['n_samples']))
    summary_df['ci_half_width_corrected'] = k_factor * summary_df['ci_half_width_standard']

    # 4. Ground truth - calculate total precipitation at each lead time (no mean/std)
    if not target_df.empty:
        # Sum precipitation across all init dates for each lead time
        precip_totals = target_df.groupby('lead_time_hours')['precipitation'].sum().reset_index()
        precip_totals['model'] = 'Ground_Truth'
        precip_totals['region'] = 'India'
        # Save precipitation totals for future use
        precip_totals.to_csv('precipitation_totals.csv', index=False)
        print("Saved precipitation totals to precipitation_totals.csv")
    else:
        precip_totals = pd.DataFrame()

    # Save summary statistics for future use
    summary_df.to_csv('forecast_summary_stats.csv', index=False)
    print("Saved forecast summary statistics to forecast_summary_stats.csv")

    # 5. Plotting - modified to show MSE and precipitation on same plot with dual y-axis
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax2 = plt.subplots(1, 1, figsize=(16, 8))
    
    # Create second y-axis for precipitation
    ax1 = ax2.twinx()

    manual_colors = {
        "Graphcast_Base_2014-08-01_2014-09-30": "tab:blue", 
        "Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz": "tab:orange", 
        "Graphcast_Finetuned2 (India)": "tab:green",
        "Base": "tab:blue",
        "Finetuned1": "tab:orange"
    }

    group_cols_for_plot = ['model', 'region'] if has_region else ['model']
    
    # Plot MSE on main axis (ax2 - left y-axis)
    for group_keys, group_data in summary_df.groupby(group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]
        display_name = model_rename.get(model_name, model_name) if model_rename else model_name
        label = display_name if region is None else f"{display_name} ({region})"
        
        if model_name == 'HRES':
            continue
            
        color = manual_colors.get(label, manual_colors.get(display_name, "black"))
        group_data = group_data.sort_values('lead_time_hours')
        
        ax2.plot(group_data['lead_time_hours'], group_data['mse_mean'], 
                marker='o', linestyle='-', color=color, label=label, linewidth=2)
        ax2.fill_between(group_data['lead_time_hours'],
                        group_data['mse_mean'] - group_data['ci_half_width_corrected'],
                        group_data['mse_mean'] + group_data['ci_half_width_corrected'],
                        color=color, alpha=0.1)

    # Plot total precipitation on second axis (ax1 - right y-axis)
    if not precip_totals.empty:
        ax1.plot(precip_totals['lead_time_hours'], precip_totals['precipitation'], 
                linestyle='-', color='red', label='Ground Truth Total Precipitation', 
                linewidth=3, alpha=0.4)

    # Configure axes
    ax2.set_title('Forecast Skill (MSE) and Ground Truth Precipitation vs. Lead Time', fontsize=16, pad=20)
    ax2.set_ylabel('Average MSE (Lower is better)', fontsize=12, color='black')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax1.set_ylabel('Total Precipitation', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    max_hours = summary_df['lead_time_hours'].max() if not summary_df.empty else 168
    ax2.set_xlabel('Lead Time (hours)', fontsize=12)
    ax2.set_xticks(np.arange(0, max_hours + 1, 24))

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.show()


# ---------------------- Main CLI -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Forecast CSV with init_date and mse columns")
    parser.add_argument("--accumulated_nc", type=str, required=True, help="Path to imerg_2014_6h_accumulated.nc")
    parser.add_argument("--output_path", type=str, default="./plots/forecast_with_truth.png")
    parser.add_argument("--k_factor", type=float, default=2.0)
    parser.add_argument("--baseline_model", type=str, default="Graphcast_Base")
    parser.add_argument("--models_to_plot", type=str, default=None)
    parser.add_argument("--model_rename", type=str, default=None)
    parser.add_argument("--use_saved_data", action="store_true", default=True, help="Use saved CSV files if available")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation even if saved files exist")

    args = parser.parse_args()
    models_to_plot = args.models_to_plot.split(",") if args.models_to_plot else None
    model_rename = dict(item.split(":") for item in args.model_rename.split(",")) if args.model_rename else None
    use_saved_data = args.use_saved_data and not args.force_recompute

    plot_forecast_with_ground_truth(
        csv_path=args.csv_path,
        accumulated_nc_path=args.accumulated_nc,
        output_path=args.output_path,
        k_factor=args.k_factor,
        baseline_model=args.baseline_model,
        models_to_plot=models_to_plot,
        model_rename=model_rename,
        use_saved_data=use_saved_data
    )

"""
python enhanced_6hr.py \
    --csv_path skill_score_proper1.csv \
    --accumulated_nc '/home/saptarishi.dhanuka_asp25/imerg_2014_6h_accumulated.nc' \
    --output_path plots/forecast_with_ground_truth6hr.png \
    --models_to_plot "Graphcast_Base_2014-08-01_2014-09-30,Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz" \
    --model_rename "Graphcast_Base_2014-08-01_2014-09-30:Base,Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz:Finetuned1" \
    --baseline_model Graphcast_Base_2014-08-01_2014-09-30
"""