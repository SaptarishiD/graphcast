# plot_with_imerg_targets.py
import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import pandas as pd
import numpy as np
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
                sel = ds_india[varname].sel(time=np.datetime64(desired_time))
                used_time = np.datetime64(desired_time)
                used_method = 'exact'
            except (KeyError, IndexError, ValueError):
                try:
                    sel = ds_india[varname].sel(time=desired_time, method='nearest', tolerance=time_tolerance)
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
    result.to_csv('test_target.csv', index=False)
    return pd.DataFrame(rows)


def plot_forecast_with_ground_truth(csv_path: str,
                                    accumulated_nc_path: str,
                                    output_path: str,
                                    k_factor: float = 2.0,
                                    baseline_model: str = "Graphcast_Base",
                                    models_to_plot: list = None,
                                    model_rename: dict = None,
                                    use_saved_data: bool = True,
                                    plot_var: str = None):
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

    if (not use_saved_data) or (not os.path.exists('test_target.csv')):
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
        target_df = create_target_dataframe(target_data) if target_data else pd.DataFrame()
    else:
        target_df = pd.read_csv('test_target.csv')
        target_df['lead_time_hours'] = pd.to_timedelta(target_df['forecast_horizon']).dt.total_seconds() / 3600
        print(f"Loaded ground truth for initializations (after month>=8 filter).")

    # 3. summary stats for forecast models (uses rmse column)
    group_cols = ['model'] + (['region'] if has_region else []) + ['lead_time_hours']
    summary_df = (
        df.groupby(group_cols)[plot_var]
          .agg(['mean', 'std', 'count'])
          .reset_index()
          .rename(columns={'mean': f'{plot_var}_mean', 'std': f'{plot_var}_std', 'count': 'n_samples'})
    )
    summary_df[f'{plot_var}_std'].fillna(0, inplace=True)

    # confidence intervals
    z_score = 1.96
    summary_df['ci_half_width_standard'] = z_score * (summary_df[f'{plot_var}_std'] / np.sqrt(summary_df['n_samples'].clip(lower=1)))
    summary_df['ci_half_width_corrected'] = k_factor * summary_df['ci_half_width_standard']

    # 4. Ground truth - calculate total precipitation at each lead time (no mean/std)
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not target_df.empty:
        precip_totals = target_df.groupby('lead_time_hours')['precipitation'].sum().reset_index()
        precip_totals['model'] = 'Ground_Truth'
        precip_totals['region'] = 'India'
        precip_totals.to_csv(f'precipitation_totals_{current_date}.csv', index=False)
        print("Saved precipitation totals to CSV.")
    else:
        precip_totals = pd.DataFrame()

    # Save summary statistics for future use
    summary_df.to_csv(f'forecast_summary_stats_{current_date}.csv', index=False)
    print("Saved forecast summary statistics to CSV.")

    # 5. Plotting - MSE (left) + precipitation (right)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.25)

    # Top panel: MSE (left axis) + precipitation (right axis)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = ax_left.twinx()

    manual_colors = {
        "Graphcast_Base_2014-08-01_2014-09-30": "tab:blue",
        "Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz": "tab:orange",
        "Graphcast_Finetuned2 (India)": "tab:green",
        "Base": "tab:blue",
        "Finetuned1": "tab:orange"
    }

    group_cols_for_plot = ['model', 'region'] if has_region else ['model']

    # For % improvement computation, pre-slice baseline by region (if any)
    baseline_mask = (summary_df['model'] == baseline_model)
    if has_region:
        baseline_by_region = {
            r: df_r.sort_values('lead_time_hours')[['lead_time_hours', f'{plot_var}_mean']].rename(columns={f'{plot_var}_mean': f'{plot_var}_base'})
            for r, df_r in summary_df[baseline_mask].groupby('region')
        }
    else:
        base_df_only = summary_df[baseline_mask].sort_values('lead_time_hours')[
            ['lead_time_hours', f'{plot_var}_mean']
        ].rename(columns={f'{plot_var}_mean': f'{plot_var}_base'})

    # Top panel: plot the MSE curves
    line_handles, line_labels = [], []
    for group_keys, group_data in summary_df.groupby(group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]
        display_name = model_rename.get(model_name, model_name) if model_rename else model_name
        label = display_name if region is None else f"{display_name} ({region})"

        if model_name == 'HRES':
            continue

        color = manual_colors.get(label, manual_colors.get(display_name, "black"))
        group_data = group_data.sort_values('lead_time_hours')

        lh, = ax_left.plot(
            group_data['lead_time_hours'],
            group_data[f'{plot_var}_mean'],
            marker='o', linestyle='-', color=color, label=label, linewidth=2
        )
        ax_left.fill_between(
            group_data['lead_time_hours'],
            group_data[f'{plot_var}_mean'] - group_data['ci_half_width_corrected'],
            group_data[f'{plot_var}_mean'] + group_data['ci_half_width_corrected'],
            color=color, alpha=0.1
        )
        line_handles.append(lh)
        line_labels.append(label)

    # Top panel: precipitation on right axis
    if not precip_totals.empty:
        rh, = ax_right.plot(
            precip_totals['lead_time_hours'],
            precip_totals['precipitation'],
            linestyle='-',
            label='Ground Truth Total Precipitation',
            linewidth=3, alpha=0.1, color='red'
        )
        line_handles.append(rh)
        line_labels.append('Ground Truth Total Precipitation')

    ax_left.set_title(f'Forecast Skill {plot_var.upper()} and Ground Truth Precipitation vs. Lead Time', fontsize=16, pad=20)
    if plot_var == 'acc':
        which_better = 'higher'
    elif plot_var == 'rmse' or plot_var == 'mse':
        which_better = 'lower'
    ax_left.set_ylabel(f'Average {plot_var.upper()} ({which_better} is better)', fontsize=12, color='black')
    ax_left.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_left.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax_right.set_ylabel('Total Precipitation', fontsize=12, color='black')
    ax_right.tick_params(axis='y', labelcolor='black')

    max_hours = summary_df['lead_time_hours'].max() if not summary_df.empty else 168
    ax_left.set_xlabel('Lead Time (hours)', fontsize=12)
    ax_left.set_xticks(np.arange(0, max_hours + 1, 24))

    ax_left.legend(line_handles, line_labels, loc='upper left', fontsize=10)

    # Bottom panel: % improvement vs baseline
    ax_imp = fig.add_subplot(gs[1, 0], sharex=ax_left)

    # Plot % improvement for each non-baseline model
    for group_keys, group_data in summary_df.groupby(group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]
        if model_name == baseline_model or model_name == 'HRES':
            continue

        display_name = model_rename.get(model_name, model_name) if model_rename else model_name
        label = display_name if region is None else f"{display_name} ({region})"
        color = manual_colors.get(label, manual_colors.get(display_name, "black"))
        group_data = group_data.sort_values('lead_time_hours')[['lead_time_hours', f'{plot_var}_mean']].rename(columns={f'{plot_var}_mean': f'{plot_var}_model'})

        # Choose the matching baseline (region-aware if applicable)
        if has_region:
            base_df = baseline_by_region.get(region)
            if base_df is None or base_df.empty:
                # no matching baseline for this region; skip
                continue
        else:
            base_df = base_df_only

        merged = pd.merge(group_data, base_df, on='lead_time_hours', how='inner')
        # Avoid division by zero: if base mse is 0, set improvement to 0
        denom = merged[f'{plot_var}_base'].replace(0, np.nan)
        merged['improvement_pct'] = 100.0 * (merged[f'{plot_var}_base'] - merged[f'{plot_var}_model']) / denom
        merged['improvement_pct'] = merged['improvement_pct'].fillna(0.0)

        ax_imp.plot(
            merged['lead_time_hours'],
            merged['improvement_pct'],
            marker='o', linestyle='-', linewidth=2, color=color, label=label
        )

    ax_imp.axhline(0.0, linestyle='--', linewidth=1.0, color='gray')
    ax_imp.set_ylabel('% Improvement vs. Baseline', fontsize=12)
    ax_imp.set_xlabel('Lead Time (hours)', fontsize=12)
    ax_imp.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Optional: keep the legend only for the bottom panel if you prefer
    ax_imp.legend(loc='upper right', fontsize=10)

    # plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.show()


# ---------------------- YAML config loader & entrypoint -----------------------
def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

def main():
    # config path: sys.argv[1] if provided, else "config.yaml"
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)

    # Required
    csv_path = cfg["csv_path"]
    accumulated_nc = cfg["accumulated_nc"]
    output_path = cfg.get("output_path", "./plots/forecast_with_truth.png")

    # Optional
    k_factor = float(cfg.get("k_factor", 2.0))
    baseline_model = cfg.get("baseline_model", "Graphcast_Base")
    models_to_plot = cfg.get("models_to_plot")  # list or None
    model_rename = cfg.get("model_rename")      # dict or None
    plot_var = cfg.get("plot_var")

    # Saved-data behavior
    use_saved_data = bool(cfg.get("use_saved_data", True))
    force_recompute = bool(cfg.get("force_recompute", False))
    use_saved_data = use_saved_data and (not force_recompute)

    plot_forecast_with_ground_truth(
        csv_path=csv_path,
        accumulated_nc_path=accumulated_nc,
        output_path=output_path,
        k_factor=k_factor,
        baseline_model=baseline_model,
        models_to_plot=models_to_plot,
        model_rename=model_rename,
        use_saved_data=use_saved_data,
        plot_var = plot_var
    )

if __name__ == "__main__":
    main()
