import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import xarray as xr
from scipy import stats
from datetime import datetime, timedelta
import glob

def parse_horizon_to_hours(horizon_str):
    """Converts a pandas timedelta string like 'X days HH:MM:SS' to total hours."""
    try:
        return pd.to_timedelta(horizon_str).total_seconds() / 3600
    except (ValueError, TypeError):
        return np.nan

def load_target_precipitation(target_files_dir, init_dates):
    """
    Load ground truth precipitation data for each initialization date.
    
    Args:
        target_files_dir: Directory containing target NetCDF files
        init_dates: List of initialization dates
    
    Returns:
        Dictionary mapping init_date to precipitation values array
    """
    target_data = {}
    
    for init_date in init_dates:
        # Format the filename - assuming format 'target_init_YYYY-MM-DD HH:MM:SS.nc'
        init_str = pd.to_datetime(init_date).strftime('%Y-%m-%d %H:%M:%S')
        filename = f'target_init_{init_str}.nc'
        filepath = os.path.join(target_files_dir, filename)
        
        try:
            # Load the NetCDF file
            dataset = xr.open_dataset(filepath)
            
            # Extract precipitation values using the provided code
            precip_values = dataset['total_precipitation_6hr'].sel(
                lat=slice(6, 38), 
                lon=slice(65, 95)
            ).sum(dim=['lat', 'lon']).values[0]
            
            target_data[init_date] = precip_values
            print(f"Loaded {len(precip_values)} precipitation values for {init_date}")
            
        except FileNotFoundError:
            print(f"Warning: Target file not found: {filepath}")
            continue
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return target_data

def create_target_dataframe(target_data):
    """
    Convert target precipitation data to DataFrame format matching the CSV structure.
    
    Args:
        target_data: Dictionary from load_target_precipitation
    
    Returns:
        DataFrame with columns: init_date, forecast_horizon, model, region, precipitation
    """
    rows = []
    
    for init_date, precip_values in target_data.items():
        for i, precip_val in enumerate(precip_values):
            # Calculate forecast horizon in nanoseconds (6-hour intervals)
            horizon_hours = (i + 1) * 6
            horizon_ns = horizon_hours * 3600 * 1e9  # Convert to nanoseconds
            
            rows.append({
                'init_date': init_date,
                'forecast_horizon': f'{int(horizon_ns)} nanoseconds',
                'model': 'Ground_Truth',
                'region': 'India',
                'precipitation': precip_val
            })
    
    return pd.DataFrame(rows)

def plot_forecast_with_ground_truth(csv_path: str, target_files_dir: str, output_path: str, 
                                   k_factor: float = 2.0, baseline_model: str = "Graphcast_Base", 
                                   models_to_plot: list[str] = None, model_rename: dict[str, str] = None):
    """
    Plot weather forecast results with ground truth precipitation data.
    """
    
    # 1. Load and preprocess the forecast data
    print(f"Loading forecast data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Handle different horizon columns
    if 'forecast_horizon_times' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_times'].apply(parse_horizon_to_hours)
    elif 'forecast_horizon_days' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_days'] * 24
    elif 'forecast_horizon' in df.columns:
        df['lead_time_hours'] = pd.to_timedelta(df['forecast_horizon']).dt.total_seconds() / 3600
    elif 'forecast_horizon_hours' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_hours']
    else:
        print("Error: Could not find a recognizable forecast horizon column.")
        return

    # Drop bad rows
    df.dropna(subset=['lead_time_hours'], inplace=True)

    # Preserve region if present
    if 'region' not in df.columns:
        df['region'] = 'India'
    has_region = 'region' in df.columns

    # 2. Load ground truth data
    print(f"Loading ground truth data from {target_files_dir}...")
    unique_init_dates = df['init_date'].unique()
    target_data = load_target_precipitation(target_files_dir, unique_init_dates)
    
    if target_data:
        target_df = create_target_dataframe(target_data)
        target_df['lead_time_hours'] = pd.to_timedelta(target_df['forecast_horizon']).dt.total_seconds() / 3600
        
        # Add ground truth to the main dataframe (but keep separate for plotting)
        print(f"Loaded ground truth data for {len(target_data)} initialization dates")
    else:
        print("Warning: No ground truth data loaded")
        target_df = pd.DataFrame()

    # Filter models if requested
    if models_to_plot is not None:
        df = df[df['model'].isin(models_to_plot)]
        if df.empty:
            print(f"Warning: No matching models found in {models_to_plot}.")
            return

    # Sort for nicer output
    df = df.sort_values(by=['model', 'lead_time_hours'] + (['region'] if has_region else []))

    # 3. Calculate summary statistics for forecast models
    group_cols = ['model'] + (['region'] if has_region else []) + ['lead_time_hours']
    summary_df = (
        df.groupby(group_cols)['mse']
          .agg(['mean', 'std', 'count'])
          .reset_index()
          .rename(columns={'mean': 'mse_mean', 'std': 'mse_std', 'count': 'n_samples'})
    )
    summary_df['mse_std'].fillna(0, inplace=True)

    # Calculate confidence intervals
    z_score = 1.96
    summary_df['ci_half_width_standard'] = z_score * (summary_df['mse_std'] / np.sqrt(summary_df['n_samples']))
    summary_df['ci_half_width_corrected'] = k_factor * summary_df['ci_half_width_standard']

    # 4. Calculate summary statistics for ground truth precipitation
    if not target_df.empty:
        target_summary = (
            target_df.groupby(['model', 'region', 'lead_time_hours'])['precipitation']
              .agg(['mean', 'std', 'count'])
              .reset_index()
              .rename(columns={'mean': 'precip_mean', 'std': 'precip_std', 'count': 'n_samples'})
        )
        target_summary['precip_std'].fillna(0, inplace=True)
        target_summary['ci_half_width'] = z_score * (target_summary['precip_std'] / np.sqrt(target_summary['n_samples']))

    print("\n--- Forecast Summary Statistics ---")
    print(summary_df.head())
    if not target_df.empty:
        print("\n--- Ground Truth Summary Statistics ---")
        print(target_summary.head())
    print("--------------------------\n")

    # 5. Create plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create three panels: MSE, % improvement, and precipitation comparison
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1.5], hspace=0.3)
    
    ax_top = fig.add_subplot(gs[0])
    ax_middle = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bottom = fig.add_subplot(gs[2], sharex=ax_top)

    # Color mapping
    manual_colors = {
        "Base (India)": "tab:blue",
        "Finetuned1 (India)": "tab:orange",
        "Graphcast_Finetuned2 (India)": "tab:green"
    }

    # ----- TOP: MSE + CI -----
    group_cols_for_plot = ['model', 'region'] if has_region else ['model']
    for group_keys, group_data in summary_df.groupby(group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]

        display_name = model_rename.get(model_name, model_name) if model_rename else model_name
        label = display_name if region is None else f"{display_name} ({region})"

        if model_name == 'HRES':
            continue

        color = manual_colors.get(label, "black")
        group_data = group_data.sort_values('lead_time_hours')

        ax_top.plot(
            group_data['lead_time_hours'],
            group_data['mse_mean'],
            marker='o', linestyle='-',
            color=color, label=label, linewidth=2
        )
        ax_top.fill_between(
            group_data['lead_time_hours'],
            group_data['mse_mean'] - group_data['ci_half_width_corrected'],
            group_data['mse_mean'] + group_data['ci_half_width_corrected'],
            color=color, alpha=0.2
        )

    ax_top.set_title('Forecast Skill (MSE) vs. Lead Time with 95% CIs', fontsize=16, pad=20)
    ax_top.set_ylabel('Average MSE (Lower is better)', fontsize=12)
    ax_top.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_top.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_top.legend(loc='upper left', fontsize=10)

    # ----- MIDDLE: % Improvement vs baseline -----
    if (summary_df['model'] == baseline_model).any():
        merge_keys = ['lead_time_hours'] + (['region'] if has_region else [])
        base = (
            summary_df[summary_df['model'] == baseline_model]
            [merge_keys + ['mse_mean']]
            .rename(columns={'mse_mean': 'mse_base'})
        )

        joined = summary_df.merge(base, on=merge_keys, how='left')
        joined['pct_improvement'] = np.where(
            joined['mse_base'] > 0,
            100.0 * (-joined['mse_mean'] + joined['mse_base']) / joined['mse_base'],
            np.nan
        )

        for group_keys, group_data in joined.groupby(group_cols_for_plot):
            model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
            region = None if isinstance(group_keys, str) else group_keys[1]

            display_name = model_rename.get(model_name, model_name) if model_rename else model_name
            label = display_name if region is None else f"{display_name} ({region})"

            if model_name in [baseline_model, 'HRES']:
                continue

            color = manual_colors.get(label, "black")
            gd = group_data.sort_values('lead_time_hours').dropna(subset=['pct_improvement'])

            if gd.empty:
                continue

            ax_middle.plot(
                gd['lead_time_hours'],
                gd['pct_improvement'],
                marker='o', linestyle='-',
                color=color, label=label, linewidth=2
            )

        ax_middle.axhline(0.0, linestyle='--', linewidth=1, color='gray')
        ax_middle.set_ylabel(f'Improvement vs {baseline_model} (%)', fontsize=12)
        ax_middle.grid(True, which='both', linestyle='--', linewidth=0.5)

    # ----- BOTTOM: Ground Truth Precipitation -----
    if not target_df.empty:
        for group_keys, group_data in target_summary.groupby(['model', 'region']):
            model_name, region = group_keys
            label = f"{model_name} ({region})"
            color = manual_colors.get(label, "tab:green")
            
            group_data = group_data.sort_values('lead_time_hours')
            
            ax_bottom.plot(
                group_data['lead_time_hours'],
                group_data['precip_mean'],
                marker='s', linestyle='-',
                color=color, label=label, linewidth=2
            )
            ax_bottom.fill_between(
                group_data['lead_time_hours'],
                group_data['precip_mean'] - group_data['ci_half_width'],
                group_data['precip_mean'] + group_data['ci_half_width'],
                color=color, alpha=0.2
            )
        
        ax_bottom.set_title('Ground Truth Precipitation vs. Lead Time', fontsize=14, pad=10)
        ax_bottom.set_ylabel('Precipitation (mm)', fontsize=12)
        ax_bottom.legend(loc='upper right', fontsize=10)
    else:
        ax_bottom.text(0.5, 0.5, "No ground truth data available", 
                      ha='center', va='center', transform=ax_bottom.transAxes)
        ax_bottom.set_ylabel('Precipitation (mm)', fontsize=12)

    ax_bottom.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Shared x-axis formatting
    max_hours = summary_df['lead_time_hours'].max()
    ax_bottom.set_xlabel('Lead Time (hours)', fontsize=12)
    ax_bottom.set_xticks(np.arange(0, max_hours + 1, 24))

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot weather forecast evaluation results with ground truth precipitation data."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV.")
    parser.add_argument("--target_files_dir", type=str, required=True, 
                        help="Directory containing target NetCDF files.")
    parser.add_argument("--output_path", type=str, default="./plots/forecast_with_truth.png", 
                        help="Where to save the plot.")
    parser.add_argument("--k_factor", type=float, default=2.0, help="Autocorrelation inflation factor.")
    parser.add_argument("--baseline_model", type=str, default="Graphcast_Base",
                        help="Model name to use as the baseline for % improvement.")
    parser.add_argument("--models_to_plot", type=str, default=None,
                        help="Comma-separated list of models to plot.")
    parser.add_argument("--model_rename", type=str, default=None,
                        help="Comma-separated mapping old:new. Example: 'Graphcast_Base:Base'")

    args = parser.parse_args()

    models_to_plot = args.models_to_plot.split(",") if args.models_to_plot else None
    model_rename = dict(item.split(":") for item in args.model_rename.split(",")) if args.model_rename else None

    plot_forecast_with_ground_truth(
        args.csv_path,
        args.target_files_dir,
        args.output_path,
        args.k_factor,
        baseline_model=args.baseline_model,
        models_to_plot=models_to_plot,
        model_rename=model_rename
    )

"""
Example usage:
python enhanced_weather_plot.py \
    --csv_path skill_score_proper1.csv \
    --target_files_dir '/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds' \
    --output_path plots/forecast_with_ground_truth.png \
    --models_to_plot "Graphcast_Base,Graphcast_Finetuned1" \
    --model_rename "Graphcast_Base:Base,Graphcast_Finetuned1:Finetuned1" \
    --baseline_model Graphcast_Base
"""