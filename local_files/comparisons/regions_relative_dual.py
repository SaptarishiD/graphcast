import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy import stats


def parse_horizon_to_hours(horizon_str):
    """Converts a pandas timedelta string like 'X days HH:MM:SS' to total hours."""
    try:
        return pd.to_timedelta(horizon_str).total_seconds() / 3600
    except (ValueError, TypeError):
        return np.nan


def parse_precip_horizon_to_hours(horizon_str):
    """Converts forecast horizon to hours, handling various formats including nanoseconds."""
    try:
        # Handle nanoseconds format
        if 'nanoseconds' in str(horizon_str):
            # Extract numeric part and convert from nanoseconds to hours
            numeric_part = float(str(horizon_str).split()[0])
            return numeric_part / (3600 * 1e9)  # nanoseconds to hours
        else:
            # Handle pandas timedelta format
            return pd.to_timedelta(horizon_str).total_seconds() / 3600
    except (ValueError, TypeError):
        return np.nan


def plot_mse_and_precipitation(csv_path: str, precip_csv_path: str, output_path: str, 
                              k_factor: float = 2.0, baseline_model: str = "graphcast_base", 
                              models_to_plot: list[str] = None, model_rename: dict[str, str] = None,
                              precip_models_to_plot: list[str] = None):

    # ===== LOAD MSE DATA =====
    print(f"Loading MSE data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # --- Handle different horizon columns for MSE data ---
    if 'forecast_horizon_times' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_times'].apply(parse_horizon_to_hours)
    elif 'forecast_horizon_days' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_days'] * 24
    elif 'forecast_horizon' in df.columns:
        df['lead_time_hours'] = pd.to_timedelta(df['forecast_horizon']).dt.total_seconds() / 3600
    elif 'forecast_horizon_hours' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_hours']
    else:
        print("Error: Could not find a recognizable forecast horizon column in MSE data.")
        return

    # Drop bad rows
    df.dropna(subset=['lead_time_hours'], inplace=True)

    # --- Preserve region if present ---
    if 'region' not in df.columns:
        df['region'] = 'India'
    has_region = 'region' in df.columns

    # Filter models if requested
    if models_to_plot is not None:
        df = df[df['model'].isin(models_to_plot)]
        if df.empty:
            print(f"Warning: No matching models found in {models_to_plot}.")
            return

    # Sort for nicer output
    df = df.sort_values(by=['model', 'lead_time_hours'] + (['region'] if has_region else []))

    # ===== LOAD PRECIPITATION DATA =====
    print(f"Loading precipitation data from {precip_csv_path}...")
    try:
        precip_df = pd.read_csv(precip_csv_path)
    except FileNotFoundError:
        print(f"Error: The precipitation file {precip_csv_path} was not found.")
        return

    # Check if required columns exist
    if 'total_precip_sum' not in precip_df.columns:
        print("Error: 'total_precip_sum' column not found in the precipitation dataset.")
        return
    
    if 'forecast_horizon' not in precip_df.columns:
        print("Error: 'forecast_horizon' column not found in the precipitation dataset.")
        return

    # Convert precipitation forecast horizon to hours
    precip_df['lead_time_hours'] = precip_df['forecast_horizon'].apply(parse_precip_horizon_to_hours)
    precip_df.dropna(subset=['lead_time_hours'], inplace=True)

    # Handle region for precipitation data
    if 'region' not in precip_df.columns:
        precip_df['region'] = 'India'

    # Filter precipitation models if requested
    if precip_models_to_plot is not None:
        precip_df = precip_df[precip_df['model'].isin(precip_models_to_plot)]
        if precip_df.empty:
            print(f"Warning: No matching precipitation models found in {precip_models_to_plot}.")

    precip_df = precip_df.sort_values(by=['model', 'lead_time_hours'] + (['region'] if 'region' in precip_df.columns else []))

    # ===== CALCULATE SUMMARY STATISTICS FOR MSE =====
    group_cols = ['model'] + (['region'] if has_region else []) + ['lead_time_hours']
    summary_df = (
        df.groupby(group_cols)['mse']
          .agg(['mean', 'std', 'count'])
          .reset_index()
          .rename(columns={'mean': 'mse_mean', 'std': 'mse_std', 'count': 'n_samples'})
    )
    summary_df['mse_std'].fillna(0, inplace=True)

    # Compute 95% CIs for MSE
    z_score = 1.96
    summary_df['ci_half_width_standard'] = z_score * (summary_df['mse_std'] / np.sqrt(summary_df['n_samples']))
    summary_df['ci_half_width_corrected'] = k_factor * summary_df['ci_half_width_standard']

    # ===== CALCULATE SUMMARY STATISTICS FOR PRECIPITATION =====
    precip_group_cols = ['model'] + (['region'] if 'region' in precip_df.columns else []) + ['lead_time_hours']
    precip_summary_df = (
        precip_df.groupby(precip_group_cols)['total_precip_sum']
          .agg(['mean', 'std', 'count'])
          .reset_index()
          .rename(columns={'mean': 'precip_mean', 'std': 'precip_std', 'count': 'n_samples'})
    )
    precip_summary_df['precip_std'].fillna(0, inplace=True)

    print("\n--- MSE Summary Statistics ---")
    print(summary_df.head())
    print("\n--- Precipitation Summary Statistics ---")
    print(precip_summary_df.head())
    print("--------------------------\n")

    # ===== PLOTTING =====
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_top, ax_middle, ax_bottom) = plt.subplots(
        nrows=3, sharex=True, figsize=(14, 15),
        gridspec_kw={'height_ratios': [2, 2, 1]}
    )

    # Manual colors
    manual_colors = {
        "Base (India)": "tab:blue",
        "Finetuned (India)": "tab:orange",
        "Ground_Truth (India)": "tab:green"
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
            color=color, label=label
        )
        ax_top.fill_between(
            group_data['lead_time_hours'],
            group_data['mse_mean'] - group_data['ci_half_width_corrected'],
            group_data['mse_mean'] + group_data['ci_half_width_corrected'],
            color=color, alpha=0.2
        )

    handles, labels = ax_top.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_top.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)
    ax_top.set_title('Forecast Skill (MSE) vs. Lead Time with Corrected 95% CIs', fontsize=16, pad=20)
    ax_top.set_ylabel('Average MSE (Lower is better)', fontsize=12)
    ax_top.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_top.grid(True, which='both', linestyle='--', linewidth=0.5)

    # ----- MIDDLE: Total Precipitation -----
    precip_group_cols_for_plot = ['model', 'region'] if 'region' in precip_df.columns else ['model']
    for group_keys, group_data in precip_summary_df.groupby(precip_group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]

        display_name = model_rename.get(model_name, model_name) if model_rename else model_name
        label = display_name if region is None else f"{display_name} ({region})"

        color = manual_colors.get(label, "tab:purple")
        group_data = group_data.sort_values('lead_time_hours')

        ax_middle.plot(
            group_data['lead_time_hours'],
            group_data['precip_mean'],
            marker='s', linestyle='-',
            color=color, label=label, markersize=6
        )

    handles_precip, labels_precip = ax_middle.get_legend_handles_labels()
    by_label_precip = dict(zip(labels_precip, handles_precip))
    ax_middle.legend(by_label_precip.values(), by_label_precip.keys(), loc='upper left', fontsize=10)
    ax_middle.set_title('Total Precipitation Sum vs. Lead Time', fontsize=16, pad=20)
    ax_middle.set_ylabel('Total Precipitation Sum', fontsize=12)
    ax_middle.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_middle.grid(True, which='both', linestyle='--', linewidth=0.5)

    # ----- BOTTOM: % Improvement vs baseline -----
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

            color = 'tab:orange'
            gd = group_data.sort_values('lead_time_hours').dropna(subset=['pct_improvement'])

            if gd.empty:
                continue

            ax_bottom.plot(
                gd['lead_time_hours'],
                gd['pct_improvement'],
                marker='o', linestyle='-',
                color=color, label=label
            )

        ax_bottom.axhline(0.0, linestyle='--', linewidth=1)
        ax_bottom.set_ylabel(f'Improvement vs {baseline_model} (%)', fontsize=12)
        ax_bottom.grid(True, which='both', linestyle='--', linewidth=0.5)
    else:
        ax_bottom.text(0.5, 0.5,
                       f"Baseline '{baseline_model}' not found.\nSkipping improvement panel.",
                       ha='center', va='center', transform=ax_bottom.transAxes)
        ax_bottom.set_axis_off()

    # Shared x-axis formatting
    max_hours = max(summary_df['lead_time_hours'].max(), 
                   precip_summary_df['lead_time_hours'].max() if not precip_summary_df.empty else 0)
    ax_bottom.set_xlabel('Lead Time (hours)', fontsize=12)
    ax_bottom.set_xticks(np.arange(0, max_hours + 1, 24))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15)

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot weather forecast MSE and precipitation evaluation results."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV with MSE data.")
    parser.add_argument("--precip_csv_path", type=str, required=True, help="Path to input CSV with precipitation data.")
    parser.add_argument("--output_path", type=str, default="./plots/mse_precipitation_plot.png", help="Where to save the plot.")
    parser.add_argument("--k_factor", type=float, default=2.0, help="Autocorrelation inflation factor.")
    parser.add_argument("--baseline_model", type=str, default="graphcast_base",
                        help="Model name to use as the baseline for % improvement.")
    parser.add_argument("--models_to_plot", type=str, default=None,
                        help="Comma-separated list of models to plot for MSE. Example: 'graphcast_base,gfs'")
    parser.add_argument("--precip_models_to_plot", type=str, default=None,
                        help="Comma-separated list of models to plot for precipitation. Example: 'Ground_Truth,Model_A'")
    parser.add_argument("--model_rename", type=str, default=None,
                        help="Comma-separated mapping old:new. Example: 'graphcast_base:GraphCast,gfs:GFS'")

    args = parser.parse_args()

    models_to_plot = args.models_to_plot.split(",") if args.models_to_plot else None
    precip_models_to_plot = args.precip_models_to_plot.split(",") if args.precip_models_to_plot else None
    model_rename = dict(item.split(":") for item in args.model_rename.split(",")) if args.model_rename else None

    plot_mse_and_precipitation(
        args.csv_path,
        args.precip_csv_path,
        args.output_path,
        args.k_factor,
        baseline_model=args.baseline_model,
        models_to_plot=models_to_plot,
        model_rename=model_rename,
        precip_models_to_plot=precip_models_to_plot
    )

"""
Example usage:
python dual_plot.py \
    --csv_path skill_score_India_2025-08-1311-32-55_ACC.csv \
    --precip_csv_path precipitation_data.csv \
    --output_path plots/mse_precipitation_plot.png \
    --models_to_plot "Graphcast_Base,Graphcast_Finetuned1" \
    --precip_models_to_plot "Ground_Truth,Model_A,Model_B" \
    --model_rename "Graphcast_Base:Base,Graphcast_Finetuned1:Finetuned" \
    --baseline_model Graphcast_Base
"""