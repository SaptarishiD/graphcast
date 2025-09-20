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

def plot_mse_with_confidence_intervals(csv_path: str, output_path: str, k_factor: float = 2.0):
    # 1. Load and preprocess the data
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # --- Handle different horizon columns ---
    if 'forecast_horizon_times' in df.columns:
        # original format: "0 days 06:00:00"
        df['lead_time_hours'] = df['forecast_horizon_times'].apply(parse_horizon_to_hours)
    elif 'forecast_horizon_days' in df.columns:
        # alternative format: days as integer
        df['lead_time_hours'] = df['forecast_horizon_days'] * 24
    elif 'forecast_horizon_hours' in df.columns:
        # here the horizon is given in hours like 6 hours 12 hours etc
        df['lead_time_hours'] = df['forecast_horizon_hours']



    elif 'forecast_horizon' in df.columns:
        # *** CHANGE: parse nanosecond‐based horizon
        # this will handle both strings like "21600000000000 nanoseconds"
        # and pure integer nanosecond values
        df['lead_time_hours'] = (
            pd.to_timedelta(df['forecast_horizon'])
              .dt.total_seconds() / 3600
        )
    else:
        print("Error: Could not find a recognizable forecast horizon column.")
        return

    # drop any bad rows
    df.dropna(subset=['lead_time_hours'], inplace=True)

    # --- Preserve region if present ---
    if 'region' not in df.columns:
        df['region'] = 'India'

    has_region = 'region' in df.columns

    # Sort for nicer output
    df = df.sort_values(by=['model', 'lead_time_hours'] + (['region'] if has_region else []))

    # 2. Calculate summary statistics
    group_cols = ['model']
    if has_region:
        group_cols.append('region')  # *** CHANGE: group by region too
    group_cols.append('lead_time_hours')

    summary_df = (
        df
        .groupby(group_cols)['mse']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mse_mean', 'std': 'mse_std', 'count': 'n_samples'})
    )

    

    # fill NaN std (single sample)
    summary_df['mse_std'].fillna(0, inplace=True)

    # 3. Compute 95% CIs
    z_score = 1.96
    summary_df['ci_half_width_standard'] = z_score * (summary_df['mse_std'] / np.sqrt(summary_df['n_samples']))
    summary_df['ci_half_width_corrected'] = k_factor * summary_df['ci_half_width_standard']

    print("\n--- Summary Statistics ---")
    print(summary_df.head())
    print("--------------------------\n")

    # 4. Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # prepare palette
    labels = summary_df['model'].unique()
    if has_region:
        # combine model+region for distinct lines
        labels = summary_df[['model','region']].drop_duplicates().apply(lambda r: f"{r['model']} ({r['region']})", axis=1)
    palette = sns.color_palette("colorblind", n_colors=len(labels))
    color_map = dict(zip(labels, palette))

    group_cols_for_plot = ['model', 'region'] if has_region else ['model']


        # --- NEW: Compute percentage improvement vs. graphcast_base ---
    baseline_df = summary_df[summary_df['model'] == 'graphcast_base']
    merge_cols = ['lead_time_hours'] + (['region'] if has_region else [])

    improvement_df = summary_df.merge(
        baseline_df[merge_cols + ['mse_mean']],
        on=merge_cols,
        suffixes=('', '_baseline')
    )

    improvement_df['pct_improvement'] = (
        (improvement_df['mse_mean_baseline'] - improvement_df['mse_mean'])
        / improvement_df['mse_mean_baseline']
    ) * 100

    # --- Plot: MSE and % improvement side by side ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # MSE plot (same as before, but on ax1)
    for group_keys, group_data in summary_df.groupby(group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]
        label = model_name if region is None else f"{model_name} ({region})"
        if model_name == 'HRES':
            continue
        color = color_map[label]
        group_data = group_data.sort_values('lead_time_hours')
        ax1.plot(
            group_data['lead_time_hours'], group_data['mse_mean'],
            marker='o', linestyle='-', color=color, label=label
        )
        ax1.fill_between(
            group_data['lead_time_hours'],
            group_data['mse_mean'] - group_data['ci_half_width_corrected'],
            group_data['mse_mean'] + group_data['ci_half_width_corrected'],
            color=color, alpha=0.2
        )
    ax1.set_title('Forecast Skill (MSE) vs. Lead Time', fontsize=16, pad=20)
    ax1.set_ylabel('Average MSE', fontsize=12)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # % improvement plot on ax2
    for group_keys, group_data in improvement_df.groupby(group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        if model_name in ['graphcast_base', 'HRES']:
            continue
        region = None if isinstance(group_keys, str) else group_keys[1]
        label = model_name if region is None else f"{model_name} ({region})"
        color = color_map[label]
        group_data = group_data.sort_values('lead_time_hours')
        ax2.plot(
            group_data['lead_time_hours'], group_data['pct_improvement'],
            marker='o', linestyle='-', color=color, label=label
        )
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_title('Percentage Improvement vs. graphcast_base', fontsize=16, pad=20)
    ax2.set_xlabel('Lead Time (hours)', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Combined legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    max_hours = summary_df['lead_time_hours'].max()
    ax2.set_xticks(np.arange(0, max_hours + 1, 24))

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    # plt.show()


    # for group_keys, group_data in summary_df.groupby(group_cols_for_plot):
    #     model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
    #     print(group_keys)
    #     region = None if isinstance(group_keys, str) else group_keys[1]
        
    #     label = model_name if region is None else f"{model_name} ({region})"
    #     if model_name == 'HRES':
    #         continue

    #     color = color_map[label]
    #     group_data = group_data.sort_values('lead_time_hours')

    #     ax.plot(
    #         group_data['lead_time_hours'],
    #         group_data['mse_mean'],
    #         marker='o',
    #         linestyle='-',
    #         color=color,
    #         label=label
    #     )
    #     ax.fill_between(
    #         group_data['lead_time_hours'],
    #         group_data['mse_mean'] - group_data['ci_half_width_corrected'],
    #         group_data['mse_mean'] + group_data['ci_half_width_corrected'],
    #         color=color,
    #         alpha=0.2
    #     )

    # # remove duplicate legend entries
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    # ax.set_title('Forecast Skill (MSE) vs. Lead Time with Corrected 95% CIs', fontsize=16, pad=20)
    # ax.set_xlabel('Lead Time (hours)', fontsize=12)
    # ax.set_ylabel('Average MSE', fontsize=12)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # max_hours = summary_df['lead_time_hours'].max()
    # ax.set_xticks(np.arange(0, max_hours + 1, 24))
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # fig.tight_layout()

    # # save
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # plt.savefig(output_path, dpi=300)
    # print(f"Plot saved to {output_path}")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot weather forecast evaluation results with confidence intervals."
    )
    parser.add_argument("--csv_path",    type=str, required=True,
                        help="Path to input CSV.")
    parser.add_argument("--output_path", type=str, default="./plots/mse_plot.png",
                        help="Where to save the plot.")
    parser.add_argument("--k_factor",    type=float, default=2.0,
                        help="Autocorrelation inflation factor.")
    args = parser.parse_args()

    plot_mse_with_confidence_intervals(args.csv_path, args.output_path, args.k_factor)
