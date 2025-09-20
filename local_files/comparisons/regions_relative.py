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

def plot_mse_with_confidence_intervals(csv_path: str, output_path: str, k_factor: float = 2.0, baseline_model: str = "graphcast_base", models_to_plot: list[str] = None, model_rename: dict[str, str] = None, var='mse'):




    # 1. Load and preprocess the data
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return


    # --- Handle different horizon columns ---
    if 'forecast_horizon_times' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_times'].apply(parse_horizon_to_hours)

    elif 'forecast_horizon_days' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_days'] * 24

    elif 'forecast_horizon' in df.columns:
        df['lead_time_hours'] = pd.to_timedelta(df['forecast_horizon']).dt.total_seconds() / 3600
    elif 'forecast_horizon_hours' in df.columns:
        # here the horizon is given in hours like 6 hours 12 hours etc
        df['lead_time_hours'] = df['forecast_horizon_hours']
    elif 'lead_time' in df.columns:
        df['lead_time_hours'] = df['lead_time']
    else:
        print("Error: Could not find a recognizable forecast horizon column.")
        return

    # drop bad rows
    df.dropna(subset=['lead_time_hours'], inplace=True)

    # --- Preserve region if present (default to single-region) ---
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

    # 2. Calculate summary statistics
    group_cols = ['model'] + (['region'] if has_region else []) + ['lead_time_hours']
    summary_df = (
        df.groupby(group_cols)[var]
          .agg(['mean', 'std', 'count'])
          .reset_index()
          .rename(columns={'mean': 'mse_mean', 'std': 'mse_std', 'count': 'n_samples'})
    )
    summary_df['mse_std'].fillna(0, inplace=True)

    # 3. Compute 95% CIs (with inflation factor)
    z_score = 1.96
    summary_df['ci_half_width_standard'] = z_score * (summary_df['mse_std'] / np.sqrt(summary_df['n_samples']))
    summary_df['ci_half_width_corrected'] = k_factor * summary_df['ci_half_width_standard']

    print("\n--- Summary Statistics ---")
    print(summary_df.head())
    print("--------------------------\n")

    # 4. Plot (two panels: top = MSE+CI, bottom = % improvement vs baseline)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2, sharex=True, figsize=(14, 10),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # prepare palette (consistent colors across panels)
    if has_region:
        label_series = summary_df[['model', 'region']].drop_duplicates().apply(lambda r: f"{r['model']} ({r['region']})", axis=1)
    else:
        label_series = summary_df['model'].drop_duplicates()

    palette = sns.color_palette("colorblind", n_colors=len(label_series))
    color_map = dict(zip(label_series, palette))

    manual_colors = {
    "Base (India)": "tab:blue",
    "Finetuned (India)": "tab:orange"
}
    manual_colors_list = ['tab:blue', 'tab:orange', 'tab:purple']

    # ----- TOP: MSE + CI -----
    count = 0
    group_cols_for_plot = ['model', 'region'] if has_region else ['model']
    for group_keys, group_data in summary_df.groupby(group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]

        # apply renaming if provided
        display_name = model_rename.get(model_name, model_name) if model_rename else model_name
        label = display_name if region is None else f"{display_name} ({region})"


        if model_name == 'HRES':
            continue

        # color = color_map.get(label, None)
        print(label)
        # color = manual_colors.get(label, "black")
        color = manual_colors_list[count]
        count+=1

        group_data = group_data.sort_values('lead_time_hours')
        print(model_name, color)

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

    # remove duplicate legend entries
    handles, labels = ax_top.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_top.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    ax_top.set_title('Forecast Skill (RMSE) vs. Lead Time with Corrected 95% CIs', fontsize=16, pad=20)
    ax_top.set_ylabel('Average RMSE (Lower is better)', fontsize=12)
    ax_top.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_top.grid(True, which='both', linestyle='--', linewidth=0.5)

    # ----- BOTTOM: % Improvement vs baseline -----
    if (summary_df['model'] == baseline_model).any():
        merge_keys = ['lead_time_hours'] + (['region'] if has_region else [])
        base = (
            summary_df[summary_df['model'] == baseline_model]
            [merge_keys + ['mse_mean']]
            .rename(columns={'mse_mean': 'mse_base'})
        )

        # join per (lead_time[, region]); drop baseline itself later
        joined = summary_df.merge(base, on=merge_keys, how='left')

        # compute % improvement; avoid divide-by-zero
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
            print(model_name)
            color = "tab:olive" if "finetuned1" in model_name.lower() else "tab:purple"
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

    # shared x-axis formatting
    max_hours = summary_df['lead_time_hours'].max()
    ax_bottom.set_xlabel('Lead Time (hours)', fontsize=12)
    ax_bottom.set_xticks(np.arange(0, max_hours + 1, 24))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15)

    # save  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot weather forecast evaluation results with confidence intervals and % improvement vs baseline."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV.")
    parser.add_argument("--output_path", type=str, default="./plots/mse_plot.png", help="Where to save the plot.")
    parser.add_argument("--k_factor", type=float, default=2.0, help="Autocorrelation inflation factor.")
    parser.add_argument("--baseline_model", type=str, default="graphcast_base",
                        help="Model name to use as the baseline for % improvement.")
    parser.add_argument("--models_to_plot", type=str, default=None,
                        help="Comma-separated list of models to plot. Example: 'graphcast_base,gfs'")
    parser.add_argument("--model_rename", type=str, default=None,
                        help="Comma-separated mapping old:new. Example: 'graphcast_base:GraphCast,gfs:GFS'")
    
    parser.add_argument("--vars_to_eval", type=str, required=True, help="Var Eval")

    

    args = parser.parse_args()

    models_to_plot = args.models_to_plot.split(",") if args.models_to_plot else None
    model_rename = dict(item.split(":") for item in args.model_rename.split(",")) if args.model_rename else None
    var_to_eval = args.vars_to_eval

    plot_mse_with_confidence_intervals(
        args.csv_path,
        args.output_path,
        args.k_factor,
        baseline_model=args.baseline_model,
        models_to_plot=models_to_plot,
        model_rename=model_rename,
        var=var_to_eval
    )


"""
python regions_relative.py \
    --csv_path skill_score_India_2025-08-2110-39-16_ACC.csv \
    --output_path plots/acc_plot_choice_2014.png \
    --models_to_plot "Graphcast_Base_2014,Graphcast_Finetuned1_2014,Graphcast_Finetuned2_2014" \
    --model_rename "Graphcast_Base_2014:Base,Graphcast_Finetuned1_2014: Finetuned1,Graphcast_Finetuned2_2014:Finetuned2" \
    --baseline_model Graphcast_Base_2014 \
    --vars_to_eval "acc"


"""
"""
python regions_relative.py \
    --csv_path skill_score_India_2025-08-2021-42-34.csv \
    --output_path plots/mse_plot_choice_2014.png \
    --models_to_plot "Graphcast_Base_2014-08-01_2014-09-30,Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz, Graphcast_Finetuned_2_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_mask_expt3_good.npz,Graphcast_Finetuned_3_graphcast_1_13_orig_2013-06-08_2013-09-23_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_mask_expt3.npz" \
    --model_rename "Graphcast_Base_2014-08-01_2014-09-30:Base,Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz:Finetuned1, Graphcast_Finetuned_2_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_mask_expt3_good.npz:Finetuned2, Graphcast_Finetuned_3_graphcast_1_13_orig_2013-06-08_2013-09-23_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_mask_expt3.npz:Finetuned3" \
    --baseline_model Graphcast_Base_2014-08-01_2014-09-30
"""