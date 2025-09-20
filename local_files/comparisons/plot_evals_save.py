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
        # pd.to_timedelta is powerful and can parse this format directly
        return pd.to_timedelta(horizon_str).total_seconds() / 3600
    except (ValueError, TypeError):
        return np.nan

def plot_mse_with_confidence_intervals(csv_path: str, output_path: str, k_factor: float = 2.0):
    """
    Loads forecast evaluation data, calculates mean MSE and confidence intervals,
    and plots the results.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path to save the output plot image.
        k_factor (float): The inflation factor 'k' to correct for temporal
                          autocorrelation, as described by Geer (2016).
                          A value of 1.0 means no correction. Common values
                          in weather forecasting can range from 1.2 to 3.0 or more.
    """
    # 1. Load and preprocess the data
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Convert forecast horizon to a numerical format (hours)
    # The column name in your sample is 'forecast_horizon_times', adjust if different
    if 'forecast_horizon_times' in df.columns:
        df['lead_time_hours'] = df['forecast_horizon_times'].apply(parse_horizon_to_hours)
    elif 'forecast_horizon_days' in df.columns: # Handle the other format
        df['lead_time_hours'] = df['forecast_horizon_days'] * 24
    else:
        print("Error: Could not find a recognizable forecast horizon column.")
        return
        
    df.dropna(subset=['lead_time_hours'], inplace=True)
    df = df.sort_values(by=['model', 'lead_time_hours'])

    # 2. Calculate summary statistics (mean, std, count) for each group
    # We group by model and lead time to get statistics over all initialization dates
    summary_df = df.groupby(['model', 'lead_time_hours'])['mse'].agg(['mean', 'std', 'count']).reset_index()
    summary_df.rename(columns={'mean': 'mse_mean', 'std': 'mse_std', 'count': 'n_samples'}, inplace=True)
    
    # Handle cases with only one sample where std is NaN
    summary_df['mse_std'].fillna(0, inplace=True)

    # 3. Calculate the 95% Confidence Interval
    # z-score for 95% confidence is 1.96
    z_score = 1.96
    
    # Standard (uncorrected) confidence interval half-width
    summary_df['ci_half_width_standard'] = z_score * (summary_df['mse_std'] / np.sqrt(summary_df['n_samples']))
    
    # Corrected confidence interval using the inflation factor 'k'
    # This accounts for temporal autocorrelation in the forecast errors
    summary_df['ci_half_width_corrected'] = k_factor * summary_df['ci_half_width_standard']

    print("\n--- Summary Statistics ---")
    print(summary_df.head())
    print("--------------------------\n")

    # 4. Plot the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use a color palette for clarity
    palette = sns.color_palette("colorblind", n_colors=summary_df['model'].nunique())
    model_colors = {model: color for model, color in zip(summary_df['model'].unique(), palette)}

    for model_name in summary_df['model'].unique():
        if model_name == 'HRES':
            continue
        model_data = summary_df[summary_df['model'] == model_name]
        color = model_colors[model_name]
        
        # Plot the mean MSE line
        ax.plot(
            model_data['lead_time_hours'],
            model_data['mse_mean'],
            label=f"{model_name} (Mean MSE)",
            color=color,
            marker='o',
            markersize=4,
            linestyle='-'
        )
        
        # Plot the corrected confidence interval as a shaded region
        ax.fill_between(
            model_data['lead_time_hours'],
            model_data['mse_mean'] - model_data['ci_half_width_corrected'],
            model_data['mse_mean'] + model_data['ci_half_width_corrected'],
            color=color,
            alpha=0.2,
            label=f"{model_name} (95% CI with k={k_factor})" if 'Graphcast' in model_name else None # Avoid cluttered legend
        )

    # 5. Finalize and save the plot
    ax.set_title('Forecast Skill (MSE) vs. Lead Time with Corrected 95% Confidence Intervals', fontsize=16, pad=20)
    ax.set_xlabel('Lead Time (hours)', fontsize=12)
    ax.set_ylabel('Average Mean Squared Error (MSE)', fontsize=12)
    
    # Format y-axis to scientific notation for readability
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Set x-axis ticks to be daily
    max_hours = summary_df['lead_time_hours'].max()
    ax.set_xticks(np.arange(0, max_hours + 1, 24))
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot weather forecast evaluation results with confidence intervals."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV file containing MSE results."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=f"./plots/finetunebasemse.png",
        help="Path to save the output plot image."
    )
    parser.add_argument(
        "--k_factor",
        type=float,
        default=2.0,
        help="Autocorrelation inflation factor 'k'. Use 1.0 for no correction."
    )
    
    args = parser.parse_args()
    
    plot_mse_with_confidence_intervals(args.csv_path, args.output_path, args.k_factor)