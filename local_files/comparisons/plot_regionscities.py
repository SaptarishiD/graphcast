import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_regional_errors(values_csv, output_dir="plots"):
    """
    Plots error (pred - truth) with CI and average precipitation per region.
    
    Parameters
    ----------
    values_csv : str
        Path to CSV containing init_date, lead_time, model, region, truth, pred
    output_dir : str
        Directory to save plots
    """
    df = pd.read_csv(values_csv)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for region, region_df in df.groupby("region"):
        records = []

        # Group by lead time across inits/models
        for lead_time, lead_df in region_df.groupby("lead_time"):
            # error = pred - truth (already means)
            errors = lead_df["pred_mean"] - lead_df["target_mean"]
            mean_error = errors.mean()
            stderr_error = errors.std(ddof=1) / np.sqrt(len(errors)) if len(errors) > 1 else 0.0

            # average precipitation (truth)
            mean_precip = lead_df["target_mean"].mean()

            records.append({
                "lead_time": lead_time,
                "mean_error": mean_error,
                "stderr_error": stderr_error,
                "mean_precip": mean_precip
            })

        stats_df = pd.DataFrame(records).sort_values("lead_time")

        # --- Plot ---
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Error with CI
        ax1.plot(stats_df["lead_time"], stats_df["mean_error"], color="tab:red", label="Error (Pred - Truth)")
        ax1.fill_between(
            stats_df["lead_time"],
            stats_df["mean_error"] - stats_df["stderr_error"],
            stats_df["mean_error"] + stats_df["stderr_error"],
            color="tab:red", alpha=0.2
        )
        ax1.set_xlabel("Lead Time (hours)")
        ax1.set_ylabel("Error (Pred - Truth)", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        # Avg precipitation on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(stats_df["lead_time"], stats_df["mean_precip"], color="tab:blue", label="Mean Precip")
        ax2.set_ylabel("Average Precipitation (mm)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        # Titles and save
        plt.title(f"Region: {region}")
        fig.tight_layout()
        plt.savefig(Path(output_dir) / f"plot_{region}.png", dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}/")



plot_regional_errors("rmse_regions_ALL_MODELS.csv", output_dir="plots")

