import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_regional_multi_model(csv_path, output_dir="plots"):
    df = pd.read_csv(csv_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Assign a unique color per model for consistency
    models = df["model"].unique()
    colors = plt.cm.tab10.colors  # up to 10 distinct colors
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    for region, region_df in df.groupby("region"):
        fig, ax1 = plt.subplots(figsize=(9, 5))

        region_records = []

        # --- RMSE per model ---
        for model, model_df in region_df.groupby("model"):
            rmse_stats = (
                model_df.groupby("lead_time")["rmse"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            rmse_stats["stderr"] = rmse_stats["std"] / np.sqrt(rmse_stats["count"])

            ax1.plot(
                rmse_stats["lead_time"],
                rmse_stats["mean"],
                color=model_colors[model],
                linestyle="-",
                label=f"RMSE ({model})",
            )
            ax1.fill_between(
                rmse_stats["lead_time"],
                rmse_stats["mean"] - rmse_stats["stderr"],
                rmse_stats["mean"] + rmse_stats["stderr"],
                color=model_colors[model],
                alpha=0.2,
            )

        ax1.set_xlabel("Lead Time (hours)")
        ax1.set_ylabel("RMSE", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        # --- Precipitation curves ---
        ax2 = ax1.twinx()

        # Target precip (truth)
        target_stats = (
            region_df.groupby("lead_time")["target_mean"]
            .mean()
            .reset_index()*2
        )
        ax2.plot(
            target_stats["lead_time"],
            target_stats["target_mean"],
            color="black",
            linestyle="--",
            label="Target Precip",
        )

        # Model precip predictions
        for model, model_df in region_df.groupby("model"):
            pred_stats = (
                model_df.groupby("lead_time")["pred_mean"]
                .mean()
                .reset_index()
            )
            ax2.plot(
                pred_stats["lead_time"],
                pred_stats["pred_mean"],
                color=model_colors[model],
                linestyle=":",
                label=f"Pred Precip ({model})",
            )

        ax2.set_ylabel("Average Precipitation (mm)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        # --- Legend ---
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title(f"Region: {region}")
        fig.tight_layout()

        plt.savefig(Path(output_dir) / f"regionscitiesplot_{region}.png", dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}/")


plot_regional_multi_model("rmse_regions_ALL_MODELS.csv", output_dir="plots")
