import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_regional_two_figs(
    csv_path,
    output_dir="plots",
    ylim_errors=None,
    ylim_precip=None,
):
    df = pd.read_csv(csv_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ensure proper dtypes/sorting
    if "lead_time" in df.columns:
        df["lead_time"] = pd.to_numeric(df["lead_time"], errors="coerce")
    df = df.dropna(subset=["lead_time"])
    df = df.sort_values(["region", "lead_time", "model"])

    # Assign a unique color per model for consistency
    models = df["model"].unique()
    colors = plt.cm.tab10.colors
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    for region, region_df in df.groupby("region"):
        # =========================
        # Figure 1: Errors per model
        # =========================
        fig_err, ax_err = plt.subplots(figsize=(9, 5))

        for model, model_df in region_df.groupby("model"):
            rmse_stats = (
                model_df.groupby("lead_time", as_index=False)
                .agg(rmse_mean=("rmse","mean"),
                     rmse_std=("rmse","std"),
                     rmse_count=("rmse","count"))
            )
            rmse_stats["stderr"] = rmse_stats["rmse_std"] / np.sqrt(rmse_stats["rmse_count"].clip(lower=1))

            ax_err.plot(
                rmse_stats["lead_time"],
                rmse_stats["rmse_mean"],
                color=model_colors[model],
                linestyle="-",
                marker="o",
                label=f"{model} RMSE",
            )
            lower = rmse_stats["rmse_mean"] - rmse_stats["stderr"]
            upper = rmse_stats["rmse_mean"] + rmse_stats["stderr"]
            if np.isfinite(lower).any() and np.isfinite(upper).any():
                ax_err.fill_between(
                    rmse_stats["lead_time"], lower, upper,
                    color=model_colors[model], alpha=0.2, linewidth=0,
                )

        ax_err.set_title(f"RMSE vs Lead Time — {region}")
        ax_err.set_xlabel("Lead Time (hours)")
        ax_err.set_ylabel("RMSE")
        if ylim_errors is not None:
            ymin, ymax = None, None
            ax_err.set_ylim(bottom=ymin if ymin is not None else ax_err.get_ylim()[0]*0.5,
                     top=ymax if ymax is not None else ax_err.get_ylim()[1]*1.5)
        ax_err.grid(True, alpha=0.3, linestyle="--")
        ax_err.legend(loc="upper right")
        fig_err.tight_layout()
        fig_err.savefig(Path(output_dir) / f"errors_{region}.png", dpi=150)
        plt.close(fig_err)

        # ===============================================
        # Figure 2: Precip — target + predicted per model
        # ===============================================
        fig_p, ax_p = plt.subplots(figsize=(9, 5))

        # Target precip: drop duplicates
        target_stats = (
            region_df[["lead_time", "target_mean"]]
            .drop_duplicates()
            .groupby("lead_time", as_index=False)["target_mean"]
            .mean()
        )
        ax_p.plot(
            target_stats["lead_time"],
            target_stats["target_mean"],
            color="black",
            linestyle="--",
            marker="s",
            label="Target precip",
        )

        # Model predicted precip
        for model, model_df in region_df.groupby("model"):
            pred_stats = (
                model_df.groupby("lead_time", as_index=False)
                .agg(pred_mean=("pred_mean","mean"),
                     pred_std=("pred_mean","std"),
                     pred_count=("pred_mean","count"))
            )
            pred_stats["stderr"] = pred_stats["pred_std"] / np.sqrt(pred_stats["pred_count"].clip(lower=1))

            ax_p.plot(
                pred_stats["lead_time"],
                pred_stats["pred_mean"],
                color=model_colors[model],
                linestyle=":",
                marker="o",
                label=f"Pred precip ({model})",
            )
            lower = pred_stats["pred_mean"] - pred_stats["stderr"]
            upper = pred_stats["pred_mean"] + pred_stats["stderr"]
            if np.isfinite(lower).any() and np.isfinite(upper).any():
                ax_p.fill_between(
                    pred_stats["lead_time"], lower, upper,
                    color=model_colors[model], alpha=0.15, linewidth=0,
                )

        ax_p.set_title(f"Precip vs Lead Time — {region}")
        ax_p.set_xlabel("Lead Time (hours)")
        ax_p.set_ylabel("Average Precipitation (mm)")
        if ylim_precip is not None:
            ymin, ymax = None, None
            ax_p.set_ylim(bottom=ymin if ymin is not None else ax_p.get_ylim()[0]*0.5,
                     top=ymax if ymax is not None else ax_p.get_ylim()[1]*1.5)
        ax_p.grid(True, alpha=0.3, linestyle="--")
        ax_p.legend(loc="upper right")
        fig_p.tight_layout()
        fig_p.savefig(Path(output_dir) / f"precip_{region}.png", dpi=150)
        plt.close(fig_p)

    print(f"Plots saved to {output_dir}/")


plot_regional_two_figs(
    "rmse_regions_ALL_MODELS.csv",
    output_dir="plots/dual",
    ylim_errors=(1,2),      # y-axis limits for RMSE
    ylim_precip=(1,2)       # y-axis limits for precip
)
