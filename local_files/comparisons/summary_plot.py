# plot_from_summary.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _parse_models_to_plot(s: str | None):
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_model_rename(s: str | None):
    if not s:
        return None
    out = {}
    for item in s.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            out[k.strip()] = v.strip()
    return out or None


# --- fuzzy color mapping (subword, case-insensitive) ---
def _get_color(label: str) -> str:
    low = label.lower()
    if "finetuned" in low or "finetuned_3" in low or "finetuned-3" in low:
        return "tab:green"
    if "base" in low:
        return "tab:red"
    return "black"


def _normalize_lead_time_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'lead_time_hours' column exists. If already present, leave as is.
    Otherwise try common fallbacks.
    """
    if "lead_time_hours" in df.columns:
        return df

    # Legacy names or units
    if "forecast_horizon_hours" in df.columns:
        df["lead_time_hours"] = df["forecast_horizon_hours"].astype(float)
        return df

    if "forecast_horizon" in df.columns:
        # try pandas to_timedelta if parsable (e.g., '21600000000000 nanoseconds' may fail)
        try:
            df["lead_time_hours"] = pd.to_timedelta(df["forecast_horizon"]).dt.total_seconds() / 3600.0
            return df
        except Exception:
            pass

    raise RuntimeError(
        "Could not find/construct 'lead_time_hours'. "
        "Expected it in the summary CSV, or provide a convertible horizon column."
    )


def _ensure_ci(summary_df: pd.DataFrame, k_factor: float = 2.0) -> pd.DataFrame:
    """
    If CI columns are present, keep them. Else compute from mse_std and n_samples.
    """
    df = summary_df.copy()

    # prefer existing columns if present
    has_corrected = "ci_half_width_corrected" in df.columns
    has_standard = "ci_half_width_standard" in df.columns

    # If both exist, just use them
    if has_corrected and has_standard:
        return df

    # Otherwise compute what we can
    needed = {"mse_std", "n_samples"}
    if not needed.issubset(df.columns):
        # Maybe original names differ? Try common variants
        rename_map = {}
        if "std" in df.columns and "mse_std" not in df.columns:
            rename_map["std"] = "mse_std"
        if "count" in df.columns and "n_samples" not in df.columns:
            rename_map["count"] = "n_samples"
        if rename_map:
            df = df.rename(columns=rename_map)

    if {"mse_std", "n_samples"}.issubset(df.columns):
        z = 1.96
        # Avoid division by zero
        denom = np.sqrt(df["n_samples"].clip(lower=1))
        df["ci_half_width_standard"] = z * (df["mse_std"].fillna(0) / denom)
        df["ci_half_width_corrected"] = k_factor * df["ci_half_width_standard"]
    else:
        # No way to compute: set zeros
        df["ci_half_width_standard"] = 0.0
        df["ci_half_width_corrected"] = 0.0

    return df


def plot_from_summary(
    summary_csv: str,
    output_path: str,
    precip_csv: str | None = None,
    models_to_plot: list[str] | None = None,
    model_rename: dict[str, str] | None = None,
    baseline_model: str | None = None,
    k_factor: float = 2.0,
):
    # 1) Load summary stats
    print(f"Loading summary stats: {summary_csv}")
    summary_df = pd.read_csv(summary_csv)

    # Basic columns we expect
    # Typical columns saved earlier:
    # ['model','region','lead_time_hours','mse_mean','mse_std','n_samples','ci_half_width_standard','ci_half_width_corrected']
    summary_df = _normalize_lead_time_hours(summary_df)

    # Region may or may not exist
    has_region = "region" in summary_df.columns
    if "model" not in summary_df.columns:
        raise RuntimeError("Summary CSV must include a 'model' column.")

    # Optional filter
    if models_to_plot:
        before = len(summary_df)
        summary_df = summary_df[summary_df["model"].isin(models_to_plot)]
        print(f"Filtered models: {models_to_plot}  ({before} -> {len(summary_df)} rows)")

    # Ensure CI columns
    summary_df = _ensure_ci(summary_df, k_factor=k_factor)

    # Sort by lead time for plotting
    summary_df = summary_df.sort_values("lead_time_hours")

    # 2) Optional precipitation totals
    precip_df = None
    if precip_csv and os.path.exists(precip_csv):
        print(f"Loading precipitation totals: {precip_csv}")
        precip_df = pd.read_csv(precip_csv)
        if "lead_time_hours" not in precip_df.columns:
            raise RuntimeError("Precipitation CSV must include 'lead_time_hours'.")
        if "precipitation" not in precip_df.columns:
            raise RuntimeError("Precipitation CSV must include 'precipitation'.")
        precip_df = precip_df.sort_values("lead_time_hours")

    # 3) Plot
    plt.style.use("seaborn-v0_8-white")
    fig, ax_left = plt.subplots(figsize=(16, 8))

    # Right axis for precipitation
    ax_right = ax_left.twinx() if precip_df is not None else None

    # Compute display names and plot
    group_cols = ["model", "region"] if has_region else ["model"]

    def _group_sort_key(gk):
        model_name = gk if isinstance(gk, str) else gk[0]
        if baseline_model and model_name == baseline_model:
            return (0, model_name)
        return (1, model_name)

    for group_keys, gdf in sorted(summary_df.groupby(group_cols), key=_group_sort_key):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]
        print(f"\nPlotting model: {model_name}, region: {region}")

        display_name = model_rename.get(model_name, model_name) if model_rename else model_name
        print(f"  Display name: {display_name}")
        if 'finetuned' in display_name.lower():
            display_name = 'Finetuned'
        label = display_name if region is None else f"{display_name} ({region})"

        color = _get_color(label)

        ax_left.plot(
            gdf["lead_time_hours"],
            gdf["mse_mean"],
            linestyle="-",
            linewidth=2,
            color=color,
            label=label,
        )
        # CI shading
        lower = gdf["mse_mean"] - gdf["ci_half_width_corrected"]
        upper = gdf["mse_mean"] + gdf["ci_half_width_corrected"]
        ax_left.fill_between(
            gdf["lead_time_hours"],
            lower,
            upper,
            alpha=0.05,
            color=color,
        )

    # Precipitation overlay (if provided)
    if ax_right is not None:
        ax_right.plot(
            precip_df["lead_time_hours"],
            precip_df["precipitation"],
            linestyle="-",
            linewidth=3,
            alpha=0.25,
            color="gray",
            label="Ground Truth Total Precipitation",
        )

    # Labels and ticks
    ax_left.set_title("Forecast Skill (RMSE) vs Lead Time", fontsize=28, pad=20)
    ax_left.set_xlabel("Lead Time (hours)", fontsize=20)
    ax_left.set_ylabel("Average RMSE (lower is better)", fontsize=20)
    ax_left.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Make y-axis "larger" so plots appear smaller
    ymin, ymax = ax_left.get_ylim()
    ax_left.set_ylim(ymin, ymax * 1.5)

    # Remove grid
    ax_left.grid(True, which='both', linestyle='--', alpha=0.1)

    # Tick font sizes
    ax_left.tick_params(axis="both", labelsize=16)
    if ax_right is not None:
        ax_right.set_ylabel("Total Precipitation (m)", fontsize=20)
        ax_right.tick_params(axis="both", labelsize=16)

    # X ticks every 24h
    max_hours = float(summary_df["lead_time_hours"].max()) if not summary_df.empty else 168.0
    xticks = np.arange(0, max_hours + 1, 24.0)
    ax_left.set_xticks(xticks)

    # Combined legend
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    if ax_right is not None:
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        lines = lines_left + lines_right
        labels = labels_left + labels_right
    else:
        lines, labels = lines_left, labels_left

    ax_left.legend(lines, labels, loc="upper left", fontsize=18)

    # Write file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot directly from summary stats CSV.")
    parser.add_argument("--summary_csv", type=str, required=True,
                        help="Path to forecast_summary_stats.csv")
    parser.add_argument("--precip_csv", type=str, default=None,
                        help="Optional: precipitation_totals.csv to overlay on right axis")
    parser.add_argument("--output_path", type=str, default="./plots/forecast_from_summary.png")
    parser.add_argument("--k_factor", type=float, default=2.0,
                        help="Inflation factor for CI if recomputing (temporal autocorr.)")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Model name to prioritize in legend ordering")
    parser.add_argument("--models_to_plot", type=str, default=None,
                        help="Comma-separated model names to include")
    parser.add_argument("--model_rename", type=str, default=None,
                        help="Comma-separated 'old:new' pairs for display names")

    args = parser.parse_args()
    models_to_plot = _parse_models_to_plot(args.models_to_plot)
    model_rename = _parse_model_rename(args.model_rename)

    plot_from_summary(
        summary_csv=args.summary_csv,
        output_path=args.output_path,
        precip_csv=args.precip_csv,
        models_to_plot=models_to_plot,
        model_rename=model_rename,
        baseline_model=args.baseline_model,
        k_factor=args.k_factor,
    )   


if __name__ == "__main__":
    main()
