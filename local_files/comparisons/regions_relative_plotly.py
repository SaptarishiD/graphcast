import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import argparse
import os
from scipy import stats


def parse_horizon_to_hours(horizon_str):
    """Converts a pandas timedelta string like 'X days HH:MM:SS' to total hours."""
    try:
        return pd.to_timedelta(horizon_str).total_seconds() / 3600
    except (ValueError, TypeError):
        return np.nan

def plot_mse_with_confidence_intervals(csv_path: str, output_path: str, k_factor: float = 2.0, baseline_model: str = "graphcast_base", models_to_plot: list[str] = None, model_rename: dict[str, str] = None):

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
        df.groupby(group_cols)['acc']
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

    # 4. Create subplots with Plotly
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=('Forecast Skill (ACC) vs. Lead Time with Corrected 95% CIs', 
                       f'Improvement vs {baseline_model} (%)')
    )

    # Define color palette
    if has_region:
        label_series = summary_df[['model', 'region']].drop_duplicates().apply(lambda r: f"{r['model']} ({r['region']})", axis=1)
    else:
        label_series = summary_df['model'].drop_duplicates()

    colors = px.colors.qualitative.Set1[:len(label_series)]
    color_map = dict(zip(label_series, colors))

    # ----- TOP: MSE + CI -----
    group_cols_for_plot = ['model', 'region'] if has_region else ['model']
    for group_keys, group_data in summary_df.groupby(group_cols_for_plot):
        model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
        region = None if isinstance(group_keys, str) else group_keys[1]

        # apply renaming if provided
        display_name = model_rename.get(model_name, model_name) if model_rename else model_name
        label = display_name if region is None else f"{display_name} ({region})"

        if model_name == 'HRES':
            continue

        color = color_map.get(label, colors[0])
        group_data = group_data.sort_values('lead_time_hours')

        # Add confidence interval as filled area
        fig.add_trace(
            go.Scatter(
                x=group_data['lead_time_hours'].tolist() + group_data['lead_time_hours'].tolist()[::-1],
                y=(group_data['mse_mean'] + group_data['ci_half_width_corrected']).tolist() + 
                  (group_data['mse_mean'] - group_data['ci_half_width_corrected']).tolist()[::-1],
                fill='toself',
                fillcolor=color,
                opacity=0.3,
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                name=f'{label} CI'
            ),
            row=1, col=1
        )

        # Add main line
        fig.add_trace(
            go.Scatter(
                x=group_data['lead_time_hours'],
                y=group_data['mse_mean'],
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(color=color, size=8),
                name=label,
                hovertemplate=f'<b>{label}</b><br>' +
                             'Lead Time: %{x} hours<br>' +
                             'ACC: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

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
            100.0 * (joined['mse_base'] - joined['mse_mean']) / joined['mse_base'],
            np.nan
        )

        for group_keys, group_data in joined.groupby(group_cols_for_plot):
            model_name = group_keys if isinstance(group_keys, str) else group_keys[0]
            region = None if isinstance(group_keys, str) else group_keys[1]

            display_name = model_rename.get(model_name, model_name) if model_rename else model_name
            label = display_name if region is None else f"{display_name} ({region})"

            if model_name in [baseline_model, 'HRES']:
                continue

            color = color_map.get(label, colors[0])
            gd = group_data.sort_values('lead_time_hours').dropna(subset=['pct_improvement'])

            if gd.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=gd['lead_time_hours'],
                    y=gd['pct_improvement'],
                    mode='lines+markers',
                    line=dict(color=color, width=3),
                    marker=dict(color=color, size=8),
                    name=label,
                    showlegend=False,  # Already shown in top plot
                    hovertemplate=f'<b>{label}</b><br>' +
                                 'Lead Time: %{x} hours<br>' +
                                 'Improvement: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )

        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7, row=2, col=1)

    else:
        # Add text if baseline not found
        fig.add_annotation(
            text=f"Baseline '{baseline_model}' not found.<br>Skipping improvement panel.",
            xref="x2", yref="y2",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=14),
            row=2, col=1
        )

    # Update layout
    max_hours = summary_df['lead_time_hours'].max()
    x_ticks = list(range(0, int(max_hours) + 1, 24))

    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        hovermode='x unified',
        template='plotly_white'
    )

    # Update x-axes
    fig.update_xaxes(
        title_text="Lead Time (hours)",
        tickmode='array',
        tickvals=x_ticks,
        row=2, col=1
    )

    # Update y-axes
    fig.update_yaxes(
        title_text="Average ACC",
        tickformat='.2e',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text=f"Improvement vs {baseline_model} (%)",
        row=2, col=1
    )

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as HTML for interactivity
    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path)
    print(f"Interactive plot saved to {html_path}")
    
    # Also save as PNG if requested
    if output_path.endswith('.png'):
        try:
            fig.write_image(output_path, width=1400, height=800, scale=2)
            print(f"Static plot saved to {output_path}")
        except Exception as e:
            print(f"Could not save PNG (requires kaleido): {e}")
            print("Install with: pip install kaleido")
    
    # Show the plot
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot weather forecast evaluation results with confidence intervals and % improvement vs baseline using Plotly."
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

    args = parser.parse_args()

    models_to_plot = args.models_to_plot.split(",") if args.models_to_plot else None
    model_rename = dict(item.split(":") for item in args.model_rename.split(",")) if args.model_rename else None

    plot_mse_with_confidence_intervals(
        args.csv_path,
        args.output_path,
        args.k_factor,
        baseline_model=args.baseline_model,
        models_to_plot=models_to_plot,
        model_rename=model_rename
    )

"""
python regions_relative_plotly.py \
    --csv_path skill_score_India_2025-08-1311-32-55_ACC.csv \
    --output_path plots/acc_plot_choice.png \
    --models_to_plot "Graphcast_Base,Graphcast_Finetuned1" \
    --model_rename "Graphcast_Base:Base,Graphcast_Finetuned1:Finetuned" \
    --baseline_model Graphcast_Base
"""