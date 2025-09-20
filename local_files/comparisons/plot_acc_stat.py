import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# --- small autocorr + inflation utilities (work in z-space for ACC) ---
def _autocorr(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < lag + 1:
        return np.nan
    x = x - x.mean()
    denom = np.dot(x, x)
    numer = np.dot(x[:-lag], x[lag:])
    return numer / denom if denom > 0 else np.nan

def inflation_factor_k(series, max_lag=2):
    """
    HAC/Bartlett inflation for serial correlation:
      k = sqrt( 1 + 2 * sum_{h=1..L} (1 - h/n) * rho(h) )
    """
    y = pd.Series(series, dtype=float).dropna().values
    n = len(y)
    if n <= 1:
        return 1.0
    rhos = []
    for h in range(1, min(max_lag, n-1) + 1):
        r = _autocorr(y, h)
        if not np.isfinite(r):
            r = 0.0
        rhos.append((1 - h / n) * r)
    k2 = 1.0 + 2.0 * np.sum(rhos)
    return float(np.sqrt(max(k2, 1.0)))  # never deflate

def summarize_acc_by_model(df,
                           lead_col='forecast_horizon_hours',
                           init_col='init_date',
                           model_col='model',
                           acc_col='acc',
                           alpha=0.20,        # 80% CI → tighter bands
                           max_lag=2,         # AR(2)-style inflation
                           center='z'         # 'z' uses tanh(mean(z)) as the point estimate
                           ):
    """
    Returns a tidy summary: one row per (model, lead) with ACC mean and CI.
    CI is computed in Fisher z-space with HAC inflation, then back-transformed.
    """
    rows = []
    zcrit = stats.norm.ppf(1 - alpha/2.0)
    for (model, lead), g in df.groupby([model_col, lead_col]):
        g = g[[init_col, acc_col]].dropna().sort_values(init_col)
        acc = pd.to_numeric(g[acc_col], errors='coerce').dropna().values
        n = len(acc)
        if n < 2:
            rows.append({
                'model': model, 'lead': lead, 'n': n,
                'acc_hat': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                'k': 1.0
            })
            continue
        acc = np.clip(acc, -0.999999, 0.999999)
        z = np.arctanh(acc)
        z_mean = float(np.mean(z))
        s_z = float(np.std(z, ddof=1)) if n > 1 else 0.0
        # inflate SE in z-space to respect temporal correlation across inits
        k = inflation_factor_k(z - z_mean, max_lag=max_lag) if n > 1 else 1.0
        se_z = k * s_z / np.sqrt(n) if n > 1 else np.nan

        # CI in z-space → back-transform
        lo_r = np.tanh(z_mean - zcrit * se_z) if np.isfinite(se_z) else np.nan
        hi_r = np.tanh(z_mean + zcrit * se_z) if np.isfinite(se_z) else np.nan

        # point estimate: tanh(mean(z)) (less biased than arithmetic mean of r)
        acc_hat = np.tanh(z_mean) if center == 'z' else float(np.mean(acc))

        rows.append({
            'model': model, 'lead': lead, 'n': n,
            'acc_hat': acc_hat, 'ci_lo': lo_r, 'ci_hi': hi_r,
            'k': k
        })
    return pd.DataFrame(rows).sort_values(['model','lead']).reset_index(drop=True)

def plot_acc_levels(summary_df,
                    baseline_model,
                    models_to_plot=None,        # NEW: choose which models to plot
                    custom_colors=None,         # NEW: control line colors
                    title="ACC by model (with Fisher-z HAC CIs)",
                    ymin=None, ymax=1.0,
                    savepath=None,
                    models_rename=None):
    """
    Plot ACC for the baseline and other models, plus % improvement subplot.
    """

    if summary_df.empty:
        raise ValueError("summary_df is empty.")

    # Restrict models if specified
    all_models = list(summary_df['model'].unique())
    if models_to_plot is None:
        models = all_models
    else:
        models = [m for m in models_to_plot if m in all_models]

    # Ensure baseline is included and listed first
    if baseline_model not in models:
        raise ValueError(f"Baseline model '{baseline_model}' not found in summary_df.")
    models = [baseline_model] + [m for m in models if m != baseline_model]

    leads = np.sort(summary_df['lead'].unique())

    # Colors: use custom if given, else fallback to matplotlib cycle
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    color_map = {}
    for i, m in enumerate(models):
        if custom_colors and m in custom_colors:
            color_map[m] = custom_colors[m]
        else:
            color_map[m] = 'black' if m == baseline_model else cycle[(i-1) % len(cycle)]

    # --- Two subplots: top = ACC, bottom = % improvement ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # ============ ACC plot ============
    for m in models:
        g = summary_df[summary_df['model'] == m].sort_values('lead')
        if g.empty:
            continue
        x = g['lead'].values
        y = g['acc_hat'].values
        lo = g['ci_lo'].values
        hi = g['ci_hi'].values

        lw = 3.0 if m == baseline_model else 2.0
        z = 5 if m == baseline_model else 4
        ax1.plot(x, y, linestyle='-', linewidth=lw, color=color_map[m], label=models_rename[m], zorder=z)  # no markers
        ax1.fill_between(x, lo, hi, color=color_map[m], alpha=(0.12 if m == baseline_model else 0.16), linewidth=0)

    ax1.set_title(title, fontsize=20, pad=10)
    ax1.set_ylabel('ACC (Higher is Better)', fontsize=18)
    ax1.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.4)
    # ax1.grid(False)
    ax1.legend(title="", ncols=2, fontsize=21)


    # y-limits
    if ymin is not None or ymax is not None:
        ax1.set_ylim(bottom=ymin if ymin is not None else ax1.get_ylim()[0],
                     top=ymax if ymax is not None else ax1.get_ylim()[1])

    # ============ % Improvement plot ============
    baseline_vals = summary_df[summary_df['model'] == baseline_model].sort_values('lead')
    for m in models:
        if m == baseline_model:
            continue
        g = summary_df[summary_df['model'] == m].sort_values('lead')
        if g.empty:
            continue
        x = g['lead'].values
        # Align with baseline
        merged = pd.merge(baseline_vals[['lead','acc_hat']], g[['lead','acc_hat']], on='lead', suffixes=('_base','_finetune'))
        improvement = 100 * (merged['acc_hat_finetune'] - merged['acc_hat_base']) / np.abs(merged['acc_hat_base'])
        ax2.plot(merged['lead'], improvement, linestyle='-', linewidth=2.0, color=color_map[m])

    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_xlabel('Lead time (hours)', fontsize=18)
    ax2.set_ylabel('% Improvement', fontsize=18)
    ax2.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.4)
    # ax2.grid(False)
    ax2.legend(fontsize=18)

    ymax = 70
    ax2.set_ylim(top=ymax if ymax is not None else ax1.get_ylim()[1])


    ax1.tick_params(axis='both', labelsize=15)
    ax2.tick_params(axis='both', labelsize=15)


    # x ticks
    step = 24 if (leads.max() - leads.min() >= 96) else max(6, int(np.median(np.diff(leads))) if len(leads) > 1 else 6)

        # Force ticks every 24 hours on both subplots
    ax1.set_xticks(np.arange(leads.min()-6, leads.max() + 1, 24))
    ax2.set_xticks(np.arange(leads.min()-6, leads.max() + 1, 24))



    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    # plt.show()





csv_path = "\
skill_score_India_2025-08-1311-32-55_ACC.csv\
"


df = pd.read_csv(csv_path)  # must include init_date, forecast_horizon_hours, model, and mse or acc



acc_summary = summarize_acc_by_model(
    df,
    lead_col="forecast_horizon_hours",
    init_col="init_date",
    model_col="model",
    acc_col="acc",
    alpha=0.20,   # 80% CI (tighter); use 0.05 for 95%
    max_lag=2,    # AR(2)-style inflation across init_date sequence
    center='z'    # plot tanh(mean(z)); change to 'r' for arithmetic mean on ACC
)


custom_colors = {
    "Graphcast_Base": "orange",
    "Graphcast_Finetuned": "blue",
    "Other_Model": "blue"
}


savepath = 'plots/acc_levels.png'

model_rename = {
    "Graphcast_Base": "Base",
    "Graphcast_Finetuned1": "Finetuned"
}
plot_acc_levels(
    acc_summary,
    baseline_model="Graphcast_Base",
    models_to_plot=["Graphcast_Base", "Graphcast_Finetuned1"],  # choose subset
    custom_colors=custom_colors,  # control colors
    title="ACC Comparison and % Improvement",
    ymin=0.0,
    ymax=1.0,
    savepath="plots/acc_levels.png",
    models_rename = model_rename
)

print(savepath)

