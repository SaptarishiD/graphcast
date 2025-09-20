import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def _distinct_colors(n):
    # simple palette
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if n <= len(base):
        return base[:n]
    # repeat if needed
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def plot_mse_rmse_verification(out_df: pd.DataFrame,
                               alpha: float = 0.05,
                               title: str = "Verification: MSE/RMSE improvement vs. baseline",
                               savepath: str | None = None):
    """
    Expects rows from verify_against_baseline(..., metric in {'mse','rmse'})
    Columns used: lead, model, mean_improvement, ci_lo, ci_hi, p_adj or p_raw, pct_improvement
    """
    if out_df.empty:
        raise ValueError("Empty results DataFrame.")
    # Accept either 'mse' or 'rmse'
    sub = out_df[out_df['metric'].str.upper().isin(['MSE','RMSE'])].copy()
    if sub.empty:
        raise ValueError("No MSE/RMSE rows found.")

    # Use adjusted p if available; else raw
    pcol = 'p_adj' if 'p_adj' in sub.columns else 'p_raw'
    sub['sig'] = (sub[pcol] < alpha)

    models = list(sub['model'].unique())
    colors = {m: c for m, c in zip(models, _distinct_colors(len(models)))}

    # layout
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.25)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # TOP: absolute improvement (baseline - model)
    for m in models:
        g = sub[sub['model'] == m].sort_values('lead')
        if g.empty:
            continue
        x = g['lead'].values
        y = g['mean_improvement'].values
        lo = g['ci_lo'].values
        hi = g['ci_hi'].values
        ax_top.plot(x, y, marker='o', linestyle='-', linewidth=2, color=colors[m], label=m)
        # CI ribbon
        ax_top.fill_between(x, lo, hi, color=colors[m], alpha=0.15, linewidth=0)
        # significance markers
        # filled markers where significant, hollow where not
        sigmask = g['sig'].values
        ax_top.scatter(x[sigmask], y[sigmask], s=40, color=colors[m], edgecolors='black', linewidths=0.6, zorder=3)
        ax_top.scatter(x[~sigmask], y[~sigmask], s=40, facecolors='none', edgecolors=colors[m], linewidths=1.2, zorder=3)

    ax_top.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax_top.set_ylabel('Absolute improvement\n(baseline − model)', fontsize=12)
    ax_top.set_title(title, fontsize=15, pad=12)
    ax_top.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax_top.legend(title="Model", ncols=2, fontsize=10)

    # BOTTOM: percent improvement (only for MSE/RMSE)
    for m in models:
        g = sub[sub['model'] == m].sort_values('lead')
        if g.empty or 'pct_improvement' not in g.columns:
            continue
        x = g['lead'].values
        y = g['pct_improvement'].values
        ax_bot.plot(x, y, marker='o', linestyle='-', linewidth=2, color=colors[m], label=m)
    ax_bot.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax_bot.set_xlabel('Lead time (hours)', fontsize=12)
    ax_bot.set_ylabel('% improvement vs. baseline', fontsize=12)
    ax_bot.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # tidy x ticks
    all_leads = np.sort(sub['lead'].unique())
    step = 24 if all_leads.max() - all_leads.min() >= 96 else max(6, int(np.median(np.diff(all_leads))))
    ax_bot.set_xticks(np.arange(all_leads.min(), all_leads.max()+1, step))

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_acc_verification(out_df: pd.DataFrame,
                          alpha: float = 0.05,
                          title: str = "Verification: ACC difference vs. baseline",
                          show_single_series_ci: bool = False,
                          savepath: str | None = None):
    """
    Expects rows from verify_against_baseline(..., metric='acc')
    Columns used: lead, model, mean_diff, ci_lo, ci_hi, p_adj or p_raw
    If show_single_series_ci=True, also displays Fisher-z CIs for the model and baseline means (shaded).
    """
    if out_df.empty:
        raise ValueError("Empty results DataFrame.")
    sub = out_df[out_df['metric'].str.upper() == 'ACC'].copy()
    if sub.empty:
        raise ValueError("No ACC rows found.")

    pcol = 'p_adj' if 'p_adj' in sub.columns else 'p_raw'
    sub['sig'] = (sub[pcol] < alpha)

    models = list(sub['model'].unique())
    colors = {m: c for m, c in zip(models, _distinct_colors(len(models)))}

    fig, ax = plt.subplots(figsize=(14, 6))

    for m in models:
        g = sub[sub['model'] == m].sort_values('lead')
        if g.empty:
            continue
        x = g['lead'].values
        y = g['mean_diff'].values  # model - baseline; >0 is better
        lo = g['ci_lo'].values
        hi = g['ci_hi'].values
        ax.plot(x, y, marker='o', linestyle='-', linewidth=2, color=colors[m], label=m)
        ax.fill_between(x, lo, hi, color=colors[m], alpha=0.15, linewidth=0)
        # significance markers
        sigmask = g['sig'].values
        ax.scatter(x[sigmask], y[sigmask], s=40, color=colors[m], edgecolors='black', linewidths=0.6, zorder=3)
        ax.scatter(x[~sigmask], y[~sigmask], s=40, facecolors='none', edgecolors=colors[m], linewidths=1.2, zorder=3)

        if show_single_series_ci and {'acc_ci_model_lo','acc_ci_model_hi'}.issubset(g.columns):
            # Optional: overlay model and baseline Fisher-z CIs as transparent ribbons around 0 line (for context)
            # Here we "center" these around zero purely for display context.
            mlo, mhi = g['acc_ci_model_lo'].values, g['acc_ci_model_hi'].values
            blo, bhi = g['acc_ci_base_lo'].values,  g['acc_ci_base_hi'].values
            # Show the *span* of model and baseline CIs (very light)
            ax.fill_between(x, mlo - g['mean_base'].values, mhi - g['mean_base'].values,
                            color=colors[m], alpha=0.05, linewidth=0)
            ax.fill_between(x, blo - g['mean_base'].values, bhi - g['mean_base'].values,
                            color='gray', alpha=0.05, linewidth=0)

    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel('Lead time (hours)', fontsize=12)
    ax.set_ylabel('ACC difference (model − baseline)', fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(title="Model", ncols=2, fontsize=10)

    all_leads = np.sort(sub['lead'].unique())
    step = 24 if all_leads.max() - all_leads.min() >= 96 else max(6, int(np.median(np.diff(all_leads))))
    ax.set_xticks(np.arange(all_leads.min(), all_leads.max()+1, step))

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    # plt.show()


# ---------- helpers for autocorrelation-corrected SE ----------
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

def inflation_factor_k(diffs, max_lag=2):
    """
    Bartlett/HAC-style inflation using sample autocorrelations up to max_lag.
    k = sqrt(1 + 2 * sum_{h=1..L} (1 - h/n) * rho(h))
    """
    y = pd.Series(diffs, dtype=float).dropna().values
    n = len(y)
    if n <= 1:
        return 1.0
    rhos = []
    for h in range(1, min(max_lag, n-1) + 1):
        r = _autocorr(y, h)
        if np.isnan(r): 
            r = 0.0
        rhos.append((1 - h / n) * r)
    k2 = 1.0 + 2.0 * np.sum(rhos)
    return float(np.sqrt(max(k2, 1.0)))  # never deflate

def paired_t_with_k(diffs, alpha=0.05, max_lag=2):
    diffs = pd.Series(diffs, dtype=float).dropna().values
    n = len(diffs)
    mean = float(np.mean(diffs)) if n else np.nan
    s = float(np.std(diffs, ddof=1)) if n > 1 else np.nan
    if n <= 1 or not np.isfinite(s) or s == 0:
        return dict(n=n, neff=np.nan, mean=mean, se=np.nan, k=1.0, 
                    t=np.nan, p=np.nan, ci=(np.nan, np.nan))
    k = inflation_factor_k(diffs, max_lag=max_lag)
    se = k * s / np.sqrt(n)
    df = max(n - 1, 1)
    t_stat = mean / se
    # two-sided p using Student t on df=n-1, as in GraphCast (with inflated SE)
    p = 2 * stats.t.sf(abs(t_stat), df)
    tcrit = stats.t.ppf(1 - alpha/2, df)
    ci = (mean - tcrit * se, mean + tcrit * se)
    return dict(n=n, neff=n/(k**2), mean=mean, se=se, k=k, t=t_stat, p=p, ci=ci)

# ---------- ACC single-sample Fisher z CI (display only) ----------
def fisher_acc_ci(acc_values, alpha=0.05):
    x = pd.Series(acc_values, dtype=float).dropna().values
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    # clip to avoid infs
    x = np.clip(x, -0.999999, 0.999999)
    z = np.arctanh(x)
    zbar = np.mean(z)
    se = 1 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha/2)
    lo, hi = zbar - zcrit*se, zbar + zcrit*se
    return (np.tanh(lo), np.tanh(hi))

# ---------- RMSE skill score CI via union bound (GraphCast Eq. 33) ----------
def ratio_ci_from_components(diff_ci, denom_ci):
    """
    diff_ci: CI for (model - baseline) or (baseline - model), be consistent with your skill definition.
    denom_ci: CI for baseline metric (must have lHRES > 0).
    Returns conservative CI for (diff / denom).
    """
    ldiff, udiff = diff_ci
    lden, uden = denom_ci
    if lden <= 0 or any(map(lambda v: not np.isfinite(v), [ldiff, udiff, lden, uden])):
        return (np.nan, np.nan)
    candidates = [ldiff/uden, ldiff/lden, udiff/uden, udiff/lden]
    return (min(candidates), max(candidates))

# ---------- main application ----------
def verify_against_baseline(df, baseline_model, metric='mse', lead_col='forecast_horizon',
                            init_col='init_date', model_col='model',
                            max_lag=2, alpha=0.05, adjust='sidak'):
    """
    df: long dataframe with columns [init_date, forecast_horizon, model, mse/acc/rmse]
    metric: 'mse', 'rmse', or 'acc'
    adjust: 'sidak', 'bh' (FDR), or None
    """
    data = df.copy()
    # If only MSE is available but you want RMSE-style comparisons, you can sqrt-transform:
    if metric.lower() == 'rmse' and 'rmse' not in data.columns and 'mse' in data.columns:
        data['rmse'] = np.sqrt(data['mse'])
    score_col = {'mse':'mse', 'rmse':'rmse', 'acc':'acc'}[metric.lower()]
    if score_col not in data.columns:
        raise ValueError(f"Expected column '{score_col}'")
    # prepare paired table for each (lead, model!=baseline)
    results = []
    leads = sorted(pd.unique(data[lead_col]))
    models = [m for m in pd.unique(data[model_col]) if m != baseline_model]
    for lead in leads:
        base = data[(data[model_col]==baseline_model) & (data[lead_col]==lead)]
        for model in models:
            mod = data[(data[model_col]==model) & (data[lead_col]==lead)]
            # inner join on initialization to ensure pairing
            merged = pd.merge(mod[[init_col, score_col]].rename(columns={score_col:'score_model'}),
                              base[[init_col, score_col]].rename(columns={score_col:'score_base'}),
                              on=init_col, how='inner').sort_values(init_col)
            if merged.empty:
                continue
            if metric.lower() in ('mse','rmse'):
                # lower is better → improvement = base - model
                diffs = (merged['score_base'] - merged['score_model']).values
                # paired test on diffs (improvement); positive mean = improvement
                test = paired_t_with_k(diffs, alpha=alpha, max_lag=max_lag)
                # baseline single-sample CI (approximate; not for significance)
                base_test = paired_t_with_k(merged['score_base'].values - merged['score_base'].values.mean(),
                                            alpha=alpha, max_lag=max_lag)
                # To get baseline CI itself, reuse s & k but around the mean
                n_b = merged['score_base'].notna().sum()
                s_b = merged['score_base'].std(ddof=1)
                k_b = inflation_factor_k(merged['score_base'].values - merged['score_base'].values.mean(),
                                         max_lag=max_lag) if n_b>1 else 1.0
                se_b = k_b * s_b / np.sqrt(max(n_b,1))
                df_b = max(n_b-1,1)
                tcrit_b = stats.t.ppf(1 - alpha/2, df_b)
                base_ci = (merged['score_base'].mean() - tcrit_b*se_b,
                           merged['score_base'].mean() + tcrit_b*se_b)
                # RMSE/MSE skill score (improvement ratio) CI (conservative)
                skill_ci = ratio_ci_from_components(test['ci'], base_ci)
                # percent improvement
                pct_imp = 100 * test['mean'] / merged['score_base'].mean()
                results.append(dict(
                    metric=metric.upper(), lead=lead, model=model,
                    n=test['n'], neff=test['neff'], k=test['k'],
                    mean_base=float(merged['score_base'].mean()),
                    mean_model=float(merged['score_model'].mean()),
                    mean_improvement=float(test['mean']),
                    ci_lo=float(test['ci'][0]), ci_hi=float(test['ci'][1]),
                    p_raw=float(test['p']),
                    pct_improvement=float(pct_imp),
                    skill_ci_lo=float(skill_ci[0]*100), skill_ci_hi=float(skill_ci[1]*100),
                ))
            else:  # ACC: higher is better
                diffs = (merged['score_model'] - merged['score_base']).values
                test = paired_t_with_k(diffs, alpha=alpha, max_lag=max_lag)
                # single-sample Fisher-z CIs (display only)
                acc_ci_model = fisher_acc_ci(merged['score_model'].values, alpha=alpha)
                acc_ci_base  = fisher_acc_ci(merged['score_base'].values,  alpha=alpha)
                results.append(dict(
                    metric='ACC', lead=lead, model=model,
                    n=test['n'], neff=test['neff'], k=test['k'],
                    mean_base=float(merged['score_base'].mean()),
                    mean_model=float(merged['score_model'].mean()),
                    mean_diff=float(test['mean']),
                    ci_lo=float(test['ci'][0]), ci_hi=float(test['ci'][1]),
                    p_raw=float(test['p']),
                    acc_ci_model_lo=float(acc_ci_model[0]), acc_ci_model_hi=float(acc_ci_model[1]),
                    acc_ci_base_lo=float(acc_ci_base[0]),   acc_ci_base_hi=float(acc_ci_base[1]),
                ))
    out = pd.DataFrame(results)
    if out.empty:
        return out
    # multiplicity adjustment per (metric) family across (lead x model)
    if 'p_raw' in out.columns and adjust:
        m = out['p_raw'].notna().sum()
        if adjust == 'sidak':
            out['p_adj'] = 1 - (1 - out['p_raw'])**m
        elif adjust == 'bh':
            # Benjamini-Hochberg FDR
            rank = out['p_raw'].rank(method='first')
            out['p_adj'] = out['p_raw'] * m / rank
            out['p_adj'] = out['p_adj'].clip(upper=1.0)
    return out.sort_values(['metric','lead','model'])



csv_path = "\
skill_score_India_2025-08-2110-39-16_ACC.csv\
"


df = pd.read_csv(csv_path)  # must include init_date, forecast_horizon, model, and mse or acc


out_acc = verify_against_baseline(
    df,
    baseline_model="Graphcast_Base_2014",  # or your chosen baseline
    metric="acc",                           # 'mse', 'rmse', or 'acc'
    lead_col="forecast_horizon",
    init_col="init_date",
    model_col="model",
    max_lag=2,          # AR(2)-style inflation; use 1 if you truly only have daily data
    alpha=0.05,
    adjust="sidak"      # or 'bh' or None
)
my_path = "plots/acc_verification.png"


plot_acc_verification(
    out_acc,
    alpha=0.05,
    title="ACC: difference vs baseline (paired t with AR(2) inflation)",
    show_single_series_ci=False,  # set True if you want the Fisher-z ribbons
    savepath=my_path
)

print(my_path)