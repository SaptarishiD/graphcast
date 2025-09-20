from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime, timedelta
from matplotlib.ticker import MultipleLocator, FuncFormatter, ScalarFormatter

def y_ticks_as_ints_with_e_minus3(ax, step=1e-4):
    # Put ticks at 0.001, 0.002, 0.003, ...
    ax.yaxis.set_major_locator(MultipleLocator(step))

    # Show an axis-wide scale factor like "×1e−3" and keep labels as integers
    sf = ScalarFormatter(useMathText=True)
    sf.set_scientific(True)
    sf.set_powerlimits((0, 0))   # always use scientific notation
    sf.set_useOffset(False)      # no additive +offset
    ax.yaxis.set_major_formatter(sf)

    # (Optional) ensure the offset text is visible and nicely sized
    ax.yaxis.get_offset_text().set_size(10)


# --- Autocorr + inflation utilities (adapted for MSE) ---
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

def summarize_mse_by_model(df,
                           lead_col='forecast_horizon_hours',
                           init_col='init_date',
                           model_col='model',
                           mse_col='mse',
                           alpha=0.20,        # 80% CI → tighter bands
                           max_lag=2,         # AR(2)-style inflation
                           log_transform=True  # Use log transform for MSE
                           ):
    """
    Returns a tidy summary: one row per (model, lead) with MSE mean and CI.
    CI is computed in log-space (if log_transform=True) with HAC inflation, then back-transformed.
    """
    rows = []
    zcrit = stats.norm.ppf(1 - alpha/2.0)
    
    for (model, lead), g in df.groupby([model_col, lead_col]):
        g = g[[init_col, mse_col]].dropna().sort_values(init_col)
        mse = pd.to_numeric(g[mse_col], errors='coerce').dropna().values
        n = len(mse)
        
        if n < 2:
            rows.append({
                'model': model, 'lead': lead, 'n': n,
                'mse_hat': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                'rmse_hat': np.nan, 'rmse_ci_lo': np.nan, 'rmse_ci_hi': np.nan,
                'k': 1.0
            })
            continue
        
        # Ensure MSE values are positive for log transform
        mse = np.clip(mse, 1e-10, None)
        
        if log_transform:
            # Work in log space for MSE (similar to Fisher-z for ACC)
            log_mse = np.log(mse)
            log_mean = float(np.mean(log_mse))
            s_log = float(np.std(log_mse, ddof=1)) if n > 1 else 0.0
            
            # Inflate SE in log-space to respect temporal correlation
            k = inflation_factor_k(log_mse - log_mean, max_lag=max_lag) if n > 1 else 1.0
            se_log = k * s_log / np.sqrt(n) if n > 1 else np.nan
            
            # CI in log-space → back-transform
            lo_mse = np.exp(log_mean - zcrit * se_log) if np.isfinite(se_log) else np.nan
            hi_mse = np.exp(log_mean + zcrit * se_log) if np.isfinite(se_log) else np.nan
            
            # Point estimate: exp(mean(log(mse))) (geometric mean)
            mse_hat = np.exp(log_mean)
        else:
            # Work in regular space (arithmetic mean)
            print('Working in regular space')
            mse_mean = float(np.mean(mse))
            s_mse = float(np.std(mse, ddof=1)) if n > 1 else 0.0
            
            # Inflate SE to respect temporal correlation
            k = inflation_factor_k(mse - mse_mean, max_lag=max_lag) if n > 1 else 1.0
            se_mse = k * s_mse / np.sqrt(n) if n > 1 else np.nan
            
            # CI in regular space
            lo_mse = mse_mean - zcrit * se_mse if np.isfinite(se_mse) else np.nan
            hi_mse = mse_mean + zcrit * se_mse if np.isfinite(se_mse) else np.nan
            
            # Point estimate: arithmetic mean
            mse_hat = mse_mean

        rmse_hat = np.sqrt(mse_hat) if np.isfinite(mse_hat) else np.nan         # NEW
        lo_rmse  = np.sqrt(max(lo_mse, 0.0)) if np.isfinite(lo_mse) else np.nan # NEW
        hi_rmse  = np.sqrt(max(hi_mse, 0.0)) if np.isfinite(hi_mse) else np.nan # NEW

        rows.append({
            'model': model, 'lead': lead, 'n': n,
            'mse_hat': mse_hat, 'ci_lo': lo_mse, 'ci_hi': hi_mse,
            'rmse_hat': rmse_hat, 'rmse_ci_lo': lo_rmse, 'rmse_ci_hi': hi_rmse,  # NEW
            'k': k
        })

        # rows.append({
        #     'model': model, 'lead': lead, 'n': n,
        #     'mse_hat': mse_hat, 'ci_lo': lo_mse, 'ci_hi': hi_mse,
        #     'k': k
        # })
    
    return pd.DataFrame(rows).sort_values(['model','lead']).reset_index(drop=True)


# --- Precipitation loading utilities (adapted from your second script) ---
def _maybe_fix_lon(ds, lon_min=65.0, lon_max=95.0):
    """If dataset lon is 0..360 but bbox is in -180..180 (or vice versa), fix coords."""
    lons = ds['lon'].values
    if lons.min() >= 0 and lon_min < 0:
        # convert ds lon to -180..180
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    if lons.max() > 180 and lon_max <= 180:
        # ds is 0..360 but bbox is given as 65..95 -> OK, nothing to do
        pass
    return ds

def load_target_precip_from_accumulated(accumulated_nc_path,
                                       init_dates,
                                       varname='precipitation',
                                       india_lat_min=6.0, india_lat_max=38.0,
                                       india_lon_min=65.0, india_lon_max=95.0,
                                       start_month=8,
                                       lead_hours_per_step=6,
                                       n_steps=28,
                                       time_tolerance=pd.Timedelta('6h'),
                                       verbose=True):
    """
    Load precipitation data for given init dates and compute totals for India region.
    """
    print("Loading precip data \n\n")

    if not os.path.exists(accumulated_nc_path):
        print(f"Warning: Precipitation file {accumulated_nc_path} not found.")
        return {}
        
    try:
        ds = xr.open_dataset(accumulated_nc_path)
        if varname not in ds:
            print(f"Warning: variable '{varname}' not found. Available: {list(ds.data_vars)}")
            return {}

        # Fix lon convention if needed
        ds = _maybe_fix_lon(ds, lon_min=india_lon_min, lon_max=india_lon_max)

        # Subset to India bbox
        try:
            ds_india = ds[[varname]].sel(lat=slice(india_lat_min, india_lat_max),
                                         lon=slice(india_lon_min, india_lon_max))
        except Exception as e:
            if verbose:
                print(f"Warning: spatial slice raised: {e}, trying reversed order.")
            ds_india = ds[[varname]].sel(lat=slice(india_lat_max, india_lat_min),
                                         lon=slice(india_lon_max, india_lon_min))

        times = pd.to_datetime(ds_india['time'].values)
        if verbose:
            print(f"Precipitation dataset time range: {times.min()} to {times.max()}")

        offsets = [pd.Timedelta(hours=lead_hours_per_step * (i + 1)) for i in range(n_steps)]
        target_data = {}

        for init in tqdm(init_dates, desc='init_dates'):
            try:
                init_ts = pd.to_datetime(init)
            except Exception:
                continue

            if init_ts.month < start_month:
                continue

            vals = []
            for off in offsets:
                desired_time = init_ts + off
                try:
                    # Try exact match first
                    sel = ds_india[varname].sel(time=np.datetime64(desired_time))
                except (KeyError, IndexError, ValueError):
                    # Fallback to nearest
                    try:
                        sel = ds_india[varname].sel(time=desired_time, method='nearest', tolerance=time_tolerance)
                    except Exception:
                        sel = None

                if sel is None:
                    vals.append(np.nan)
                    continue

                # Sum over spatial dimensions
                try:
                    ssum = sel.sum(dim=['lat', 'lon']).values
                    ssum_scalar = float(ssum) if np.isscalar(ssum) else float(ssum.flatten()[0])
                except Exception:
                    ssum_scalar = np.nan

                vals.append(ssum_scalar)

            target_data[str(init_ts)] = np.array(vals, dtype=float)

        ds.close()
        print("Precip data")
        print(target_data)
        return target_data
        
    except Exception as e:
        print(f"Error loading precipitation data: {e}")
        return {}


def plot_mse_with_precipitation(summary_df,
                                precip_data=None,
                                baseline_model=None,
                                models_to_plot=None,
                                custom_colors=None,
                                models_rename=None,
                                title="RMSE comparison and % Improvement",
                                ymin=None, ymax=None,
                                savepath=None,
                                show_improvement=True):
    """
    Plot MSE for models with confidence intervals, plus precipitation on second y-axis.
    Optionally show % improvement subplot.
    """
    if summary_df.empty:
        raise ValueError("summary_df is empty.")

    # Restrict models if specified
    all_models = list(summary_df['model'].unique())
    if models_to_plot is None:
        models = all_models
    else:
        models = [m for m in models_to_plot if m in all_models]

    # Ensure baseline is included and listed first if specified
    if baseline_model and baseline_model in models:
        models = [baseline_model] + [m for m in models if m != baseline_model]

    leads = np.sort(summary_df['lead'].unique())

    # Colors: use custom if given, else fallback to matplotlib cycle
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    color_map = {}
    for i, m in enumerate(models):
        if custom_colors and m in custom_colors:
            color_map[m] = custom_colors[m]
        else:
            color_map[m] = 'black' if m == baseline_model else cycle[i % len(cycle)]

    # Create figure with subplots
    if show_improvement and baseline_model:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
        ax3 = None

    # Create second y-axis for precipitation
    ax2 = ax1.twinx()

    # ============ MSE plot (left y-axis) ============
    for m in models:
        g = summary_df[summary_df['model'] == m].sort_values('lead')
        if g.empty:
            continue
        x = g['lead'].values
        y = g['mse_hat'].values
        lo = g['ci_lo'].values
        hi = g['ci_hi'].values

        display_name = models_rename.get(m, m) if models_rename else m
        lw = 3.0 if m == baseline_model else 2.0
        z = 5 if m == baseline_model else 4
        
        ax1.plot(x, y, linestyle='-', linewidth=lw, color=color_map[m], 
                label=display_name, zorder=z)
        ax1.fill_between(x, lo, hi, color=color_map[m], 
                        alpha=(0.12 if m == baseline_model else 0.16), linewidth=0)

    # ============ Precipitation plot (right y-axis) ============
    if precip_data is not None and not precip_data.empty:
        precip_data_sorted = precip_data.sort_values('lead_time_hours')
        ax2.plot(precip_data_sorted['lead_time_hours'], precip_data_sorted['precipitation'], 
                linestyle='-', color='olive', label='Total Precipitation', 
                linewidth=3, alpha=0.2, zorder=1)

    # Configure main axis (MSE)
    ax1.set_title(title, fontsize=25, pad=20)
    ax1.set_ylabel('Avg RMSE (Lower is Better)', fontsize=20, color='black')
    ax1.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.4)
    ax1.tick_params(axis='both', labelsize=18)
    
    # Configure precipitation axis
    ax2.set_ylabel('Total Precipitation (m)', fontsize=20, color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=18)

    # y-limits for MSE
    ymin = 0
    ymax = 10e-5
    pptymin = 1e7
    pptymax = 1.7e7
    if ymin is not None or ymax is not None:
        ax1.set_ylim(bottom=ymin if ymin is not None else ax1.get_ylim()[0]*0,
                     top=ymax*1.5 if ymax is not None else ax1.get_ylim()[1]*1.5)

    # y_ticks_as_ints_with_e_minus3(ax1) 
    # Force scientific notation with offset text
    sf = ScalarFormatter(useMathText=True)
    sf.set_scientific(True)
    sf.set_powerlimits((0, 0))  # always use scientific notation
    sf.set_useOffset(False)

    ax1.yaxis.set_major_formatter(sf)


    ax2.set_ylim(bottom=pptymin if pptymin is not None else ax2.get_ylim()[0]*0,
                top=pptymax if pptymax is not None else ax2.get_ylim()[1]*1.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
              fontsize=20, ncol=2 if len(lines1 + lines2) > 4 else 1)

    # ============ % Improvement plot (if requested) ============
    if show_improvement and baseline_model and ax3 is not None:
        baseline_vals = summary_df[summary_df['model'] == baseline_model].sort_values('lead')
        for m in models:
            if m == baseline_model:
                continue
            g = summary_df[summary_df['model'] == m].sort_values('lead')
            if g.empty:
                continue
            
            # Merge with baseline
            merged = pd.merge(baseline_vals[['lead','mse_hat']], g[['lead','mse_hat']], 
                             on='lead', suffixes=('_base','_model'))
            # For MSE, improvement means reduction (negative % change)
            improvement = -100 * (merged['mse_hat_model'] - merged['mse_hat_base']) / merged['mse_hat_base']
            
            display_name = models_rename.get(m, m) if models_rename else m
            ax3.plot(merged['lead'], improvement, linestyle='-', linewidth=2.0, 
                    color='#2D6DAB', label=display_name)

        ax3.axhline(0, color='black', linewidth=1, linestyle='--')
        ax3.set_xlabel('Lead time (hours)', fontsize=20)
        ax3.set_ylabel('% Improvement', fontsize=20)
        ax3.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.4)
        ax3.tick_params(axis='both', labelsize=18)
        if len([m for m in models if m != baseline_model]) > 1:
            ax3.legend(fontsize=25)

    # Set x-axis ticks
    if ax3 is not None:
        for ax in [ax1, ax3]:
            ax.xaxis.set_major_locator(MultipleLocator(24))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
            ax.set_xlim(leads.min(), leads.max()+6)
    else:
        ax1.set_xlabel('Lead time (hours)', fontsize=16)
        ax1.xaxis.set_major_locator(MultipleLocator(24))
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
        ax1.set_xlim(leads.min(), leads.max()+6)

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {savepath}")
    
    plt.show()


# ============ Main execution function ============
def create_mse_precipitation_plot(csv_path, 
                                 accumulated_nc_path=None,
                                 baseline_model="Graphcast_Base_2014-08-01_2014-09-30,India",
                                 models_to_plot=None,
                                 models_rename=None,
                                 alpha=0.20,
                                 max_lag=2,
                                 log_transform=True,
                                 custom_colors=None,
                                 savepath="plots/mse_with_precipitation.png",
                                 show_improvement=True):
    """
    Main function to create MSE plot with precipitation overlay.
    """
    # Load forecast data
    print(f"Loading forecast data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Handle different horizon column names
    if 'forecast_horizon_hours' in df.columns:
        lead_col = 'forecast_horizon_hours'
    elif 'lead_time_hours' in df.columns:
        lead_col = 'lead_time_hours'
    else:
        # Try to parse from other formats
        if 'forecast_horizon_hours' in df.columns:
            df['lead_time_hours'] = pd.to_timedelta(df['forecast_horizon_hours']).dt.total_seconds() / 3600
            lead_col = 'lead_time_hours'
        elif 'lead_time' in df.columns:
            # Handle format like "0 days 06:00:00"
            df['lead_time_hours'] = pd.to_timedelta(df['lead_time']).dt.total_seconds() / 3600
            lead_col = 'lead_time_hours'
        else:
            raise ValueError("Could not find lead time column in CSV. Expected one of: 'forecast_horizon_hours_times', 'lead_time_hours', 'forecast_horizon_hours', 'lead_time'")
    
    # Filter models if specified
    if models_to_plot:
        df = df[df['model'].isin(models_to_plot)]
        print(f"Filtered to models: {models_to_plot}")
    
    # Summarize MSE with confidence intervals
    print("Computing MSE summary with confidence intervals...")
    mse_summary = summarize_mse_by_model(
        df,
        lead_col=lead_col,
        init_col='init_date',
        model_col='model',
        mse_col='mse',
        alpha=alpha,
        max_lag=max_lag,
        log_transform=log_transform
    )
    
    # Load precipitation data if path provided
    precip_data = None
    print(accumulated_nc_path)
    print(f"Path exists {os.path.exists(accumulated_nc_path)}")
    if os.path.exists('2014_precip_data.csv'):
        precip_data = pd.read_csv('2014_precip_data.csv')
    else:
        if accumulated_nc_path and os.path.exists(accumulated_nc_path):
            print("Loading precipitation data...")
            unique_init_dates = df['init_date'].unique()
            target_data = load_target_precip_from_accumulated(
                accumulated_nc_path,
                unique_init_dates,
                start_month=8  # August onward
            )
            
            if target_data:
                print("We have precip target data")
                # Convert to DataFrame with totals by lead time
                precip_rows = []
                for init_date, precip_values in target_data.items():
                    for i, precip_val in enumerate(precip_values):
                        if not np.isnan(precip_val):
                            horizon_hours = (i + 1) * 6
                            precip_rows.append({
                                'init_date': init_date,
                                'lead_time_hours': horizon_hours,
                                'precipitation': precip_val
                            })
                
                if precip_rows:
                    precip_df = pd.DataFrame(precip_rows)
                    # Sum precipitation across all init dates for each lead time
                    precip_data = precip_df.groupby('lead_time_hours')['precipitation'].sum().reset_index()
                    print(f"Loaded precipitation data for {len(precip_data)} lead times")
    
    # Create the plot
    print("Creating plot...")
    plot_mse_with_precipitation(
        summary_df=mse_summary,
        precip_data=precip_data,
        baseline_model=baseline_model,
        models_to_plot=models_to_plot,
        custom_colors=custom_colors,
        models_rename=models_rename,
        savepath=savepath,
        show_improvement=show_improvement
    )
    
    return mse_summary, precip_data


# ============ Example usage ============
if __name__ == "__main__":
    # Example configuration
    csv_path = "skill_score_India_2025-08-2021-42-34_mse_false.csv" # Your MSE CSV file
    csv_path1 = "skill_score_proper1_conv_2014.csv" # Your MSE CSV file

        # Read the CSV
    df = pd.read_csv(csv_path)

    # Convert forecast_horizon_hours_times to Timedelta and then to total hours
    df["forecast_horizon_hours"] = pd.to_timedelta(df["forecast_horizon"]).dt.total_seconds() / 3600

    # Optionally drop the old column and rename
    df = df.drop(columns=["forecast_horizon"]).rename(columns={"forecast_horizon_hours": "forecast_horizon_hours"})

    # Save back to CSV
    df.to_csv(csv_path1, index=False)

    print(df.head())
    accumulated_nc_path = '/Datastorage/saptarishi.dhanuka_asp25/imerg30min/imerg_2014_6h_accumulated.nc'  # Path to your precipitation NC file or None
    
    custom_colors = {
        "Graphcast_Base_2014-08-01_2014-09-30": "orange",
        "Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz": "#2D6DAB",
        "Graphcast_Finetuned_2_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_mask_expt3_good.npz": "red"
    }
    
    models_rename = {
        "Graphcast_Base_2014-08-01_2014-09-30": "Base",
        "Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz": "Finetuned"
    }
    
    # Create the plot
    mse_summary, precip_data = create_mse_precipitation_plot(
        csv_path=csv_path1,
        accumulated_nc_path=accumulated_nc_path,
        baseline_model="Graphcast_Base_2014-08-01_2014-09-30",
        models_to_plot=["Graphcast_Base_2014-08-01_2014-09-30", "Graphcast_Finetuned_1_graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz"],
        models_rename=models_rename,
        custom_colors=custom_colors,
        alpha=0.20,  # 80% CI
        log_transform=True,  # Use log-space for MSE (recommended)
        savepath="plots/mse_with_precipitation.png",
        show_improvement=True
    )
    
    print("MSE Summary:")
    # print(mse_summary)
    if precip_data is not None:
        precip_data.to_csv('2014_precip_data.csv')
        print("\nPrecipitation Data:")
        print(precip_data.head())