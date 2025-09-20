"""
python /home/saptarishi.dhanuka_asp25/weather/graphcast_dir/graphcast/local_files/eval_forecast.py \ 
--eval_start "2024-08-01" \ 
--eval_end "2024-09-15" \ 
--eval_dataset_choice "imerg" \ 
--vars_to_eval "total_precipitation_6hr"
--params-path-new "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2024-06-01_2024-07-30_FORECAST28.npz"
"""

"""
Complete evaluation of forecast against different datasets
"""
import os
import sys
import logging
import argparse
import dataclasses
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
import zarr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from collections import defaultdict
from scipy import stats


import jax
import optax

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from graphcast import checkpoint, data_utils, rollout, graphcast, normalization
import save_params_utils
import setup_jax_functions
from plotting import scale, select, plot_data, save_animation, save_static_plot, plot_sample_from_ds
from metrics import compute_rmse, compute_mae, compute_bias, compute_acc
from utils import regrid_hres_fine_to_coarse, generate_sample_era5_dataset, grads_fn, parse_args, process_to_graphcast_format, compute_mse, compute_mse_diffs

def compute_difference_with_targets_sims(preds_old, targets, time_step, var_name, plots_dir='./output', plot=True):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [68, 98, 6, 38]  # Focus on India

    var_pred_old = preds_old[var_name]
    var_targets = targets[var_name]

    if var_pred_old.lat.values[0] - var_pred_old.lat.values[1] < 0:
        var_base = var_pred_old.isel(time=time_step).sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax)).squeeze()
    else:
        var_base = var_pred_old.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()

    if var_targets.lat.values[0] - var_targets.lat.values[1] < 0:
        var_targets = var_targets.isel(time=time_step).sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax)).squeeze()
    else:
        var_targets = var_targets.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()

    difference = var_base - var_targets


    if plot:
        difference.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
            cbar_kwargs={'label': 'Difference'})
        ax.set_extent(extent)
        ax.coastlines()
        ax.set_title(f"Finetuned Prediction vs Ground Truth at Time Step {time_step}")

        plt.tight_layout()
        plt.savefig(f'{plots_dir}/difference_base{time_step}_era5.png')
        plt.close()

    return (difference)

# --- NEW FUNCTION FOR STATISTICAL TESTING ---
def perform_significance_test(series_finetuned_rmse, series_base_rmse):
    """
    Performs a paired t-test with correction for auto-correlation on RMSE series.
    
    This follows the methodology of correcting the sample size based on the 
    lag-1 auto-correlation of the difference series, as described in meteorological
    verification literature. A negative t-statistic indicates the finetuned
    model has lower mean RMSE (is better).

    Args:
        series_finetuned_rmse: A list or array of RMSE scores for the finetuned model.
        series_base_rmse: A list or array of RMSE scores for the base model.
    
    Returns:
        A tuple containing:
        - mean_diff (float): Mean of (RMSE_finetuned - RMSE_base).
        - t_stat (float): The calculated t-statistic.
        - p_value (float): The two-sided p-value.
        - n_eff (float): The effective sample size after auto-correlation correction.
    """
    diff_series = np.array(series_finetuned_rmse) - np.array(series_base_rmse)
    n = len(diff_series)

    if n < 4:  # Need sufficient samples to estimate auto-correlation and perform test
        return np.nan, np.nan, np.nan, n

    # Calculate lag-1 auto-correlation of the difference series
    # Using pandas is convenient and robust for this
    r1 = pd.Series(diff_series).autocorr(lag=1)
    if pd.isna(r1):
        r1 = 0  # If no correlation can be computed (e.g., constant series)

    # Calculate effective sample size (Zwiers and von Storch, 1995)
    # This corrects for the reduction in degrees of freedom due to auto-correlation.
    if abs(r1) < 0.99:
        n_eff = n * (1 - r1) / (1 + r1)
    else:  # If highly correlated, effective sample size is very small
        n_eff = 2.0
    
    n_eff = max(2.0, n_eff) # Ensure at least 2 for df > 0

    # Perform one-sample t-test on the differences
    # The null hypothesis is that the mean difference is 0.
    mean_diff = np.mean(diff_series)
    std_diff = np.std(diff_series, ddof=1)

    # The standard error is adjusted with n_eff
    if std_diff > 0:
        se_eff = std_diff / np.sqrt(n_eff)
        t_stat = mean_diff / se_eff
    else: # No variance in the difference series
        t_stat = 0.0 if mean_diff == 0 else np.inf * np.sign(mean_diff)

    # Two-sided p-value from the t-distribution
    p_value = stats.t.sf(np.abs(t_stat), df=n_eff - 1) * 2

    return mean_diff, t_stat, p_value, n_eff


sys.path.append('/home/saptarishi.dhanuka_asp25/weather/graphcast_dir/gc_dist')
import trainer.dataloader
from dist_utils import construct_era5_imerg

args = parse_args()

eval_start = args.eval_start
eval_end = args.eval_end
dataset_choice = args.eval_dataset_choice
eval_vars = args.vars_to_eval
apath = args.eval_data_path
params_path_old = args.params_path_old
params_path_new = args.params_path_new
norms_dir = args.norms_dir
plots_dir = args.plots_dir
latmin, latmax, lonmin, lonmax = args.latmin, args.latmax, args.lonmin, args.lonmax
plot_timesteps = args.plot_timesteps
output_pred_old_dir = args.output_pred_old_dir
output_pred_finetuned_dir = args.output_pred_finetuned_dir



if dataset_choice == "imerg":
    apath = '/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/'
    dbase,_ = trainer.dataloader.open_databases(apath,None) # Note no need for a separate verification dbase
    dbase = construct_era5_imerg(dbase)
    eval_time_ds = dbase.sel(time=slice(eval_start, eval_end))
    del dbase


select_time_eval = process_to_graphcast_format(eval_time_ds)
print(f"Time range: {select_time_eval.time.values[0]} - {select_time_eval.time.values[-1]}")
data_len = len(select_time_eval.time.values)

select_time_evals = []
for time_slice in range(0, data_len, 28):
    select_time_evals.append(select_time_eval.isel(time=slice(time_slice, time_slice+28)))


print('Sims initialisations')
for ds in select_time_evals:
    print(ds.datetime.values[0][0])


sims = len(select_time_evals)
print(f"Num sims: {sims}")



eval_trial_batches = []
from dask.diagnostics import ProgressBar
for i in tqdm(range(sims-1), desc="Computing sims"):
    with ProgressBar():
        eval_trial_batches.append(select_time_evals[i].load())


with open(params_path_old, 'rb') as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

params = ckpt.params

with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/diffs_stddev_by_level.nc', 'rb') as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()
with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/stddev_by_level.nc', 'rb') as f:
    stddev_by_level = xr.load_dataset(f).compute()
with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/mean_by_level.nc', 'rb') as f:
    mean_by_level = xr.load_dataset(f).compute()

state = {}
model_config = ckpt.model_config
task_config = ckpt.task_config
setup_jax_functions.configs['model_config'] = model_config
setup_jax_functions.configs['task_config'] = task_config
setup_jax_functions.configs['state'] = state
setup_jax_functions.configs['params'] = params
setup_jax_functions.configs['stddev_by_level'] = stddev_by_level
setup_jax_functions.configs['diffs_stddev_by_level'] = diffs_stddev_by_level
setup_jax_functions.configs['mean_by_level'] = mean_by_level

setup_jax_functions.update_configs({
'params': ckpt.params,
'state': {},
'model_config': ckpt.model_config,
'task_config': ckpt.task_config,
'mean_by_level': mean_by_level,
'stddev_by_level': stddev_by_level,
'diffs_stddev_by_level': diffs_stddev_by_level
})

init_jitted = jax.jit(setup_jax_functions.with_configs(setup_jax_functions.run_forward.init))
loss_fn_jitted = setup_jax_functions.drop_state(setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(setup_jax_functions.loss_fn.apply))))
grads_fn_jitted = setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(setup_jax_functions.grads_fn)))
run_forward_jitted = setup_jax_functions.drop_state(setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(
setup_jax_functions.run_forward.apply))))

def run_model(params, state, inputs, targets_template, forcings):
    predictions = run_forward_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=inputs,
    targets_template=targets_template,
    forcings=forcings,
    params=params,
    state=state
)
    return predictions

f = open(params_path_new, "rb")
new_ckpt = checkpoint.load(f, graphcast.CheckPoint)
new_params = new_ckpt.params
f.close()
params_keys_list = list(new_params.keys())
any_not_nan = any(not np.isnan(x) for x in list(new_params[f'{params_keys_list[0]}'].values())[1])
if not any_not_nan:
    print("All values are NaN in params!")
    exit()
jax.config.update("jax_enable_x64", True)

jax.config.update("jax_enable_x64", True)

def compute_mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def compute_mse_diffs(diff):
    return (diff ** 2).mean(dim=["lat", "lon"]).values

# Combine list of xarray Datasets into a single xarray Dataset via summing
def accumulate_diff(diff_list, use_abs=False):
    accumulated = None
    for ds in diff_list:
        var = ds
        if use_abs:
            var = np.abs(var)
        if accumulated is None:
            accumulated = var.copy(deep=True)
        else:
            accumulated += var
    return accumulated

# Plotting function
def plot_accumulated_diff(data_array, title, simnum, cmap='RdBu', vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = data_array.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': 'Accumulated Error (mm)'}
    )
    ax.coastlines()
    ax.set_title(title)
    ax.set_extent([65, 100, 5, 40])  # Set extent over India
    plt.savefig(f'./output/20240601_20240730_train/{title}_sim_{simnum}_accumulated_diff_map')


# --- MODIFICATION: Initialize data structures for statistical analysis ---
# We will collect the RMSE for each model at each lead time across all initializations
rmse_results_base = defaultdict(list)
rmse_results_finetuned = defaultdict(list)
lead_times_in_hours = None


for i in tqdm(range(sims-1), desc="Sims"):

    time_len = len(eval_trial_batches[i].time.values)
    eval_sim_data = eval_trial_batches[i].isel(time=slice(0, time_len-1))
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    eval_sim_data, target_lead_times=slice("6h", "100h"),
    **dataclasses.asdict(task_config))
    forecast_len = len(eval_targets.time.values)

    assert eval_trial_batches[i].sizes["time"] >= 3

    print(f"\nRunning sim {i+1}/{sims-1} | Eval batch time dimensions: {eval_trial_batches[i].sizes['time']}")

    # --- MODIFICATION: Get lead times in hours (only on first sim) ---
    if i == 0:
        init_time = eval_inputs.time.values[-1]
        lead_times_in_hours = [int((t - init_time) / np.timedelta64(1, 'h')) for t in eval_targets.time.values]
        print(f"Forecast lead times (hours): {lead_times_in_hours}")


    task_config_dict =  dataclasses.asdict(task_config)

    targets_template = eval_targets * np.nan

    print("Running base model...")
    predictions_old = run_model(params, state, eval_inputs, targets_template, eval_forcings)

    print("Running finetuned model...")
    predictions_finetuned = run_model(new_params, state, eval_inputs, targets_template, eval_forcings)


    diff_fine, diff_old = [], []
    for time_step in tqdm(range(forecast_len), desc='Diff compute over timesteps', leave=False):
        diff_fine.append(compute_difference_with_targets_sims(predictions_finetuned,eval_targets,time_step, 'total_precipitation_6hr', plot=False))
        diff_old.append(compute_difference_with_targets_sims(predictions_old,eval_targets,time_step, 'total_precipitation_6hr', plot=False))

    # ---- Accumulate and plot raw diffs ----
    accum_fine_raw = accumulate_diff(diff_fine, use_abs=False)
    accum_old_raw = accumulate_diff(diff_old, use_abs=False)
    plot_accumulated_diff(accum_fine_raw, 'Accumulated Raw Difference (Fine-Tuned)', simnum=i)
    plot_accumulated_diff(accum_old_raw, 'Accumulated Raw Difference (Old)', simnum=i)

    # ---- Accumulate and plot absolute diffs ----
    accum_fine_abs = accumulate_diff(diff_fine, use_abs=True)
    accum_old_abs = accumulate_diff(diff_old, use_abs=True)
    plot_accumulated_diff(accum_fine_abs, 'Accumulated Absolute Difference (Fine-Tuned)',simnum=i, cmap='Reds', vmin=0)
    plot_accumulated_diff(accum_old_abs, 'Accumulated Absolute Difference (Old)',simnum=i, cmap='Reds', vmin=0)
        
    # Compute the MSE for each time step and aggregate
    mse_finetuned = []
    mse_base = []

    for diff_counter in range(len(diff_fine)):
        mse_diff_finetuned = compute_mse_diffs(diff_fine[diff_counter])
        mse_diff_base = compute_mse_diffs(diff_old[diff_counter])
        
        mse_finetuned.append(mse_diff_finetuned)
        mse_base.append(mse_diff_base)

    # Plot the line chart for the current simulation
    plt.figure(figsize=(10, 6))
    plt.plot(lead_times_in_hours, mse_base, label='MSE Base', marker='o')
    plt.plot(lead_times_in_hours, mse_finetuned, label='MSE Finetuned', marker='o')

    plt.xlabel('Lead Time (hours)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'MSE Comparison for initialization: {eval_trial_batches[i].time.values[0]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./output/20240601_20240730_train/mse_comparison_sim_{i}.png")
    plt.close()

    # --- MODIFICATION: Store RMSE results for final statistical analysis ---
    for j in range(forecast_len):
        rmse_results_base[j].append(np.sqrt(mse_base[j]))
        rmse_results_finetuned[j].append(np.sqrt(mse_finetuned[j]))


# --- NEW SECTION: Perform and Display Statistical Significance Test Results ---
print("\n" + "="*20)
print("STATISTICAL SIGNIFICANCE TEST: Finetuned Model vs. Base Model")
print(f"Paired t-test on RMSE for '{eval_vars}' with auto-correlation correction.")
print(f"Total initializations (nominal sample size n): {sims-1}")
print("A negative t-statistic means the finetuned model has a lower average RMSE.")
print("p-value < 0.05 suggests a statistically significant difference.")
print("="*80)
print(f"{'Lead Time':>12} | {'Mean RMSE Diff':>15} | {'N_eff':>8} | {'t-statistic':>15} | {'p-value':>12}")
print("-"*80)

# Make sure lead_times_in_hours has been populated
if lead_times_in_hours:
    for idx, lead_time in enumerate(lead_times_in_hours):
        # Retrieve the time series of RMSEs for this lead time
        base_rmses = rmse_results_base[idx]
        finetuned_rmses = rmse_results_finetuned[idx]

        # Perform the significance test
        mean_diff, t_stat, p_val, n_eff = perform_significance_test(finetuned_rmses, base_rmses)
        
        # Format and print results in a table
        print(f"{lead_time:>10} h | {mean_diff:>15.5f} | {n_eff:>8.2f} | {t_stat:>15.5f} | {p_val:>12.5f}")
else:
    print("Could not perform statistical tests: No simulation data was processed.")

print("="*80)