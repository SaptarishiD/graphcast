# <eval_forecast.py>
"""
python /home/saptarishi.dhanuka_asp25/weather/graphcast_dir/graphcast/local_files/eval_forecast.py \ 
--eval_start "2024-08-01" \ 
--eval_end "2024-09-15" \ 
--eval_dataset_choice "imerg" \ 
--vars_to_eval "total_precipitation_6hr" \ 
--params_path_new1 "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2024-06-01_2024-09-15_FORECAST28.npz" \ 
--params_path_new2 "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2024-06-01_2024-07-30_FORECAST28.npz" \
--output_csv_path "./evaluation_results/forecast_mse.csv"
"""

"""
Complete evaluation of forecast against different datasets with rainfall analysis
"""
# <eval_forecast.py>
# Parameters
eval_start = "2024-08-01"
eval_end = "2024-09-05"
dataset_choice = "imerg"
eval_vars = "total_precipitation_6hr"
apath = "/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/"
params_path_old = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/origs/graphcast_1_13.npz'

params_path_new1 = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_shapefile_2024-06-01_2024-07-30_FORECAST28_new.npz'
params_path_new2 = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_shapefile_2024-06-01_2024-07-30_FORECAST28.npz'
norms_dir = '/Datastorage/saptarishi.dhanuka_asp25/gc_norms/'
plots_dir = 'plots/evals'
latmin, latmax, lonmin, lonmax = 6, 38, 35, 65
plot_timesteps = 7
output_pred_old_dir = '/Datastorage/saptarishi.dhanuka_asp25/preds_dir/'
output_pred_finetuned_dir = '/Datastorage/saptarishi.dhanuka_asp25/preds_dir/'

"""
Complete evaluation of forecast against different datasets with rainfall analysis
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

import jax
import optax

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
from graphcast import checkpoint, data_utils, rollout, graphcast, normalization
import setup_jax_functions
from plotting import scale, select, plot_data, save_animation, save_static_plot, compute_difference_with_targets_sims, plot_sample_from_ds
from metrics import compute_rmse, compute_mae, compute_bias, compute_acc
from utils import regrid_hres_fine_to_coarse, generate_sample_era5_dataset, grads_fn, parse_args, process_to_graphcast_format, compute_mse, compute_mse_diffs

sys.path.append('/home/saptarishi.dhanuka_asp25/weather/graphcast_dir/gc_dist')
import trainer.dataloader
from dist_utils import construct_era5_imerg
from datetime import datetime

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
args = parse_args()

current_date = datetime.now().strftime('%Y-%m-%d%H-%M-%S')

# Add new argument to the parser in utils.py
# For now, let's handle it here if it's not in the original file
if not hasattr(args, 'output_csv_path'):
    args.output_csv_path = f"./forecast_evaluation_mse_{current_date}.csv"

eval_start = args.eval_start
eval_end = args.eval_end
dataset_choice = args.eval_dataset_choice
eval_vars = args.vars_to_eval
apath = args.eval_data_path
params_path_old = args.params_path_old
params_path_new1 = args.params_path_new1
params_path_new2 = args.params_path_new2

# Define India bounding box
INDIA_BBOX = {'lat_min': 6, 'lat_max': 37, 'lon_min': 68, 'lon_max': 97}

# --- Data and Model Loading ---
logging.info(f"Loading dataset '{dataset_choice}' from {eval_start} to {eval_end}")
if dataset_choice == "imerg":
    apath = '/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/'
    dbase,_ = trainer.dataloader.open_databases(apath,None)
    dbase = construct_era5_imerg(dbase)
    # Load a slightly larger window to ensure we have the day before the start_date for initialization
    load_start_date = pd.to_datetime(eval_start) - pd.Timedelta(days=1)
    eval_time_ds = dbase.sel(time=slice(load_start_date.strftime('%Y-%m-%d'), eval_end))
    del dbase

select_time_eval = process_to_graphcast_format(eval_time_ds)
logging.info(f"Full data range loaded: {select_time_eval.time.values[0]} to {select_time_eval.time.values[-1]}")

logging.info("Loading models and normalization stats...")
with open(params_path_old, 'rb') as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params

with open(params_path_new1, "rb") as f:
    new_params1 = checkpoint.load(f, graphcast.CheckPoint).params
with open(params_path_new2, "rb") as f:
    new_params2 = checkpoint.load(f, graphcast.CheckPoint).params

with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/diffs_stddev_by_level.nc', 'rb') as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()
with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/stddev_by_level.nc', 'rb') as f:
    stddev_by_level = xr.load_dataset(f).compute()
with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/mean_by_level.nc', 'rb') as f:
    mean_by_level = xr.load_dataset(f).compute()

# --- JAX Function Setup ---
logging.info("Setting up JAX functions...")
state = {}
model_config = ckpt.model_config
task_config = ckpt.task_config
setup_jax_functions.update_configs({
    'params': ckpt.params, 'state': state, 'model_config': ckpt.model_config, 'task_config': task_config,
    'mean_by_level': mean_by_level, 'stddev_by_level': stddev_by_level, 'diffs_stddev_by_level': diffs_stddev_by_level
})
run_forward_jitted = setup_jax_functions.drop_state(setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(
    setup_jax_functions.run_forward.apply))))
jax.config.update("jax_enable_x64", True)

def run_model(params, state, inputs, targets_template, forcings):
    return run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
        params=params,
        state=state
    )

# --- Helper Functions for Rainfall Analysis ---
def subset_to_india(data_array):
    """Subset xarray data to India bounding box, with automatic lat/lon ordering check."""
    # Determine if latitude is ascending or descending
    lat_ascending = data_array.lat[0] < data_array.lat[-1]
    lon_ascending = data_array.lon[0] < data_array.lon[-1]

    # Create slices based on ordering
    lat_slice = slice(INDIA_BBOX['lat_min'], INDIA_BBOX['lat_max']) if lat_ascending \
                else slice(INDIA_BBOX['lat_max'], INDIA_BBOX['lat_min'])
    
    lon_slice = slice(INDIA_BBOX['lon_min'], INDIA_BBOX['lon_max']) if lon_ascending \
                else slice(INDIA_BBOX['lon_max'], INDIA_BBOX['lon_min'])

    return data_array.sel(lat=lat_slice, lon=lon_slice)


def calculate_total_rainfall(data_array, subset_region=True):
    """Calculate total rainfall sum over spatial dimensions"""
    if subset_region:
        data_array = subset_to_india(data_array)
    return data_array.sum(dim=['lat', 'lon'])

def plot_spatial_map(data, title, save_path, vmin=None, vmax=None, cmap='Blues'):
    """Plot spatial map with India boundaries"""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent to India
    ax.set_extent([INDIA_BBOX['lon_min'], INDIA_BBOX['lon_max'], 
                   INDIA_BBOX['lat_min'], INDIA_BBOX['lat_max']], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
    
    # Plot data
    im = ax.contourf(data.lon, data.lat, data, levels=20, transform=ccrs.PlateCarree(),
                     cmap=cmap, vmin=vmin, vmax=vmax, extend='max')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='mm/6hr')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_rainfall_plots(ground_truth, predictions_dict, init_date, forecast_step, output_dir):
    """Create spatial plots for ground truth, predictions, and differences"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Subset all data to India
    gt_india = subset_to_india(ground_truth)
    
    # Determine common color scale
    all_data = [gt_india]
    pred_india = {}
    for name, pred in predictions_dict.items():
        if eval_vars in pred:
            pred_data = subset_to_india(pred[eval_vars])
        else:
            pred_data = subset_to_india(pred)
        pred_india[name] = pred_data
        all_data.append(pred_data)
    
    # Calculate common vmin, vmax
    vmin = 0
    vmax = float(np.nanmax([np.nanmax(d.values) for d in all_data]))
    
    # Plot ground truth
    plot_spatial_map(gt_india, f'Ground Truth - {init_date.strftime("%Y-%m-%d")} - Step {forecast_step}',
                     f'{output_dir}/ground_truth_{init_date.strftime("%Y%m%d")}_step_{forecast_step:02d}.png',
                     vmin=vmin, vmax=vmax)
    
    # Plot predictions
    for model_name, pred_data in pred_india.items():
        plot_spatial_map(pred_data, f'{model_name} - {init_date.strftime("%Y-%m-%d")} - Step {forecast_step}',
                         f'{output_dir}/{model_name.lower()}_{init_date.strftime("%Y%m%d")}_step_{forecast_step:02d}.png',
                         vmin=vmin, vmax=vmax)
    
    # Plot differences between predictions
    model_names = list(pred_india.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            diff = pred_india[model_names[i]] - pred_india[model_names[j]]
            diff_max = float(np.nanmax(np.abs(diff.values)))
            plot_spatial_map(diff, f'Difference: {model_names[i]} - {model_names[j]} - Step {forecast_step}',
                           f'{output_dir}/diff_{model_names[i].lower()}_{model_names[j].lower()}_step_{forecast_step:02d}.png',
                           vmin=-diff_max, vmax=diff_max, cmap='RdBu_r')

# --- HRES Data Handling ---
import glob
def extract_hres_date(path):
    parts = path.split('_')
    year = 2024  # fixed, or extract if you want dynamic
    month = int(parts[5])
    day = int(parts[6])
    return datetime(year, month, day)

logging.info("Mapping HRES forecast files...")
hres_file_list = glob.glob("/Datastorage/saptarishi.dhanuka_asp25/forecasts_hres/raw_hres/2024/*.grib")
hres_files_map = {extract_hres_date(p): p for p in hres_file_list}

# =============================================================================
# === SYSTEMATIC EVALUATION WITH RAINFALL ANALYSIS ===========================
# =============================================================================
all_results = []
all_rainfall_totals = []  # New: store total rainfall data
initialization_dates = pd.to_datetime(pd.date_range(start=eval_start, end=eval_end, freq='D'))

# Max forecast length is 7 days (28 steps of 6 hours)
MAX_FORECAST_STEPS = 28
target_lead_times_str = f"{(MAX_FORECAST_STEPS) * 6}h"
target_lead_times_slice = slice("6h", target_lead_times_str)
forecast_horizons_def = {1: 4, 3: 12, 7: 28}  # In 6-hourly steps

logging.info(f"Starting evaluation for {len(initialization_dates)} initialization dates.")

# Create output directories for plots
plots_dir = "./evaluation_results/spatial_plots"
os.makedirs(plots_dir, exist_ok=True)

for init_idx, init_date in enumerate(tqdm(initialization_dates, desc="Evaluating Forecasts")):
    logging.info(f"Processing initialization date: {init_date}")

    # 1. Load data for this initialization
    try:
        start_slice = init_date - pd.Timedelta(hours=6)
        end_slice = init_date + pd.Timedelta(days=7)
        start_slice -= select_time_eval.datetime.values[0][0]
        end_slice -= select_time_eval.datetime.values[0][0]

        from dask.diagnostics import ProgressBar
        with ProgressBar():
            eval_sim_data = select_time_eval.sel(time=slice(start_slice, end_slice)).compute()

        if eval_sim_data.sizes["time"] < 2:
            logging.warning(f"Not enough data for initialization {init_date}. Found {eval_sim_data.sizes['time']} steps. Skipping.")
            continue
    except Exception as e:
        logging.warning(f"Could not load data for {init_date}: {e}. Skipping.")
        continue

    # 2. Prepare inputs, targets, and forcings for a 7-day rollout
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        eval_sim_data, target_lead_times=target_lead_times_slice, **dataclasses.asdict(task_config))
    
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)

    if (eval_inputs.dims.mapping['time'] != 2):
        print("\nError in extracting inputs targets and forcings\n")
        continue

    if eval_targets.sizes['time'] < MAX_FORECAST_STEPS:
        logging.warning(f"Not enough target data for a 7-day forecast from {init_date}. Have {eval_targets.sizes['time']} steps. Skipping.")
        continue

    targets_template = eval_targets * np.nan
    ground_truth_var = eval_targets[eval_vars]

    # 3. Run all Graphcast models
    logging.debug(f"Running Graphcast models for {init_date}")
    print("Base run")
    predictions_base = run_model(params, state, eval_inputs, targets_template, eval_forcings)
    print("Finetuned 1")
    predictions_ft1 = run_model(new_params1, state, eval_inputs, targets_template, eval_forcings)
    print("Finetuned 2")
    predictions_ft2 = run_model(new_params2, state, eval_inputs, targets_template, eval_forcings)
    
    models_to_eval = {
        'Graphcast_Base': predictions_base,
        'Graphcast_Finetuned1': predictions_ft1,
        'Graphcast_Finetuned2': predictions_ft2,
    }

    # 4. Load, regrid, and align HRES forecast
    hres_predictions = None
    hres_file_path = hres_files_map.get(init_date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0))
    if hres_file_path:
        try:
            logging.debug(f"Processing HRES file: {hres_file_path}")
            hres_ds = xr.open_dataset(hres_file_path, engine='cfgrib')
            hres_regridded = regrid_hres_fine_to_coarse(hres_ds, variable='tp', coarse_resolution=1.0)
            hres_predictions = hres_regridded.drop_vars({'time'}).rename({'step': 'time'}).isel(time=slice(1,None)).assign_coords(time=eval_targets.time)
            models_to_eval['HRES'] = hres_predictions
        except Exception as e:
            logging.warning(f"Failed to process HRES file {hres_file_path}: {e}")
    else:
        logging.warning(f"No HRES file found for init date {init_date}")

    # 5. Calculate total rainfall for ground truth
    ground_truth_rainfall_totals = calculate_total_rainfall(ground_truth_var).values

    # 6. Calculate MSE and total rainfall for each model and forecast horizon
    for model_name, predictions in tqdm(models_to_eval.items(), desc="MSE Calc"):
        if eval_vars in predictions:
            pred_var = predictions[eval_vars]
        else:
            pred_var = predictions

        times = pred_var.time.values
        
        # Calculate total rainfall for this model
        model_rainfall_totals = calculate_total_rainfall(pred_var).values
            
        for step_idx, timestep in enumerate(times):
            # Slice predictions and targets to the current forecast horizon
            pred_sliced = pred_var.sel(time=timestep)
            targ_sliced = ground_truth_var.sel(time=timestep)
            
            # MSE calculation
            mse = float(((pred_sliced - targ_sliced)**2).mean())

            # Store MSE result
            result_row = {
                'init_date': init_date.strftime('%Y-%m-%d %H:%M:%S'),
                'forecast_horizon_times': timestep,
                'forecast_step': step_idx + 1,
                'model': model_name,
                'mse': mse,
                'total_rainfall_pred': float(model_rainfall_totals[step_idx]),
                'total_rainfall_truth': float(ground_truth_rainfall_totals[step_idx])
            }
            all_results.append(result_row)

            # Store rainfall data for plotting
            rainfall_row = {
                'init_date': init_date.strftime('%Y-%m-%d %H:%M:%S'),
                'forecast_step': step_idx + 1,
                'model': model_name,
                'total_rainfall': float(model_rainfall_totals[step_idx])
            }
            all_rainfall_totals.append(rainfall_row)

            pd.DataFrame([result_row]).to_csv('skill_score.csv', mode='a', index=False)

            logging.debug(f"Result: {init_date}, {model_name}, {step_idx+1}-step MSE = {mse:.6f}, Total Rainfall = {model_rainfall_totals[step_idx]:.2f}")

    # 7. Create spatial plots for selected forecast horizons (every 7 steps = daily)
    if init_idx % 5 == 0:  # Create plots for every 5th initialization to save space
        for step in [3, 6, 12, 18, 27]:  # Selected forecast steps
            if step < len(times):
                timestep = times[step]
                targ_sliced = ground_truth_var.sel(time=timestep)
                
                # Get predictions for this timestep
                preds_for_plotting = {}
                for model_name, predictions in models_to_eval.items():
                    if eval_vars in predictions:
                        pred_var = predictions[eval_vars]
                    else:
                        pred_var = predictions
                    preds_for_plotting[model_name] = pred_var.sel(time=timestep)
                
                # Create plots
                step_plots_dir = f"{plots_dir}/init_{init_date.strftime('%Y%m%d')}"
                create_rainfall_plots(targ_sliced, preds_for_plotting, init_date, step + 1, step_plots_dir)

# 8. Save all results to CSV files
logging.info("Evaluation loop finished. Saving results to CSV.")
results_df = pd.DataFrame(all_results)
rainfall_df = pd.DataFrame(all_rainfall_totals)

# Add ground truth rainfall to rainfall dataframe
ground_truth_rainfall_df = results_df.groupby(['init_date', 'forecast_step']).agg({
    'total_rainfall_truth': 'first'
}).reset_index()
ground_truth_rainfall_df['model'] = 'Ground_Truth'
ground_truth_rainfall_df['total_rainfall'] = ground_truth_rainfall_df['total_rainfall_truth']
ground_truth_rainfall_df = ground_truth_rainfall_df[['init_date', 'forecast_step', 'model', 'total_rainfall']]

rainfall_combined_df = pd.concat([rainfall_df, ground_truth_rainfall_df], ignore_index=True)

output_dir = os.path.dirname(args.output_csv_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(args.output_csv_path, index=False)
rainfall_combined_df.to_csv(args.output_csv_path.replace('.csv', '_rainfall.csv'), index=False)

# 9. Create rainfall and MSE comparison plots
logging.info("Creating summary plots...")

# Plot MSE vs forecast step for all models
plt.figure(figsize=(12, 8))
mse_summary = results_df.groupby(['forecast_step', 'model']).agg({'mse': 'mean'}).reset_index()

for model in mse_summary['model'].unique():
    model_data = mse_summary[mse_summary['model'] == model]
    plt.plot(model_data['forecast_step'], model_data['mse'], marker='o', label=model, linewidth=2)

plt.xlabel('Forecast Step (6-hourly)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('MSE vs Forecast Lead Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./evaluation_results/mse_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot total rainfall vs forecast step for all models
plt.figure(figsize=(12, 8))
rainfall_summary = rainfall_combined_df.groupby(['forecast_step', 'model']).agg({'total_rainfall': 'mean'}).reset_index()

for model in rainfall_summary['model'].unique():
    model_data = rainfall_summary[rainfall_summary['model'] == model]
    linestyle = '--' if model == 'Ground_Truth' else '-'
    linewidth = 3 if model == 'Ground_Truth' else 2
    plt.plot(model_data['forecast_step'], model_data['total_rainfall'], 
             marker='o', label=model, linestyle=linestyle, linewidth=linewidth)

plt.xlabel('Forecast Step (6-hourly)', fontsize=12)
plt.ylabel('Total Rainfall over India (mm)', fontsize=12)
plt.title('Total Rainfall vs Forecast Lead Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./evaluation_results/rainfall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Combined plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot MSE on left y-axis
color = 'tab:red'
ax1.set_xlabel('Forecast Step (6-hourly)', fontsize=12)
ax1.set_ylabel('Mean Squared Error', color=color, fontsize=12)
for model in mse_summary['model'].unique():
    model_data = mse_summary[mse_summary['model'] == model]
    ax1.plot(model_data['forecast_step'], model_data['mse'], 
             marker='o', label=f'{model} (MSE)', color=color, alpha=0.7, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

# Plot rainfall on right y-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Total Rainfall over India (mm)', color=color, fontsize=12)
for model in rainfall_summary['model'].unique():
    model_data = rainfall_summary[rainfall_summary['model'] == model]
    linestyle = '-' if model == 'Ground_Truth' else ':'
    linewidth = 3 if model == 'Ground_Truth' else 2
    ax2.plot(model_data['forecast_step'], model_data['total_rainfall'], 
             marker='s', label=f'{model} (Rainfall)', color=color, alpha=0.8, 
             linestyle=linestyle, linewidth=linewidth)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('MSE and Total Rainfall vs Forecast Lead Time', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('./evaluation_results/combined_mse_rainfall.png', dpi=300, bbox_inches='tight')
plt.close()

logging.info(f"Evaluation complete. Results saved to {args.output_csv_path}")
logging.info("Summary plots saved in ./evaluation_results/")
print("\n--- Sample of Evaluation Results ---")
print(results_df.head())
print("\n--- Sample of Rainfall Results ---")
print(rainfall_combined_df.head())
print("------------------------------------\n")