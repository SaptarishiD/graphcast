# <get_predictions.py>
"""
python /home/saptarishi.dhanuka_asp25/weather/graphcast_dir/graphcast/local_files/eval_forecast.py \ 
--eval_start "2024-08-01" \ 
--eval_end "2024-09-15" \ 
--eval_dataset_choice "imerg" \ 
--vars_to_eval "total_precipitation_6hr" \ 
--params_path_new1 "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2024-06-01_2024-09-15_FORECAST28.npz" \ 
--params_path_new2 "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/finetuned/graphcast_1_13_orig_2024-06-01_2024-07-30_FORECAST28_ppt_weight100_indiaFalse.npz" \
--output_csv_path "./evaluation_results/forecast_mse.csv"
"""

"""
Runs model for each init date and save preds
"""
chosen_year = 2025
eval_start = "2024-06-01"
eval_end = "2024-06-30"
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
region_vise = True
regions = ['Central_Northeast', 'Hilly_Regions', 'Northeast', 'Northwest', 'South_Peninsular', 'West_Central']
world_regions = ['India']

"""
Complete evaluation of forecast against different datasets with rainfall analysis
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import sys
import logging
import argparse
import dataclasses
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
import time
import zarr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

import jax
import optax


# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.50'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# import utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
from graphcast import checkpoint, data_utils, rollout, graphcast, normalization
import setup_jax_functions
from plotting import scale, select, plot_data, save_animation, save_static_plot, compute_difference_with_targets_sims, plot_sample_from_ds
from metrics import compute_rmse, compute_mae, compute_bias, compute_acc
from utils import regrid_hres_fine_to_coarse, generate_sample_era5_dataset, grads_fn, parse_args, process_to_graphcast_format, compute_mse, compute_mse_diffs, mask_dbase_india_buffer, mask_dbase_regions

sys.path.append('/home/saptarishi.dhanuka_asp25/weather/graphcast_dir/gc_dist')
import trainer.dataloader
from dist_utils import construct_era5_imerg, construct_era5_imerg_6hourly
from datetime import datetime

print("Imports done")

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser(description="Evaluate GraphCast forecasts against datasets like ERA5/IMERG")

# Evaluation range
parser.add_argument('--eval_start', type=str, required=True,
                    help='Start date for evaluation period (format: YYYY-MM-DD)')
parser.add_argument('--eval_end', type=str, required=True,
                    help='End date for evaluation period (format: YYYY-MM-DD)')

# Dataset choice
parser.add_argument('--eval_dataset_choice', type=str, choices=['era5', 'imerg', 'imd', 'stations'], default='era5',
                    help='Dataset choice for evaluation')

parser.add_argument('--vars_to_eval', type=str, required=True, help='Which variables to evaluate')

# Data paths
parser.add_argument('--eval_data_path', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/',
                    help='Path to Eval dataset')

parser.add_argument('--params_path_old', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/origs/graphcast_1_13.npz',
                    help='Path to original GraphCast parameters (.npz)')

parser.add_argument('--params_path_new1', type=str, default=None,
                    help='Path to fine-tuned GraphCast parameters (.npz)')


parser.add_argument('--params_path_new2', type=str, default=None,
                    help='Path to fine-tuned GraphCast parameters (.npz)')


parser.add_argument('--norms_dir', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/norms_gc/',
                    help='Directory containing normalization .nc files')

# Output paths
parser.add_argument('--output_pred_old_dir', type=str,
                    default='/Datastorage/saptarishi.dhanuka_asp25/preds_dir/',
                    help='Output path for original GraphCast predictions')
parser.add_argument('--output_pred_finetuned_dir', type=str,
                    default='/Datastorage/saptarishi.dhanuka_asp25/preds_dir/',
                    help='Output path for fine-tuned GraphCast predictions')
parser.add_argument('--plots_dir', type=str, default='plots/evals',
                    help='Directory to save eval plots')

parser.add_argument('--plot_timesteps', type=int, default=4,
                    help='Number of timesteps to generate plots for')

# Spatial extent
parser.add_argument('--latmin', type=float, default=6, help='Minimum latitude for plotting')
parser.add_argument('--latmax', type=float, default=38, help='Maximum latitude for plotting')
parser.add_argument('--lonmin', type=float, default=68, help='Minimum longitude for plotting')
parser.add_argument('--lonmax', type=float, default=98, help='Maximum longitude for plotting')
parser.add_argument('--chosen_year', type=int, default=None, required=True, help='Eval year')

parser.add_argument('--real_time', type=str, default=None, help='Eval year')
parser.add_argument('--resolution', type=str, default=None, help='Eval year')
parser.add_argument('--hour6', type=bool, default=None, help='Eval year')


parser.add_argument(
"--params_paths",
nargs="+",   # allows multiple values
type=str,
required=True,
help="List of new model parameter paths to evaluate."
)

args = parser.parse_args()

# Add new argument to the parser in utils.py
# For now, let's handle it here if it's not in the original file
if not hasattr(args, 'output_csv_path'):
    args.output_csv_path = "./forecast_evaluation_mse.csv"

eval_start = args.eval_start
eval_end = args.eval_end
dataset_choice = args.eval_dataset_choice
eval_vars = args.vars_to_eval
apath = args.eval_data_path
params_path_old = args.params_path_old
params_path_new1 = args.params_path_new1
params_path_new2 = args.params_path_new2
chosen_year = args.chosen_year
resolution = args.resolution
hour6 = args.hour6

# --- Data and Model Loading ---
logging.info(f"Loading dataset '{dataset_choice}' from {eval_start} to {eval_end}")
if dataset_choice == "imerg":

    if resolution == '0.25':
        print(f"Not using IMERG for the 0.25 data for now")
        apath = "/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_025_cache/"
        dbase,_, _ = trainer.dataloader.open_databases(apath,None)


    else:
        apath = '/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/'
        dbase,_, _ = trainer.dataloader.open_databases(apath,None)
        if hour6:
            print('6 hourly truths')
            dbase = construct_era5_imerg_6hourly(dbase, year=chosen_year, save=True)
        else:
            if args.real_time != 'True':
                dbase = construct_era5_imerg(dbase, year=chosen_year, save=True)
    # Load a slightly larger window to ensure we have the day before the start_date for initialization
else:
    print("Not using IMERG")
    dbase,_, _ = trainer.dataloader.open_databases(apath,None)

load_start_date = pd.to_datetime(eval_start) - pd.Timedelta(days=1)
eval_time_ds = dbase.sel(time=slice(load_start_date.strftime('%Y-%m-%d'), eval_end))
del dbase

select_time_eval = process_to_graphcast_format(eval_time_ds)
logging.info(f"Full data range loaded: {select_time_eval.time.values[0]} to {select_time_eval.time.values[-1]}")


# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.08'

logging.info("Loading models and normalization stats...")
with open(params_path_old, 'rb') as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params

if params_path_new1:
    with open(params_path_new1, "rb") as f:
        new_params1 = checkpoint.load(f, graphcast.CheckPoint).params
if params_path_new2:
    with open(params_path_new2, "rb") as f:
        new_params2 = checkpoint.load(f, graphcast.CheckPoint).params

with open('/Datastorage/saptarishi.dhanuka_asp25/norms_gc/diffs_stddev_by_level.nc', 'rb') as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()
with open('/Datastorage/saptarishi.dhanuka_asp25/norms_gc/stddev_by_level.nc', 'rb') as f:
    stddev_by_level = xr.load_dataset(f).compute()
with open('/Datastorage/saptarishi.dhanuka_asp25/norms_gc/mean_by_level.nc', 'rb') as f:
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


all_results = []
initialization_dates = pd.to_datetime(pd.date_range(start=eval_start, end=eval_end, freq='D'))

# Max forecast length is 7 days (28 steps of 6 hours)
MAX_FORECAST_STEPS = 28
target_lead_times_str = f"{(MAX_FORECAST_STEPS) * 6}h" # +1 to be safe with slicing
target_lead_times_slice = slice("6h", target_lead_times_str)
forecast_horizons_def = {1: 4, 3: 12, 7: 28} # In 6-hourly steps


current_date = datetime.now().strftime('%Y-%m-%d%H-%M-%S')

logging.info(f"Starting evaluation for {len(initialization_dates)} initialization dates.")
all_results = []
idx = 0
for init_date in tqdm(initialization_dates, desc="Evaluating Forecasts", leave=True):
    logging.info(f"Processing initialization date: {init_date}")

    # 1. Load data for this initialization
    try:
        # Graphcast requires two time steps for input: the init time and 6 hours prior
        start_slice = init_date - pd.Timedelta(hours=6)
        end_slice = init_date + pd.Timedelta(days=7)

        start_slice -= select_time_eval.datetime.values[0][0]
        end_slice -= select_time_eval.datetime.values[0][0]

        # Use .copy(deep=True) to avoid memory issues with repeated slicing
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
    
    # print("Eval Inputs:   ", eval_inputs.dims.mapping)
    # print("Eval Targets:  ", eval_targets.dims.mapping)
    # print("Eval Forcings: ", eval_forcings.dims.mapping)

    if (eval_inputs.dims.mapping['time'] != 2):
        print("\nError in extracting inputs targets and forcings\n")
        continue

    # Ensure we have enough target data for the longest forecast
    if eval_targets.sizes['time'] < MAX_FORECAST_STEPS:
        logging.warning(f"Not enough target data for a 7-day forecast from {init_date}. Have {eval_targets.sizes['time']} steps. Skipping.")
        continue

    targets_template = eval_targets * np.nan
    ground_truth_var = eval_targets[eval_vars]

    # 3. Run all Graphcast models
    logging.debug(f"Running Graphcast models for {init_date}")

    # print("Base run")
    # predictions_base = run_model(params, state, eval_inputs, targets_template, eval_forcings)
    
    # print("Finetuned 2")
    # predictions_ft2 = run_model(new_params2, state, eval_inputs, targets_template, eval_forcings)

    target_path = f'/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/target_{dataset_choice}_init_6h_precip_{init_date}.nc'
    if not os.path.exists(target_path):
        eval_targets['total_precipitation_6hr'].to_netcdf(target_path)

    # if not os.path.exists(f'/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/025res/base_precip025_init_{init_date}.nc'):
    #     predictions_base['total_precipitation_6hr'].to_netcdf(f'/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/025res/base_precip025_init_{init_date}.nc')

    # if not os.path.exists(f'/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/fine_{params_path_new1.split("/")[-1].split(".")[0]}_6h_init_{init_date}.nc'):
    #     print("Finetuned 1")
    #     predictions_ft1 = run_model(new_params1, state, eval_inputs, targets_template, eval_forcings)
    #     predictions_ft1['total_precipitation_6hr'].to_netcdf(f'/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/fine_{params_path_new1.split("/")[-1].split(".")[0]}_6h_init_{init_date}.nc')


    # predictions_ft1['total_precipitation_6hr'].to_netcdf(f'/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/fine_{params_path_new1.split("/")[-1].split(".")[0]}_6h_init_{init_date}.nc')
    # predictions_ft2.to_netcdf(f'/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/fine_init_{init_date}.nc')

    del eval_sim_data
    del eval_inputs
    del eval_targets
    del eval_forcings
    del targets_template
    del ground_truth_var
    if 'predictions_base' in locals():
        del predictions_base
    if 'predictions_ft1' in locals():
        del predictions_ft1
    if 'predictions_ft2' in locals():
        del predictions_ft2
    if 'eval_targets' in locals():
        del eval_targets
    
    gc.collect() # Manually trigger garbage collection

# </get_predictions.py>