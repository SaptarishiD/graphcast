# <eval_forecast_save.py>


"""
Complete evaluation of forecast against different datasets
"""
year_chosen = 2014
# <eval_forecast.py>
# Parameters
eval_start = f"{year_chosen}-08-01"
eval_end = f"{year_chosen}-09-05"
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


# First, add the ACC computation function (adapted from eval_acc.py)
def compute_precipitation_acc_multi_init(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    climatology_precip: xr.DataArray,
    init_date: pd.Timestamp
) -> xr.DataArray:
    """
    Computes latitude-weighted ACC for precipitation for a single initialization.
    
    Args:
        predictions: xarray Dataset with predicted 'total_precipitation_6hr'
        targets: xarray Dataset with target 'total_precipitation_6hr'  
        climatology_precip: xarray DataArray of precipitation climatology
        init_date: pandas Timestamp of initialization date
        
    Returns:
        xarray DataArray containing ACC values for each forecast time step
    """
    # Select the variable and align data structures
    if 'total_precipitation_6hr' in predictions.data_vars:
        pred_pr = predictions['total_precipitation_6hr']
    else:
        pred_pr = predictions
    
    if 'total_precipitation_6hr' in targets:
        targ_pr = targets['total_precipitation_6hr']
    else:
        targ_pr = targets
    
    # Remove batch dimension if present
    if 'batch' in pred_pr.dims:
        pred_pr = pred_pr.squeeze('batch')
    if 'batch' in targ_pr.dims:
        targ_pr = targ_pr.squeeze('batch')
    
    # Ensure proper dimension order
    if 'time' in targ_pr.dims and 'lat' in targ_pr.dims and 'lon' in targ_pr.dims:
        targ_pr = targ_pr.transpose('time', 'lat', 'lon')
    
    # Create absolute time coordinates based on initialization date
    time_deltas = pred_pr['time'].to_pandas()
    valid_times = init_date + time_deltas
    
    # Derive dayofyear and hour using .dt accessor
    dayofyear_vals = xr.DataArray(valid_times.dt.dayofyear, coords={'time': pred_pr.time})
    hour_vals = xr.DataArray(valid_times.dt.hour, coords={'time': pred_pr.time})
    
    # Prepare climatology: rename and interpolate
    climatology_renamed = climatology_precip.rename({'latitude': 'lat', 'longitude': 'lon'})
    climatology_interp = climatology_renamed.sel(
        dayofyear=dayofyear_vals,
        hour=hour_vals
    ).interp_like(pred_pr)
    
    # Compute anomalies
    pred_anomaly = pred_pr - climatology_interp
    targ_anomaly = targ_pr - climatology_interp
    
    # Compute latitude weights
    weights = np.cos(np.deg2rad(pred_anomaly['lat']))
    weights.name = "weights"
    
    # Calculate ACC for each time step
    acc_list = []
    for t in pred_anomaly.time:
        pred_anom_t = pred_anomaly.sel(time=t)
        targ_anom_t = targ_anomaly.sel(time=t)
        
        # Compute weighted means
        pred_mean = pred_anom_t.weighted(weights).mean(dim=['lat', 'lon'])
        targ_mean = targ_anom_t.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Compute deviations from means
        pred_dev = pred_anom_t - pred_mean
        targ_dev = targ_anom_t - targ_mean
        
        # Calculate ACC components
        numerator = (weights * pred_dev * targ_dev).sum(dim=['lat', 'lon'])
        denominator_pred = (weights * pred_dev**2).sum(dim=['lat', 'lon'])
        denominator_targ = (weights * targ_dev**2).sum(dim=['lat', 'lon'])
        
        acc_t = numerator / np.sqrt(denominator_pred * denominator_targ)
        acc_list.append(acc_t)
    
    # Combine into single DataArray
    acc = xr.concat(acc_list, dim='time')
    acc = acc.assign_coords(time=pred_anomaly.time)
    
    return acc


def compute_precipitation_acc_debugged(
    predictions_finetuned: xr.Dataset,
    targets: xr.Dataset,
    climatology_precip: xr.DataArray
) -> xr.DataArray:
    """
    Computes a latitude-weighted Anomaly Correlation Coefficient (ACC) for precipitation.

    This function is debugged to correctly use the .dt accessor for pandas
    datetime properties. It handles timedelta conversion, aligns dataset
    structures, interpolates climatology, and applies latitude weighting.

    Args:
        predictions_finetuned: An xarray Dataset containing the predicted 'total_precipitation_6hr'.
        targets: An xarray Dataset containing the target 'total_precipitation_6hr'.
        climatology_precip: An xarray DataArray of precipitation climatology.

    Returns:
        An xarray DataArray containing the ACC value for each forecast time step.
    """
    # 1. Select the variable and align data structures
    if isinstance(predictions_finetuned, xr.Dataset):
        pred_pr = predictions_finetuned['total_precipitation_6hr'].squeeze('batch')
        targ_pr = targets['total_precipitation_6hr'].squeeze('batch').transpose('time', 'lat', 'lon')

    

    # 2. Create absolute time coordinates
    start_time = pd.to_datetime(f'{year_chosen}-08-01 00:00:00')
    # Convert the 'time' coordinate (timedelta) to a pandas Series for easy addition
    time_deltas = pred_pr['time'].to_pandas()
    valid_times = start_time + time_deltas

    # 3. Correctly derive dayofyear and hour using the .dt accessor
    # This is the fix for the AttributeError.
    dayofyear_vals = xr.DataArray(valid_times.dt.dayofyear, coords={'time': pred_pr.time})
    hour_vals = xr.DataArray(valid_times.dt.hour, coords={'time': pred_pr.time})

    # 4. Prepare climatology: rename and interpolate
    climatology_renamed = climatology_precip.rename({'latitude': 'lat', 'longitude': 'lon'})
    climatology_interp = climatology_renamed.sel(
        dayofyear=dayofyear_vals,
        hour=hour_vals
    ).interp_like(pred_pr)

    # 5. Compute anomalies
    pred_pr = pred_pr.copy(data=np.asarray(pred_pr.data))
    targ_pr = targ_pr.copy(data=np.asarray(targ_pr.data))
    
    pred_anomaly = pred_pr - climatology_interp
    targ_anomaly = targ_pr - climatology_interp

    # 6. Compute latitude weights
    weights = np.cos(np.deg2rad(pred_anomaly['lat']))
    weights.name = "weights"

    # 7. Calculate the weighted Anomaly Correlation Coefficient
    pred_anomaly_mean = pred_anomaly.weighted(weights).mean(dim=['lat', 'lon'])
    targ_anomaly_mean = targ_anomaly.weighted(weights).mean(dim=['lat', 'lon'])

    pred_anomaly_dev = pred_anomaly - pred_anomaly_mean
    targ_anomaly_dev = targ_anomaly - targ_anomaly_mean

    numerator = (weights * pred_anomaly_dev * targ_anomaly_dev).sum(dim=['lat', 'lon'])
    denominator_pred_sq = (weights * pred_anomaly_dev**2).sum(dim=['lat', 'lon'])
    denominator_targ_sq = (weights * targ_anomaly_dev**2).sum(dim=['lat', 'lon'])
    
    acc = numerator / np.sqrt(denominator_pred_sq * denominator_targ_sq)
    # acc.name = "anomaly_correlation_coefficient"

    return acc





# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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
from dist_utils import construct_era5_imerg
from datetime import datetime

print("Imports done")

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
args = parse_args()

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
year_chosen = args.chosen_year

# --- Data and Model Loading ---
logging.info(f"Loading dataset '{dataset_choice}' from {eval_start} to {eval_end}")
if dataset_choice == "imerg":
    apath = '/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/'
    dbase,_ = trainer.dataloader.open_databases(apath,None)
    dbase = construct_era5_imerg(dbase, year=year_chosen)
    # Load a slightly larger window to ensure we have the day before the start_date for initialization
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








# --- HRES Data Handling ---
import glob
def extract_hres_date(path):
    parts = path.split('_')
    # print(parts)
    year = args.chosen_year  # fixed, or extract if you want dynamic
    month = int(parts[5])
    day = int(parts[6])
    return datetime(year, month, day)

logging.info("Mapping HRES forecast files...")
hres_file_list = glob.glob(f"/Datastorage/saptarishi.dhanuka_asp25/forecasts_hres/raw_hres/{year_chosen}/*.grib")
hres_files_map = {extract_hres_date(p): p for p in hres_file_list}

# =============================================================================
# === NEW SYSTEMATIC EVALUATION AND VERIFICATION PROCEDURE ====================
# =============================================================================
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
clim_ppt = xr.open_zarr('/Datastorage/saptarishi.dhanuka_asp25/era5_data/precip_climatology.zarr')
clim_region = clim_ppt.sel(latitude=slice(37, 6), longitude=slice(65, 95))

for init_date in tqdm(initialization_dates, desc="Evaluating Forecasts"):
    logging.info(f"Processing initialization date: {init_date}")

    # 1. Load data for this initialization
    try:
        # Graphcast requires two time steps for input: the init time and 6 hours prior
        start_slice = init_date - pd.Timedelta(hours=6)
        end_slice = init_date + pd.Timedelta(days=7)

        start_slice -= select_time_eval.datetime.values[0][0]
        end_slice -= select_time_eval.datetime.values[0][0]

        # Use .copy(deep=True) to avoid memory issues with repeated slicing
        eval_sim_data = select_time_eval.sel(time=slice(start_slice, end_slice)).copy(deep=True)

        if eval_sim_data.sizes["time"] < 2:
            logging.warning(f"Not enough data for initialization {init_date}. Found {eval_sim_data.sizes['time']} steps. Skipping.")
            continue
    except Exception as e:
        logging.warning(f"Could not load data for {init_date}: {e}. Skipping.")
        continue


    from dask.diagnostics import ProgressBar
    with ProgressBar():
        eval_sim_data = select_time_eval.sel(time=slice(start_slice, end_slice)).compute()

    # 2. Prepare inputs, targets, and forcings for a 7-day rollout
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        eval_sim_data, target_lead_times=target_lead_times_slice, **dataclasses.asdict(task_config))
    
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)

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
    print("Base run")
    predictions_base = run_model(params, state, eval_inputs, targets_template, eval_forcings)
    print("Finetuned 1")
    predictions_ft1 = run_model(new_params1, state, eval_inputs, targets_template, eval_forcings)
    print("Finetuned 2")
    predictions_ft2 = run_model(new_params2, state, eval_inputs, targets_template, eval_forcings)
    
    models_to_eval = {
        f'Graphcast_Base_{year_chosen}': predictions_base,
        f'Graphcast_Finetuned1_{year_chosen}': predictions_ft1,
        f'Graphcast_Finetuned2_{year_chosen}': predictions_ft2,
    }

    # 4. Load, regrid, and align HRES forecast for the same initialization date
    hres_predictions = None
    hres_file_path = hres_files_map.get(init_date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0))
    if hres_file_path:
        try:
            logging.debug(f"Processing HRES file: {hres_file_path}")
            hres_ds = xr.open_dataset(hres_file_path, engine='cfgrib')
            hres_regridded = regrid_hres_fine_to_coarse(hres_ds, variable='tp', coarse_resolution=1.0)
            # Align time dimension with Graphcast targets
            hres_predictions = hres_regridded.drop_vars({'time'}).rename({'step': 'time'}).isel(time=slice(1,None)).assign_coords(time=eval_targets.time)
            models_to_eval['HRES'] = hres_predictions
        except Exception as e:
            logging.warning(f"Failed to process HRES file {hres_file_path}: {e}")
    else:
        logging.warning(f"No HRES file found for init date {init_date}")



    for model_name, predictions in tqdm(models_to_eval.items(), desc="ACC Calc"):

        if model_name == 'HRES':
            continue
        # if eval_vars in predictions:
        #     pred_var = predictions[eval_vars]
        # else:
        #     pred_var = predictions

        pred_var = predictions

        # Get corresponding targets for this model's forecast length
        target_var = ground_truth_var.isel(time=slice(0, pred_var.sizes['time']))
        targs = eval_targets.isel(time=slice(0, pred_var.sizes['time']))
        
        # *** CHANGE: loop over each region
        for region in tqdm(world_regions, desc=f"World regions for {model_name}"):
            
            # Apply region mask to predictions and targets
            # pred_region = mask_dbase_india_buffer(pred_var)
            # targ_region = mask_dbase_india_buffer(target_var)
            
            # Select climatology for the region (adjust slicing as needed)
            assert clim_region.latitude.values[0] > clim_region.latitude.values[1], "Climatology latitude values are not in asc order"
            assert clim_region.longitude.values[0] < clim_region.longitude.values[1], "Climatology longitude values are not in desc order"


            # check if predictions is dataset and if not print what it is and print model name init date etc
            if not isinstance(predictions, xr.Dataset):
                logging.error(f"Predictions for {model_name} on {init_date} is not an xarray Dataset. It is: {type(predictions)}")
                assert isinstance(predictions, xr.Dataset)
                


            assert isinstance(eval_targets, xr.Dataset)
        

            acc = compute_precipitation_acc_debugged(predictions.sel(lat=slice(6, 37), lon=slice(65, 95)), eval_targets.sel(lat=slice(6, 37), lon=slice(65, 95)), clim_ppt.sel(latitude=slice(37, 6), longitude=slice(65, 95)))



            accvals = acc['total_precipitation_6hr']
            lead_hours = (acc["time"].astype("timedelta64[h]").astype(int)).values / 3600

            for accval, lead_hour in zip(accvals.values, lead_hours):

                result_row = {
                    'init_date': init_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'forecast_horizon': str(lead_hour),
                    'model': model_name,
                    'region': region,
                    'acc': accval
                }
                all_results.append(result_row)

                csv_filename = f'skill_score_{region}_{current_date}_ACC.csv'
                pd.DataFrame([result_row]).to_csv(
                    csv_filename,
                    mode='a',
                    header=not os.path.exists(csv_filename),
                    index=False
                )
                # print((f"[{region}] {init_date}, {model_name}, {lead_hour} ACC = {accval:.6f}"))
                logging.debug(f"[{region}] {init_date}, {model_name}, {lead_hour} ACC = {accval:.6f}")


# 6. Save all results to a CSV file
logging.info("Evaluation loop finished. Saving results to CSV.")
results_df = pd.DataFrame(all_results)

output_dir = os.path.dirname(args.output_csv_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(args.output_csv_path, index=False)
logging.info(f"Evaluation complete. Results saved to skill_score.csv and args csv")
print("\n--- Sample of Evaluation Results ---")
print(results_df.head())
print("------------------------------------\n")

# </eval_forecast_save.py>