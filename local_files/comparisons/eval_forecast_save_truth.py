# <eval_forecast_save.py>
"""
python /home/saptarishi.dhanuka_asp25/weather/graphcast_dir/graphcast/local_files/eval_forecast_save.py \ 
--eval_start "2014-08-01" \ 
--eval_end "2014-09-30" \ 
--eval_dataset_choice "imerg" \ 
--vars_to_eval "total_precipitation_6hr" \ 
--params_path_new1 "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_mask_expt5.npz" \ 
--params_path_new2 "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_mask_expt3.npz" \
--output_csv_path "./evaluation_results/forecast_mse.csv"
"""

"""
Complete evaluation of forecast against different datasets
"""
# <eval_forecast.py>
# Parameters
eval_start = "2014-08-01"
eval_end = "2014-09-05"
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
import glob

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
eval_vars = "total_precipitation_6hr"
world_regions = ['India']
target_files_pattern = "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds/target_init_2014-*.nc"
current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Storage for results
all_precip_results = []


def extract_precipitation_data():
    """Extract precipitation data from target files"""
    
    # Find all target files for 2014
    target_files = sorted(glob.glob(target_files_pattern))
    logging.info(f"Found {len(target_files)} target files for 2014")
    
    if not target_files:
        logging.error(f"No target files found matching pattern: {target_files_pattern}")
        return
    
    for target_file in tqdm(target_files, desc="Extracting Precipitation Data"):
        logging.info(f"Processing target file: {target_file}")
        
        # Extract initialization date from filename
        try:
            filename = os.path.basename(target_file)
            init_date_str = filename.replace("target_init_", "").replace(".nc", "")
            init_date = pd.to_datetime(init_date_str)
        except Exception as e:
            logging.warning(f"Could not parse date from filename {target_file}: {e}. Skipping.")
            continue
        
        # Load eval targets from file
        try:
            eval_targets = xr.open_dataset(target_file)
            logging.info(f"Loaded targets with dimensions: {eval_targets.dims}")
            
            # Check if we have the required variable
            if eval_vars not in eval_targets.data_vars:
                logging.warning(f"Variable {eval_vars} not found in {target_file}. Available vars: {list(eval_targets.data_vars.keys())}. Skipping.")
                continue
            
            # Extract ground truth precipitation data
            ground_truth_var = eval_targets[eval_vars]
            
            # Remove batch dimension if present
            if 'batch' in ground_truth_var.dims:
                ground_truth_var = ground_truth_var.squeeze('batch')
            
            logging.info(f"Ground truth variable shape: {ground_truth_var.shape}")
            
        except Exception as e:
            logging.warning(f"Could not load target file {target_file}: {e}. Skipping.")
            continue
        
        # Extract precipitation values at each time step
        times = ground_truth_var.time.values
        
        for timestep in times:
            # Slice ground truth to the current forecast horizon
            truth_sliced = ground_truth_var.sel(time=timestep)
            
            # Loop over each region
            for region in world_regions:
                # Apply region mask
                truth_region = mask_dbase_india_buffer(truth_sliced)
                
                # Compute statistics over the region
                mean_precip = float(truth_region.mean())
                max_precip = float(truth_region.max())
                min_precip = float(truth_region.min())
                std_precip = float(truth_region.std())
                total_precip_sum = float(truth_region.sum(skipna=True))

                
                # Store precipitation result
                result_row = {
                    'init_date': init_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'forecast_horizon': str(timestep),
                    'model': 'Ground_Truth',
                    'region': region,
                    'mean_precipitation': mean_precip,
                    'max_precipitation': max_precip,
                    'min_precipitation': min_precip,
                    'std_precipitation': std_precip,
                    'total_precip_sum': total_precip_sum
                }
                all_precip_results.append(result_row)
                
                # Write individual CSV per region
                csv_filename = f'precipitation_data_{region}_{current_date}.csv'
                pd.DataFrame([result_row]).to_csv(
                    csv_filename,
                    mode='a',
                    header=not os.path.exists(csv_filename),
                    index=False
                )
                
                logging.debug(f"[{region}] {init_date}, Ground Truth, {timestep} Mean Precip = {mean_precip:.6f}")
                
                # Save the full precipitation xarray for later inspection
                precip_ds = truth_region.to_dataset(name='precipitation')
                precip_ds.attrs.update({
                    'init_date': init_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'model': 'Ground_Truth',
                    'region': region,
                    'forecast_horizon': str(timestep)
                })
                nc_filename = (
                    f'precip_{region}_Ground_Truth_'
                    f"{init_date.strftime('%Y%m%d%H')}_t{timestep}.nc"
                )
                precip_ds.to_netcdf(nc_filename)
                logging.debug(f"Saved precipitation xarray to {nc_filename}")
        
        # Close the dataset to free memory
        eval_targets.close()

def save_results():
    """Save all precipitation results to CSV"""
    if not all_precip_results:
        logging.warning("No precipitation results to save.")
        return
    
    logging.info("Saving all precipitation results to CSV.")
    precip_results_df = pd.DataFrame(all_precip_results)
    
    # Save consolidated results
    output_path = f'precipitation_results_consolidated_{current_date}.csv'
    precip_results_df.to_csv(output_path, index=False)
    
    logging.info(f"Precipitation data extraction complete. Results saved to {output_path}")
    print("\n--- Sample of Precipitation Data Results ---")
    print(precip_results_df.head())
    print(f"Total records: {len(precip_results_df)}")
    print("--------------------------------------------\n")

if __name__ == "__main__":
    logging.info("Starting precipitation data extraction...")
    extract_precipitation_data()
    save_results()
    logging.info("Process completed successfully.")