import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import timedelta
from graphcast import checkpoint, data_utils, graphcast
import setup_jax_functions
import jax
import dataclasses

# ───────────────────────────────────────────────────────────────────────────────
# USER-DEFINABLE PARAMETERS
# ───────────────────────────────────────────────────────────────────────────────
ERA5_ZARR_PATH = "/Datastorage/saptarishi.dhanuka_asp25/era5_data/wb_era5_jan2016_temp_ppt.zarr/"
CKPT_BASE_PATH     = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/origs/graphcast_1_13.npz'
CKPT_FINETUNE_PATH = "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig.npz"


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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import jax

# Import required modules from graphcast
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from graphcast import checkpoint, data_utils, rollout, graphcast, normalization
import save_params_utils
import setup_jax_functions

# Configure JAX and environment
jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0' 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Define temperature zones for regional analysis
temperature_zones = {
    "northwest_india": {
        "lat_min": 24.0,
        "lat_max": 31.0,
        "lon_min": 69.0,
        "lon_max": 77.0,
    },
    "north_central_plains": {
        "lat_min": 23.0,
        "lat_max": 28.0,
        "lon_min": 77.0,
        "lon_max": 85.0,
    },
    "northeast_india": {
        "lat_min": 22.0,
        "lat_max": 28.0,
        "lon_min": 89.0,
        "lon_max": 97.0,
    },
    "western_coast": {
        "lat_min": 8.0,
        "lat_max": 20.0,
        "lon_min": 73.0,
        "lon_max": 76.0,
    },
    "eastern_coast": {
        "lat_min": 10.0,
        "lat_max": 21.0,
        "lon_min": 80.0,
        "lon_max": 86.0,
    },
    "deccan_plateau": {
        "lat_min": 13.0,
        "lat_max": 20.0,
        "lon_min": 74.0,
        "lon_max": 82.0,
    },
    "western_himalayas": {
        "lat_min": 30.0,
        "lat_max": 35.0,
        "lon_min": 75.0,
        "lon_max": 80.0,
    },
    "southern_peninsula": {
        "lat_min": 8.0,
        "lat_max": 13.0,
        "lon_min": 76.0,
        "lon_max": 80.0,
    },
}

# Set up logging
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_file = f"loggers/forecast_comparison_{current_date}.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
os.makedirs("plots/comparison", exist_ok=True)
os.makedirs("plots/comparison/regions", exist_ok=True)

logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

class WeatherForecastComparison:
    """
    Class to compare base and fine-tuned weather forecasting models.
    """
    def __init__(self, 
                base_model_path, 
                finetuned_model_path, 
                normalization_data_path,
                model_levels=13,
                model_resolution=1.0):
        """
        Initialize the comparison framework.
        
        Args:
            base_model_path: Path to base model checkpoint
            finetuned_model_path: Path to fine-tuned model checkpoint
            normalization_data_path: Path to normalization data
            model_levels: Number of pressure levels
            model_resolution: Model resolution
        """
        self.model_levels = model_levels
        self.model_resolution = model_resolution
        self.variables_to_compare = ['2m_temperature', 'total_precipitation_6hr']
        
        # Load normalization data
        self.mean_by_level = xr.load_dataset(f"{normalization_data_path}/mean_by_level.nc").compute()
        self.stddev_by_level = xr.load_dataset(f"{normalization_data_path}/stddev_by_level.nc").compute()
        self.diffs_stddev_by_level = xr.load_dataset(f"{normalization_data_path}/diffs_stddev_by_level.nc").compute()
        
        # Load base model
        with open(base_model_path, 'rb') as f:
            self.base_ckpt = checkpoint.load(f, graphcast.CheckPoint)
        self.base_params = self.base_ckpt.params
        
        # Load fine-tuned model
        with open(finetuned_model_path, 'rb') as f:
            self.finetuned_ckpt = checkpoint.load(f, graphcast.CheckPoint)
        self.finetuned_params = self.finetuned_ckpt.params
        
        # Set up model and task configuration
        self.model_config = self.base_ckpt.model_config
        self.task_config = self.base_ckpt.task_config
        
        # Initialize JAX functions
        self._setup_jax_functions()
        
        logger.info(f"Model configuration: {self.model_config}")
        logger.info(f"Task configuration: {self.task_config}")
    
    def _setup_jax_functions(self):
        """Set up JAX functions for model inference."""
        setup_jax_functions.update_configs({
            'params': self.base_params,
            'state': {},
            'model_config': self.model_config,
            'task_config': self.task_config,
            'mean_by_level': self.mean_by_level,
            'stddev_by_level': self.stddev_by_level,
            'diffs_stddev_by_level': self.diffs_stddev_by_level
        })
        
        self.run_forward_jitted = setup_jax_functions.drop_state(
            setup_jax_functions.with_params(
                jax.jit(setup_jax_functions.with_configs(setup_jax_functions.run_forward.apply))
            )
        )
    
    def run_model(self, params, inputs, targets_template, forcings):
        """
        Run a model with given parameters.
        
        Args:
            params: Model parameters
            inputs: Input data
            targets_template: Template for targets
            forcings: Forcing data
            
        Returns:
            Model predictions
        """
        predictions = self.run_forward_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            params=params,
            state={}
        )
        return predictions
    
    def load_era5_data(self, data_path):
        """
        Load ERA5 data.
        
        Args:
            data_path: Path to ERA5 data
            
        Returns:
            ERA5 dataset
        """
        era5_data = xr.open_zarr(data_path)
        
        # Ensure the data is properly formatted
        if 'latitude' in era5_data.coords and 'longitude' in era5_data.coords:
            old_lats = era5_data['latitude'].values
            old_lons = era5_data['longitude'].values
            new_lats = np.arange(-90.0, 90.0 + 1e-8, self.model_resolution)
            new_lats = np.flip(new_lats)
            new_lons = np.arange(0, 359.75 + 1e-8, self.model_resolution)
            
            era5_data = era5_data.interp(
                {'latitude': new_lats, 'longitude': new_lons}, 
                method='linear',
                kwargs={'fill_value': None}
            )
            
            era5_data = era5_data.expand_dims(batch=1)
            era5_data = era5_data.rename({'latitude': 'lat', 'longitude': 'lon'})
            
            # Handle time dimension
            datetime_array = era5_data['time'].values
            time_array = np.arange(0, len(datetime_array) * 21600000000000, 21600000000000, dtype='timedelta64[ns]')
            
            era5_data = era5_data.assign_coords(datetime=('time', time_array))
            
            temp_time = era5_data.coords["time"].copy()
            temp_datetime = era5_data.coords["datetime"].copy()
            
            era5_data = era5_data.assign_coords(
                time=temp_datetime,
                datetime=temp_time
            )
            
            old_datetime = era5_data["datetime"].values
            new_datetime = old_datetime[np.newaxis, :]
            era5_data = era5_data.assign_coords(datetime=(("batch", "time"), new_datetime))
        
        return era5_data
    
    def extract_simulation_periods(self, era5_data, forecast_days=7):
        """
        Extract periods for simulation from ERA5 data.
        
        Args:
            era5_data: ERA5 dataset
            forecast_days: Number of days to forecast (default: 7)
            
        Returns:
            List of time slices for simulation
        """
        forecast_steps = forecast_days * 4  # Assuming 6-hourly data (4 steps per day)
        total_time_steps = era5_data.dims['time']
        
        # We need enough time steps for initialization plus forecast
        simulation_periods = []
        
        # Check if we have at least one valid period
        if total_time_steps < forecast_steps:
            logger.error(f"ERA5 data has only {total_time_steps} time steps, which is insufficient for a {forecast_days}-day forecast.")
            return simulation_periods
        
        # Create time slices for each possible starting point
        max_start_idx = total_time_steps - forecast_steps
        for start_idx in range(0, max_start_idx, forecast_steps):
            end_idx = start_idx + forecast_steps
            simulation_periods.append((start_idx, end_idx))
        
        logger.info(f"Extracted {len(simulation_periods)} simulation periods from ERA5 data.")
        return simulation_periods
    
    def run_simulation(self, era5_data, start_idx, end_idx):
        """
        Run a single simulation for both models.
        
        Args:
            era5_data: ERA5 dataset
            start_idx: Starting time index
            end_idx: Ending time index
            
        Returns:
            Dictionary containing simulation results
        """
        # Extract data for this simulation period
        simulation_data = era5_data.isel(time=slice(start_idx, end_idx))

        simulation_data.load()
        
        # Extract inputs, targets, and forcings
        task_config_dict = dataclasses.asdict(self.task_config)
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            simulation_data, 
            target_lead_times=slice("6h", f"{6*end_idx-start_idx}h"),
            **task_config_dict
        )
        
        # Create a template for targets
        targets_template = targets * np.nan
        
        # Run base model
        base_predictions = self.run_model(
            self.base_params, 
            inputs, 
            targets_template, 
            forcings
        )
        
        # Run fine-tuned model
        finetuned_predictions = self.run_model(
            self.finetuned_params, 
            inputs, 
            targets_template, 
            forcings
        )
        
        return {
            'inputs': inputs,
            'targets': targets,
            'forcings': forcings,
            'base_predictions': base_predictions,
            'finetuned_predictions': finetuned_predictions,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
    
    def compute_metrics(self, simulation_results):
        """
        Compute metrics for simulation results.
        
        Args:
            simulation_results: Results from a simulation
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for variable in self.variables_to_compare:
            # Extract variable data
            targets = simulation_results['targets'][variable]
            base_preds = simulation_results['base_predictions'][variable]
            finetuned_preds = simulation_results['finetuned_predictions'][variable]
            
            # Compute RMSE over time
            base_rmse = np.sqrt(((base_preds - targets)**2).mean(dim=['lat', 'lon']))
            finetuned_rmse = np.sqrt(((finetuned_preds - targets)**2).mean(dim=['lat', 'lon']))
            
            # Compute absolute difference over time
            base_abs_diff = np.abs(base_preds - targets).mean(dim=['lat', 'lon'])
            finetuned_abs_diff = np.abs(finetuned_preds - targets).mean(dim=['lat', 'lon'])
            
            metrics[f'{variable}_base_rmse'] = base_rmse
            metrics[f'{variable}_finetuned_rmse'] = finetuned_rmse
            metrics[f'{variable}_base_abs_diff'] = base_abs_diff
            metrics[f'{variable}_finetuned_abs_diff'] = finetuned_abs_diff
            
            # Compute regional metrics
            region_metrics = {}
            for region_name, region_bounds in temperature_zones.items():
                # Extract regional data
                region_targets = targets.sel(
                    lat=slice(region_bounds['lat_max'], region_bounds['lat_min']),
                    lon=slice(region_bounds['lon_min'], region_bounds['lon_max'])
                )
                
                region_base_preds = base_preds.sel(
                    lat=slice(region_bounds['lat_max'], region_bounds['lat_min']),
                    lon=slice(region_bounds['lon_min'], region_bounds['lon_max'])
                )
                
                region_finetuned_preds = finetuned_preds.sel(
                    lat=slice(region_bounds['lat_max'], region_bounds['lat_min']),
                    lon=slice(region_bounds['lon_min'], region_bounds['lon_max'])
                )
                
                # Compute regional RMSE
                region_base_rmse = np.sqrt(((region_base_preds - region_targets)**2).mean(dim=['lat', 'lon']))
                region_finetuned_rmse = np.sqrt(((region_finetuned_preds - region_targets)**2).mean(dim=['lat', 'lon']))
                
                region_metrics[f'{region_name}_base_rmse'] = region_base_rmse
                region_metrics[f'{region_name}_finetuned_rmse'] = region_finetuned_rmse
            
            metrics[f'{variable}_region_metrics'] = region_metrics
        
        return metrics
    
    def plot_rmse_comparison(self, metrics, variable, simulation_id):
        """
        Plot RMSE comparison between base and fine-tuned models.
        
        Args:
            metrics: Metrics dictionary
            variable: Variable name
            simulation_id: Simulation identifier
        """
        base_rmse = metrics[f'{variable}_base_rmse'].values.squeeze()
        finetuned_rmse = metrics[f'{variable}_finetuned_rmse'].values.squeeze()
        time_steps = np.arange(len(base_rmse))
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, base_rmse, label='Base Model', marker='o')
        plt.plot(time_steps, finetuned_rmse, label='Fine-tuned Model', marker='s')
        
        plt.xlabel('Forecast Lead Time (6-hour steps)')
        plt.ylabel('RMSE')
        plt.title(f'RMSE Comparison: {variable} - Simulation {simulation_id}')
        plt.legend()
        plt.grid(True)
        
        var_name = variable.replace('_', '-')
        plt.savefig(f'plots/comparison/rmse_{var_name}_sim{simulation_id}.png')
        plt.close()
    
    def plot_abs_diff_comparison(self, metrics, variable, simulation_id):
        """
        Plot absolute difference comparison between base and fine-tuned models.
        
        Args:
            metrics: Metrics dictionary
            variable: Variable name
            simulation_id: Simulation identifier
        """
        base_abs_diff = metrics[f'{variable}_base_abs_diff'].values.squeeze()
        finetuned_abs_diff = metrics[f'{variable}_finetuned_abs_diff'].values.squeeze()
        time_steps = np.arange(len(base_abs_diff))
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, base_abs_diff, label='Base Model', marker='o')
        plt.plot(time_steps, finetuned_abs_diff, label='Fine-tuned Model', marker='s')
        
        plt.xlabel('Forecast Lead Time (6-hour steps)')
        plt.ylabel('Mean Absolute Difference')
        plt.title(f'Absolute Difference Comparison: {variable} - Simulation {simulation_id}')
        plt.legend()
        plt.grid(True)
        
        var_name = variable.replace('_', '-')
        plt.savefig(f'plots/comparison/abs_diff_{var_name}_sim{simulation_id}.png')
        plt.close()
    
    def plot_rmse_map(self, simulation_results, variable, time_step, simulation_id):
        """
        Plot RMSE map for India for a specific time step.
        
        Args:
            simulation_results: Results from a simulation
            variable: Variable name
            time_step: Time step to plot
            simulation_id: Simulation identifier
        """
        # Extract data for this time step
        targets = simulation_results['targets'][variable].isel(time=time_step).squeeze()
        base_preds = simulation_results['base_predictions'][variable].isel(time=time_step).squeeze()
        finetuned_preds = simulation_results['finetuned_predictions'][variable].isel(time=time_step).squeeze()
        
        # Compute squared errors
        base_se = (base_preds - targets) ** 2
        finetuned_se = (finetuned_preds - targets) ** 2
        
        # India extent
        extent = [68, 98, 6, 38]
        
        # Create figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot base model RMSE
        ax = axs[0]
        base_se.sel(lat=slice(extent[3], extent[2]), lon=slice(extent[0], extent[1])).plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='viridis',
            cbar_kwargs={'label': f'Squared Error - {variable}'}
        )
        ax.coastlines()
        ax.set_extent(extent)
        ax.set_title(f'Base Model - {variable} - Time Step {time_step}')
        
        # Plot fine-tuned model RMSE
        ax = axs[1]
        finetuned_se.sel(lat=slice(extent[3], extent[2]), lon=slice(extent[0], extent[1])).plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='viridis',
            cbar_kwargs={'label': f'Squared Error - {variable}'}
        )
        ax.coastlines()
        ax.set_extent(extent)
        ax.set_title(f'Fine-tuned Model - {variable} - Time Step {time_step}')
        
        plt.tight_layout()
        var_name = variable.replace('_', '-')
        plt.savefig(f'plots/comparison/rmse_map_{var_name}_time{time_step}_sim{simulation_id}.png')
        plt.close()
    
    def plot_regional_rmse(self, metrics, variable, region_name, simulation_id):
        """
        Plot regional RMSE comparison.
        
        Args:
            metrics: Metrics dictionary
            variable: Variable name
            region_name: Region name
            simulation_id: Simulation identifier
        """
        region_metrics = metrics[f'{variable}_region_metrics']
        base_rmse = region_metrics[f'{region_name}_base_rmse'].values.squeeze()
        finetuned_rmse = region_metrics[f'{region_name}_finetuned_rmse'].values.squeeze()
        time_steps = np.arange(len(base_rmse))
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, base_rmse, label='Base Model', marker='o')
        plt.plot(time_steps, finetuned_rmse, label='Fine-tuned Model', marker='s')
        
        plt.xlabel('Forecast Lead Time (6-hour steps)')
        plt.ylabel('RMSE')
        plt.title(f'Regional RMSE Comparison: {variable} - {region_name} - Simulation {simulation_id}')
        plt.legend()
        plt.grid(True)
        
        var_name = variable.replace('_', '-')
        region_name_file = region_name.replace('_', '-')
        plt.savefig(f'plots/comparison/regions/rmse_{var_name}_{region_name_file}_sim{simulation_id}.png')
        plt.close()
    
    def plot_regional_rmse_map(self, simulation_results, variable, region_name, time_step, simulation_id):
        """
        Plot regional RMSE map for a specific time step.
        
        Args:
            simulation_results: Results from a simulation
            variable: Variable name
            region_name: Region name
            time_step: Time step to plot
            simulation_id: Simulation identifier
        """
        region_bounds = temperature_zones[region_name]
        
        # Extract data for this time step
        targets = simulation_results['targets'][variable].isel(time=time_step).squeeze()
        base_preds = simulation_results['base_predictions'][variable].isel(time=time_step).squeeze()
        finetuned_preds = simulation_results['finetuned_predictions'][variable].isel(time=time_step).squeeze()
        
        # Extract regional data
        region_targets = targets.sel(
            lat=slice(region_bounds['lat_max'], region_bounds['lat_min']),
            lon=slice(region_bounds['lon_min'], region_bounds['lon_max'])
        )
        
        region_base_preds = base_preds.sel(
            lat=slice(region_bounds['lat_max'], region_bounds['lat_min']),
            lon=slice(region_bounds['lon_min'], region_bounds['lon_max'])
        )
        
        region_finetuned_preds = finetuned_preds.sel(
            lat=slice(region_bounds['lat_max'], region_bounds['lat_min']),
            lon=slice(region_bounds['lon_min'], region_bounds['lon_max'])
        )
        
        # Compute squared errors
        base_se = (region_base_preds - region_targets) ** 2
        finetuned_se = (region_finetuned_preds - region_targets) ** 2
        
        # Define region extent
        extent = [
            region_bounds['lon_min'],
            region_bounds['lon_max'],
            region_bounds['lat_min'],
            region_bounds['lat_max']
        ]
        
        # Create figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot base model RMSE
        ax = axs[0]
        base_se.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='viridis',
            cbar_kwargs={'label': f'Squared Error - {variable}'}
        )
        ax.coastlines()
        ax.set_extent(extent)
        ax.set_title(f'Base Model - {variable} - {region_name} - Time Step {time_step}')
        
        # Plot fine-tuned model RMSE
        ax = axs[1]
        finetuned_se.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='viridis',
            cbar_kwargs={'label': f'Squared Error - {variable}'}
        )
        ax.coastlines()
        ax.set_extent(extent)
        ax.set_title(f'Fine-tuned Model - {variable} - {region_name} - Time Step {time_step}')
        
        plt.tight_layout()
        var_name = variable.replace('_', '-')
        region_name_file = region_name.replace('_', '-')
        plt.savefig(f'plots/comparison/regions/rmse_map_{var_name}_{region_name_file}_time{time_step}_sim{simulation_id}.png')
        plt.close()
    
    def run_comparison(self, era5_data_path):
        """
        Run the comparison between base and fine-tuned models.
        
        Args:
            era5_data_path: Path to ERA5 data
        """
        # Load ERA5 data
        era5_data = self.load_era5_data(era5_data_path)
        
        # Extract simulation periods
        simulation_periods = self.extract_simulation_periods(era5_data)
        
        all_metrics = []
        
        for sim_id, (start_idx, end_idx) in enumerate(tqdm(simulation_periods, desc="Running Simulations")):
            logger.info(f"Running simulation {sim_id}: time steps {start_idx} to {end_idx}")
            
            # Run simulation
            simulation_results = self.run_simulation(era5_data, start_idx, end_idx)
            
            # Compute metrics
            metrics = self.compute_metrics(simulation_results)
            all_metrics.append(metrics)
            
            # Plot RMSE comparison
            for variable in self.variables_to_compare:
                # 1. Line graph of RMSE
                self.plot_rmse_comparison(metrics, variable, sim_id)
                
                # 2. Line graph of absolute difference
                self.plot_abs_diff_comparison(metrics, variable, sim_id)
                
                # 3. RMSE maps for one time step (middle of forecast)
                mid_step = (simulation_results['end_idx'] - simulation_results['start_idx']) // 2
                self.plot_rmse_map(simulation_results, variable, mid_step, sim_id)
                
                # 4. Regional analysis
                region_metrics = metrics[f'{variable}_region_metrics']
                for region_name in temperature_zones.keys():
                    # Line graph of regional RMSE
                    self.plot_regional_rmse(metrics, variable, region_name, sim_id)
                    
                    # Regional RMSE map
                    self.plot_regional_rmse_map(simulation_results, variable, region_name, mid_step, sim_id)
        
        return all_metrics

def main():
    parser = argparse.ArgumentParser(description='Compare base and fine-tuned weather forecasting models')
    parser.add_argument('--base_model_path', default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13.npz', 
                        help='Path to base model checkpoint')
    parser.add_argument('--finetuned_model_path', default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig.npz', 
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--normalization_data_path', default='/Datastorage/saptarishi.dhanuka_asp25/gc_norms', 
                        help='Path to normalization data')
    parser.add_argument('--era5_data_path', default='/Datastorage/saptarishi.dhanuka_asp25/era5_data/wb_era5_jan2016_temp_ppt.zarr/', 
                        help='Path to ERA5 data')
    parser.add_argument('--model_levels', default=13, type=int, choices=[13, 37], 
                        help='Number of pressure levels')
    parser.add_argument('--model_resolution', default=1.0, type=float, choices=[0.25, 1.0], 
                        help='Model resolution')
    
    args = parser.parse_args()
    
    comparator = WeatherForecastComparison(
        args.base_model_path,
        args.finetuned_model_path,
        args.normalization_data_path,
        args.model_levels,
        args.model_resolution
    )
    
    all_metrics = comparator.run_comparison(args.era5_data_path)
    
    # Save metrics to a file
    metrics_file = f"metrics_comparison_{current_date}.npz"
    np.savez(metrics_file, metrics=all_metrics)
    
    logger.info(f"Comparison completed. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()


