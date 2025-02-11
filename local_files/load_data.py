import xarray
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from graphcast import graphcast
import numpy as np
import pandas as pd
def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

@dataclass
class PlotConfig:
    variable: str = "2m_temperature"
    level: float = 500
    robust: bool = True
    max_steps: Optional[int] = None

def get_available_datasets(gcs_bucket, dir_prefix: str) -> List[str]:
    """Get list of available dataset files from GCS bucket."""
    return [
        blob.name.removeprefix(dir_prefix + "dataset/")
        for blob in gcs_bucket.list_blobs(prefix=dir_prefix + "dataset/")
        if blob.name != dir_prefix + "dataset/"
    ]

def data_valid_for_model(
    file_name: str,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig
) -> bool:
    """Check if dataset is valid for given model configuration."""
    file_parts = parse_file_parts(file_name.removesuffix(".nc"))
    return (
        model_config.resolution in (0, float(file_parts["res"])) and
        len(task_config.pressure_levels) == int(file_parts["levels"]) and
        (
            ("total_precipitation_6hr" in task_config.input_variables and
             file_parts["source"] in ("era5", "fake")) or
            ("total_precipitation_6hr" not in task_config.input_variables and
             file_parts["source"] in ("hres", "fake"))
        )
    )

def get_filtered_datasets(
    gcs_bucket,
    dir_prefix: str,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig
) -> List[Tuple[str, str]]:
    """
    Get list of datasets that are valid for the given model configuration.
    
    Returns:
        List of tuples (description, filename)
    """
    datasets = get_available_datasets(gcs_bucket, dir_prefix)
    return [
        (
            ", ".join([f"{k}: {v}" for k, v in parse_file_parts(option.removesuffix(".nc")).items()]),
            option
        )
        for option in datasets
        if data_valid_for_model(option, model_config, task_config)
    ]

def load_dataset(
    gcs_bucket,
    dir_prefix: str,
    dataset_file: str,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    fake: Optional[bool] = True
) -> xarray.Dataset:
    """
    Load and validate a dataset.
    
    Args:
        gcs_bucket: Google Cloud Storage bucket
        dir_prefix: Prefix for files in the bucket
        dataset_file: Name of dataset file to load
        model_config: Model configuration
        task_config: Task configuration
    
    Returns:
        Loaded dataset
    """
    if fake:
        return generate_sample_era5_dataset(model_config)
        
    
    if not data_valid_for_model(dataset_file, model_config, task_config):
        raise ValueError(
            f"Invalid dataset file: {dataset_file}. Choose a dataset compatible with your model configuration."
        )
    
    with gcs_bucket.blob(f"{dir_prefix}dataset/{dataset_file}").open("rb") as f:
        example_batch = xarray.load_dataset(f).compute()
    
    if example_batch.dims["time"] < 3:  # 2 for input, >=1 for targets
        raise ValueError("Dataset must have at least 3 time steps")
        
    print(", ".join([
        f"{k}: {v}" 
        for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()
    ]))
    
    return example_batch

def get_plot_config(
    dataset: xarray.Dataset,
    variable: Optional[str] = None,
    level: Optional[float] = None,
    robust: Optional[bool] = None,
    max_steps: Optional[int] = None
) -> PlotConfig:
    """
    Create plot configuration with defaults based on dataset.
    
    Args:
        dataset: Dataset to plot
        variable: Variable to plot (defaults to "2m_temperature")
        level: Pressure level to plot (defaults to 500 if available)
        robust: Whether to use robust scaling (defaults to True)
        max_steps: Maximum number of time steps to plot (defaults to all available)
    
    Returns:
        PlotConfig object
    """
    config = PlotConfig()
    
    # Set variable
    if variable is not None:
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable {variable} not found in dataset")
        config.variable = variable
    elif "2m_temperature" in dataset.data_vars:
        config.variable = "2m_temperature"
    else:
        config.variable = list(dataset.data_vars.keys())[0]
    
    # Set level
    if level is not None:
        if "level" in dataset.coords and level not in dataset.coords["level"].values:
            raise ValueError(f"Level {level} not found in dataset")
        config.level = level
    elif "level" in dataset.coords:
        config.level = float(dataset.coords["level"].values[0])
    
    # Set robust
    if robust is not None:
        config.robust = robust
    
    # Set max_steps
    if max_steps is not None:
        if max_steps > dataset.dims["time"]:
            raise ValueError(f"max_steps ({max_steps}) exceeds available time steps ({dataset.dims['time']})")
        config.max_steps = max_steps
    else:
        config.max_steps = dataset.dims["time"]
    
    return config

def generate_sample_era5_dataset(
    date='2022-01-01', 
    model_config = None,
    time_steps=3
):
    """
    Generate a sample ERA5 dataset with random values matching original specifications.
    
    Parameters:
    - date: Base date for the dataset
    - lon_res: Longitude resolution
    - lat_res: Latitude resolution
    - levels: Number of vertical levels
    - time_steps: Number of time steps
    
    Returns:
    xarray.Dataset with random values
    """
    # Generate coordinate arrays
    lons = np.arange(0, 360, model_config.resolution)
    lats = np.arange(-90, 90 + model_config.resolution, model_config.resolution)
    level_values = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 
                              225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 
                              750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
    times = pd.timedelta_range(start='0 days', periods=time_steps, freq='6H')
    
    # Create datetime coordinates
    base_datetime = pd.to_datetime(date)

    # Create datetime coordinates
    base_datetime = pd.to_datetime(date)
    datetime_coords = np.array([base_datetime + pd.Timedelta(t) for t in times])
    
    # Create dataset with random data
    ds = xarray.Dataset(
        data_vars={
            'geopotential_at_surface': (['lat', 'lon'], np.random.uniform(20000, 30000, size=(len(lats), len(lons)))),
            'land_sea_mask': (['lat', 'lon'], np.random.choice([0.0, 1.0], size=(len(lats), len(lons)))),
            '2m_temperature': (['batch', 'time', 'lat', 'lon'], np.random.uniform(240, 310, size=(1, len(times), len(lats), len(lons)))),
            'mean_sea_level_pressure': (['batch', 'time', 'lat', 'lon'], np.random.uniform(95000, 105000, size=(1, len(times), len(lats), len(lons)))),
            '10m_v_component_of_wind': (['batch', 'time', 'lat', 'lon'], np.random.uniform(-10, 10, size=(1, len(times), len(lats), len(lons)))),
            '10m_u_component_of_wind': (['batch', 'time', 'lat', 'lon'], np.random.uniform(-10, 10, size=(1, len(times), len(lats), len(lons)))),
            'total_precipitation_6hr': (['batch', 'time', 'lat', 'lon'], np.random.uniform(0, 0.01, size=(1, len(times), len(lats), len(lons)))),
            'toa_incident_solar_radiation': (['batch', 'time', 'lat', 'lon'], np.random.uniform(0, 2000000, size=(1, len(times), len(lats), len(lons)))),
            'temperature': (['batch', 'time', 'level', 'lat', 'lon'], np.random.uniform(250, 300, size=(1, len(times), len(level_values), len(lats), len(lons)))),
            'geopotential': (['batch', 'time', 'level', 'lat', 'lon'], np.random.uniform(0, 500000, size=(1, len(times), len(level_values), len(lats), len(lons)))),
            'u_component_of_wind': (['batch', 'time', 'level', 'lat', 'lon'], np.random.uniform(-10, 10, size=(1, len(times), len(level_values), len(lats), len(lons)))),
            'v_component_of_wind': (['batch', 'time', 'level', 'lat', 'lon'], np.random.uniform(-10, 10, size=(1, len(times), len(level_values), len(lats), len(lons)))),
            'vertical_velocity': (['batch', 'time', 'level', 'lat', 'lon'], np.random.uniform(-1, 1, size=(1, len(times), len(level_values), len(lats), len(lons)))),
            'specific_humidity': (['batch', 'time', 'level', 'lat', 'lon'], np.random.uniform(0, 0.01, size=(1, len(times), len(level_values), len(lats), len(lons))))
        },
        coords={
            'lon': lons,
            'lat': lats,
            'level': level_values,
            'time': times,
            'datetime': (['batch', 'time'], datetime_coords.reshape(1, -1)),
            'batch': [0]
        }
    )
    
    return ds
"""
# Get list of available datasets filtered for model configuration
datasets = get_filtered_datasets(gcs_bucket, dir_prefix, model_config, task_config)

# Load the first compatible dataset
if datasets:
    dataset = load_dataset(
        gcs_bucket,
        dir_prefix,
        datasets[0][1],  # filename from first dataset tuple
        model_config,
        task_config
    )

    # Get plot configuration with defaults
    plot_config = get_plot_config(dataset)

    # Or specify custom plot configuration
    custom_plot_config = get_plot_config(
        dataset,
        variable="specific_humidity",
        level=850,
        robust=False,
        max_steps=10
    )
"""