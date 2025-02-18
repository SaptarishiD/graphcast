#<extracting_and_plotting.py>
import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import xarray
import graphcast
from graphcast import data_utils
from plotting import save_animation, scale, select
from load_data import PlotConfig

def plot_example_data(
    dataset: xarray.Dataset,
    plot_config: PlotConfig,
    plot_size: float = 7
) -> None:
    """
    Plot dataset according to configuration.
    
    Args:
        dataset: Dataset to plot
        plot_config: Plot configuration
        plot_size: Size of the plot
    """
    data = {
        " ": scale(
            select(
                dataset,
                plot_config.variable,
                plot_config.level,
                plot_config.max_steps
            ),
            robust=plot_config.robust
        ),
    }
    
    fig_title = plot_config.variable
    if "level" in dataset[plot_config.variable].coords:
        fig_title += f" at {plot_config.level} hPa"
    
    save_animation(data, fig_title, plot_size, plot_config.robust)

@dataclass
class ExtractionConfig:
    train_steps: int = 1
    eval_steps: int = None  # Will be set to max possible steps by default
    
    def __post_init__(self):
        if self.eval_steps is None:
            self.eval_steps = self.train_steps

def validate_extraction_config(
    config: ExtractionConfig,
    dataset: xarray.Dataset
) -> None:
    """Validate extraction configuration against dataset."""
    max_steps = dataset.sizes["time"] - 2  # Need 2 steps for input
    
    if config.train_steps < 1 or config.train_steps > max_steps:
        raise ValueError(
            f"train_steps must be between 1 and {max_steps}, got {config.train_steps}"
        )
    
    if config.eval_steps < 1 or config.eval_steps > max_steps:
        raise ValueError(
            f"eval_steps must be between 1 and {max_steps}, got {config.eval_steps}"
        )

def extract_train_eval_data(
    dataset: xarray.Dataset,
    task_config: graphcast.TaskConfig,
    extraction_config: Optional[ExtractionConfig] = None
) -> Tuple[Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset],
           Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]]:
    """
    Extract training and evaluation data from dataset.
    
    Args:
        dataset: Source dataset
        task_config: Task configuration
        extraction_config: Extraction configuration (optional)
    
    Returns:
        Tuple of (train_data, eval_data), where each is a tuple of
        (inputs, targets, forcings)
    """
    if extraction_config is None:
        extraction_config = ExtractionConfig(
            train_steps=1,
            eval_steps=dataset.sizes["time"] - 2
        )
    
    validate_extraction_config(extraction_config, dataset)
    
    # Extract training data
    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        dataset,
        target_lead_times=slice("6h", f"{extraction_config.train_steps*6}h"),
        **dataclasses.asdict(task_config)
    )
    
    # Extract evaluation data
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        dataset,
        target_lead_times=slice("6h", f"{extraction_config.eval_steps*6}h"),
        **dataclasses.asdict(task_config)
    )
    
    print("All Examples:  ", dataset.dims.mapping)
    print("Train Inputs:  ", train_inputs.dims.mapping)
    print("Train Targets: ", train_targets.dims.mapping)
    print("Train Forcings:", train_forcings.dims.mapping)
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)
    
    return (
        (train_inputs, train_targets, train_forcings),
        (eval_inputs, eval_targets, eval_forcings)
    )

# Example usage:
"""
# Create plot configuration
plot_config = PlotConfig(
    variable="2m_temperature",
    level=500,
    robust=True,
    max_steps=10
)

# Plot the data
plot_example_data(dataset, plot_config)

# Create extraction configuration
extraction_config = ExtractionConfig(
    train_steps=1,
    eval_steps=dataset.sizes["time"] - 2  # Use maximum possible steps
)

# Extract training and evaluation data
train_data, eval_data = extract_train_eval_data(
    dataset,
    task_config,
    extraction_config
)

# Access the extracted data
train_inputs, train_targets, train_forcings = train_data
eval_inputs, eval_targets, eval_forcings = eval_data
"""

#</extracting_and_plotting.py>
# 
# #<gcs_downloader.py>
import os
from pathlib import Path
import xarray
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class StatisticsFiles:
    diffs_stddev_by_level: xarray.Dataset
    mean_by_level: xarray.Dataset
    stddev_by_level: xarray.Dataset

def download_statistics(
    gcs_bucket,
    dir_prefix: str,
    local_dir: str = "stats",
    force_download: bool = False
) -> None:
    """
    Download statistics files from GCS bucket to local directory.
    
    Args:
        gcs_bucket: Google Cloud Storage bucket
        dir_prefix: Prefix for files in the bucket
        local_dir: Local directory to save files to
        force_download: Whether to download files even if they exist locally
    """
    # Create local directory if it doesn't exist
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Files to download
    files = [
        "diffs_stddev_by_level.nc",
        "mean_by_level.nc",
        "stddev_by_level.nc"
    ]
    
    # Download each file
    for filename in files:
        local_file = local_path / filename
        
        # Skip if file exists and force_download is False
        if local_file.exists() and not force_download:
            print(f"Skipping {filename} (already exists)")
            continue
            
        print(f"Downloading {filename}...")
        with local_file.open("wb") as f_out:
            with gcs_bucket.blob(dir_prefix + f"stats/{filename}").open("rb") as f_in:
                f_out.write(f_in.read())

def load_statistics(local_dir: str = "stats") -> StatisticsFiles:
    """
    Load statistics from local files.
    
    Args:
        local_dir: Directory containing the statistics files
    
    Returns:
        StatisticsFiles object containing loaded datasets
    """
    local_path = Path(local_dir)
    
    if not local_path.exists():
        raise FileNotFoundError(f"Directory {local_dir} not found")
        
    try:
        diffs_stddev = xarray.load_dataset(local_path / "diffs_stddev_by_level.nc").compute()
        means = xarray.load_dataset(local_path / "mean_by_level.nc").compute()
        stddev = xarray.load_dataset(local_path / "stddev_by_level.nc").compute()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing statistics files in {local_dir}. "
            "Run download_statistics() first."
        ) from e
        
    return StatisticsFiles(diffs_stddev, means, stddev)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and load GraphCast statistics")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--prefix", required=True, help="Directory prefix in bucket")
    parser.add_argument("--local-dir", default="stats", help="Local directory for files")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    
    args = parser.parse_args()
    
    # Initialize GCS bucket (assuming google-cloud-storage is imported)
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(args.bucket)
    # gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    # dir_prefix = "graphcast/"
    # Download statistics
    download_statistics(bucket, args.prefix, args.local_dir, args.force)
    
    # Load and print basic info about the statistics
    stats = load_statistics(args.local_dir)
    # print("\nStatistics loaded successfully:")
    # print(f"Diffs StdDev shape: {stats.diffs_stddev_by_level.dims}")
    # print(f"Means shape: {stats.mean_by_level.dims}")
    # print(f"StdDev shape: {stats.stddev_by_level.dims}")
#</gcs_downloader.py>
# 
# 
# #<load_data.py>
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
#</load_data.py>
# 
# 
# #<loading_model.py>
import graphcast
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
from graphcast import checkpoint
class ModelSource(Enum):
    RANDOM = "random"
    CHECKPOINT = "checkpoint"

@dataclass
class RandomModelConfig:
    mesh_size: int = 4  # min: 4, max: 6
    gnn_msg_steps: int = 4  # min: 1, max: 32
    latent_size: int = 32  # options: [16, 32, 64, 128, 256, 512]
    pressure_levels: int = 13  # options: [13, 37]

def get_available_checkpoints(gcs_bucket, dir_prefix: str) -> List[str]:
    """Get list of available parameter files from GCS bucket."""
    return [
        blob.name.removeprefix(dir_prefix + "params/")
        for blob in gcs_bucket.list_blobs(prefix=dir_prefix + "params/")
        if blob.name != dir_prefix + "params/"
    ]

def load_model_config(
    gcs_bucket,
    dir_prefix: str,
    source: ModelSource = ModelSource.CHECKPOINT,
    checkpoint_file: Optional[str] = None,
    random_config: Optional[RandomModelConfig] = None
) -> tuple[Optional[Dict], Dict, graphcast.ModelConfig, graphcast.TaskConfig]:
    """
    Load model configuration either from checkpoint or create random configuration.
    
    Args:
        gcs_bucket: Google Cloud Storage bucket
        dir_prefix: Prefix for model files in the bucket
        source: Source of the model configuration (random or checkpoint)
        checkpoint_file: Name of checkpoint file to load (if source is checkpoint)
        random_config: Random model configuration parameters (if source is random)
    
    Returns:
        Tuple of (params, state, model_config, task_config)
    """
    if source == ModelSource.RANDOM:
        if random_config is None:
            random_config = RandomModelConfig()
            
        params = None  # Filled in by the model
        state = {}
        model_config = graphcast.ModelConfig(
            resolution=0,
            mesh_size=random_config.mesh_size,
            latent_size=random_config.latent_size,
            gnn_msg_steps=random_config.gnn_msg_steps,
            hidden_layers=1,
            radius_query_fraction_edge_length=0.6
        )
        task_config = graphcast.TaskConfig(
            input_variables=graphcast.TASK.input_variables,
            target_variables=graphcast.TASK.target_variables,
            forcing_variables=graphcast.TASK.forcing_variables,
            pressure_levels=graphcast.PRESSURE_LEVELS[random_config.pressure_levels],
            input_duration=graphcast.TASK.input_duration,
        )
        
    else:  # ModelSource.CHECKPOINT
        # If no checkpoint specified, use the first available one
        available_checkpoints = get_available_checkpoints(gcs_bucket, dir_prefix)
        if not checkpoint_file:
            checkpoint_file = available_checkpoints[0]
        elif checkpoint_file not in available_checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_file} not found. Available checkpoints: {available_checkpoints}")
            
        # Load the checkpoint
        with gcs_bucket.blob(f"{dir_prefix}params/{checkpoint_file}").open("rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
            
        params = ckpt.params
        state = {}
        model_config = ckpt.model_config
        task_config = ckpt.task_config
        
        # print("Model description:\n", ckpt.description, "\n")
        # print("Model license:\n", ckpt.license, "\n")
        
    return params, state, model_config, task_config

# Example usage:
"""
# Load default (smallest trained model from checkpoint)
params, state, model_config, task_config = load_model_config(
    gcs_bucket=gcs_bucket,
    dir_prefix=dir_prefix
)

# Load specific checkpoint
params, state, model_config, task_config = load_model_config(
    gcs_bucket=gcs_bucket,
    dir_prefix=dir_prefix,
    checkpoint_file="specific_checkpoint.ckpt"
)

# Create random configuration
random_config = RandomModelConfig(
    mesh_size=6,
    gnn_msg_steps=8,
    latent_size=64,
    pressure_levels=37
)

params, state, model_config, task_config = load_model_config(
    gcs_bucket=gcs_bucket,
    dir_prefix=dir_prefix,
    source=ModelSource.RANDOM,
    random_config=random_config
)
"""
#</loading_model.py>import xarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.animation as animation
from typing import Optional, Dict, Tuple
import math
import datetime
from pathlib import Path

def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax),
            ("RdBu_r" if center is not None else "viridis"))

def save_animation(
    data: Dict[str, Tuple[xarray.Dataset, matplotlib.colors.Normalize, str]],
    fig_title: str,
    output_path: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    fps: int = 4
) -> None:
    """
    Creates and saves an animation of the data to a file.
    
    Args:
        data: Dictionary of data to plot
        fig_title: Title for the figure
        output_path: Path to save the animation (supports .mp4, .gif)
        plot_size: Size multiplier for the plot
        robust: Whether to use robust scaling
        cols: Number of columns in the grid
        fps: Frames per second for the animation
    """
    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols,
                                plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"))
        images.append(im)

    def update(frame):
        if "time" in first_data.dims:
            td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
            figure.suptitle(f"{fig_title}, {td}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        for im, (plot_data, norm, cmap) in zip(images, data.values()):
            im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))
        return images

    ani = animation.FuncAnimation(
        fig=figure, func=update, frames=max_steps, interval=1000//fps)
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save animation based on file extension
    extension = Path(output_path).suffix.lower()
    if extension == '.mp4':
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(output_path, writer=writer)
    elif extension == '.gif':
        ani.save(output_path, writer='pillow', fps=fps)
    else:
        raise ValueError(f"Unsupported file extension: {extension}. Use .mp4 or .gif")
    
    plt.close(figure)

def save_static_plot(
    data: Dict[str, Tuple[xarray.Dataset, matplotlib.colors.Normalize, str]],
    fig_title: str,
    output_path: str,
    time_index: int = 0,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    dpi: int = 100
) -> None:
    """
    Creates and saves a static plot of the data at a specific time index.
    
    Args:
        data: Dictionary of data to plot
        fig_title: Title for the figure
        output_path: Path to save the plot (supports any format matplotlib supports)
        time_index: Time index to plot
        plot_size: Size multiplier for the plot
        robust: Whether to use robust scaling
        cols: Number of columns in the grid
        dpi: DPI for the output image
    """
    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols,
                                plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=time_index, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"))

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)

#<save_params_utils.py>
import jax.numpy as jnp
import numpy as np
import os


def flatten_dict(d, parent_key='', sep='//'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_model_params(d, file_path):
    flat_dict = flatten_dict(d)
    # Convert JAX arrays to NumPy for saving
    np_dict = {k: np.array(v) if isinstance(v, jnp.ndarray) else v for k, v in flat_dict.items()}
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez(file_path, **np_dict)

#</save_params_utils.py>