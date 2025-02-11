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