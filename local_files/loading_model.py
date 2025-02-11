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
        
        print("Model description:\n", ckpt.description, "\n")
        print("Model license:\n", ckpt.license, "\n")
        
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