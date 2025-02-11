import argparse
import os
import dataclasses
import xarray
import numpy as np
import pandas as pd
import jax
import setup_jax_functions
from graphcast import checkpoint, data_utils, rollout, graphcast


mean_by_level = None
stddev_by_level = None
diffs_stddev_by_level = None
model_config = None
task_config = None
params = None
state = None


def generate_sample_era5_dataset(
    date='2022-01-01', 
    model_config = None,
    task_config = None,
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
    lons = np.arange(0, 360, model_config.resolution)
    lats = np.arange(-90, 90 + model_config.resolution, model_config.resolution)
    # if task_config.levels == 13:
    level_values = np.array(task_config.pressure_levels)
    # level_values = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 
    #                           225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 
    #                           750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
    times = pd.timedelta_range(start='0 days', periods=time_steps, freq='6H')
    
    base_datetime = pd.to_datetime(date)

    base_datetime = pd.to_datetime(date)
    datetime_coords = np.array([base_datetime + pd.Timedelta(t) for t in times])
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_levels', default=13, type=int, choices=[13, 37], help='Number of Pressure Levels')
    parser.add_argument('--model_resolution', default=1.0, type=float, choices=[1.0, 0.25], help='Model Resolution')
    parser.add_argument('--data_type', default='fake', type=str, choices=['fake', 'era5_1', 'era5_0.25'])
    parser.add_argument('--data_path', default=None, help='Path to load era5 data from if necessary')
    parser.add_argument('--means_path', default='./', help='Path to load mean and stdev for scaling from')
    global mean_by_level
    global stddev_by_level
    global diffs_stddev_by_level
    global model_config
    global task_config
    global params
    global state

    args = parser.parse_args()
    if args.model_levels == 37:
      filename = 'gc_weights/graphcast_0.25_37.npz'
    elif args.model_resolution == 0.25:
      filename = 'gc_weights/graphcast_0.25_13.npz'
    else:
      filename = 'gc_weights/graphcast_1_13.npz'
    with open(filename, 'rb') as f:
      ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    print(model_config)
    task_config = ckpt.task_config
    print(task_config)
    setup_jax_functions.configs['model_config'] = model_config
    setup_jax_functions.configs['task_config'] = task_config
    setup_jax_functions.configs['state'] = state
    setup_jax_functions.configs['params'] = params
    setup_jax_functions.configs['stddev_by_level'] = stddev_by_level
    setup_jax_functions.configs['diffs_stddev_by_level'] = diffs_stddev_by_level
    setup_jax_functions.configs['mean_by_level'] = mean_by_level
    example_batch =  generate_sample_era5_dataset(model_config=model_config, task_config=task_config)
    assert example_batch.dims["time"] >= 3

    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{6}h"),
    **dataclasses.asdict(task_config))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{6}h"),
        **dataclasses.asdict(task_config))
    print("All Examples:  ", example_batch.dims.mapping)
    print("Train Inputs:  ", train_inputs.dims.mapping)
    print("Train Targets: ", train_targets.dims.mapping)
    print("Train Forcings:", train_forcings.dims.mapping)
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)
    with open('diffs_stddev_by_level.nc', 'rb') as f:
      diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open('stddev_by_level.nc', 'rb') as f:
      stddev_by_level = xarray.load_dataset(f).compute()
    with open('mean_by_level.nc', 'rb') as f:
      mean_by_level = xarray.load_dataset(f).compute()
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

    if params is None:
      params, state = init_jitted(
          rng=jax.random.PRNGKey(0),
          inputs=train_inputs,
          targets_template=train_targets,
          forcings=train_forcings)
    print(example_batch)
    print(train_inputs)
    print(eval_inputs)
    loss_fn_jitted = setup_jax_functions.drop_state(setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(setup_jax_functions.loss_fn.apply))))
    grads_fn_jitted = setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(setup_jax_functions.grads_fn)))
    run_forward_jitted = setup_jax_functions.drop_state(setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(
        setup_jax_functions.run_forward.apply))))
    
    predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
    print(predictions)
    loss, diagnostics = loss_fn_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings)
    print("Loss:", float(loss))
    loss, diagnostics, next_state, grads = grads_fn_jitted(
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings)
    mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
    print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")
    print("Inputs:  ", train_inputs.dims.mapping)
    print("Targets: ", train_targets.dims.mapping)
    print("Forcings:", train_forcings.dims.mapping)

    predictions = run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets * np.nan,
        forcings=train_forcings)
    print(predictions)

if __name__=="__main__":
  main()
"""assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)"""