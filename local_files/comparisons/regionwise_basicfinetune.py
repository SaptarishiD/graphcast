#<run_graphcast_train_one_step.py>
import argparse
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'
import dataclasses
import xarray as xr
import numpy as np
import pandas as pd
import jax
import optax
import setup_jax_functions
from graphcast import checkpoint, data_utils, rollout, graphcast
from datetime import datetime
import save_params_utils

import logging

from plotting import scale, select, plot_data, save_animation, save_static_plot

from metrics import compute_rmse, compute_mae, compute_bias, compute_acc


jax.config.update('jax_disable_jit', True)


mean_by_level = None
stddev_by_level = None
diffs_stddev_by_level = None
model_config = None
task_config = None
params = None
state = None



current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
# def diff_predictions(new_predictions: xr.Dataset, old_predictions: xr:Dataset):



def generate_sample_era5_dataset(
    date='2022-01-01', 
    model_config = None,
    task_config = None,
    time_steps=4
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
    xr.Dataset with random values
    """
    lons = np.arange(0, 360, model_config.resolution)
    lats = np.arange(-90, 90 + model_config.resolution, model_config.resolution)
    # if task_config.levels == 13:
    level_values = np.array(task_config.pressure_levels)
    # level_values = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 
    #                           225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 
    #                           750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
    times = pd.timedelta_range(start='0 days', periods=time_steps, freq='6h')
    
    base_datetime = pd.to_datetime(date)

    base_datetime = pd.to_datetime(date)
    datetime_coords = np.array([base_datetime + pd.Timedelta(t) for t in times])
    
    ds = xr.Dataset(
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
    
    import numpy as np
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file = f"loggers/printing_logs{current_date}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger()

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
      filename = '../gc_weights/graphcast_0.25_37.npz'
    elif args.model_resolution == 0.25:
      filename = '../gc_weights/graphcast_0.25_13.npz'
    else:
      filename = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13.npz'
    with open(filename, 'rb') as f:
      ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params

    state = {}
    model_config = ckpt.model_config
    print(model_config)
    task_config = ckpt.task_config
    # print(task_config)
    setup_jax_functions.configs['model_config'] = model_config
    setup_jax_functions.configs['task_config'] = task_config
    setup_jax_functions.configs['state'] = state
    setup_jax_functions.configs['params'] = params
    setup_jax_functions.configs['stddev_by_level'] = stddev_by_level
    setup_jax_functions.configs['diffs_stddev_by_level'] = diffs_stddev_by_level
    setup_jax_functions.configs['mean_by_level'] = mean_by_level



    # example_batch =  generate_sample_era5_dataset(model_config=model_config, task_config=task_config)

    # with open('/Datastorage/saptarishi.dhanuka_asp25/era5_data/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc', 'rb') as f:
      
    f = open('/Datastorage/saptarishi.dhanuka_asp25/era5_data/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-12.nc', 'rb')
    example_batch = xr.load_dataset(f).compute()
    f.close()

    assert example_batch.sizes["time"] >= 3

    

    """
    arco = xr.open_zarr("/Datastorage/divij.khaitan_asp25/era5/era5_2020.zarr")
    old_lats = arco['latitude'].values
    old_lons = arco['longitude'].values
    new_lats = np.arange(-90.0, 90.0 + 1e-8, 1.0)
    new_lats = np.flip(new_lats)
    new_lons = np.arange(0, 359.75 + 1e-8, 1.0)
    arco = arco.interp({'latitude': new_lats, 'longitude': new_lons}, 
                            method='linear',
                            kwargs={'fill_value': None})
    
    input_vars = ['10m_u_component_of_wind',
                'geopotential_at_surface',
                '10m_v_component_of_wind',
                'specific_humidity',
                'land_sea_mask',
                'vertical_velocity',
                'geopotential',
                'v_component_of_wind',
                'temperature',
                'total_precipitation_6hr',
                'mean_sea_level_pressure',
                '2m_temperature',
                'u_component_of_wind']
    
    arco = arco.drop_vars(['toa_incident_solar_radiation',
    'year_progress_sin',
    'year_progress_cos',
    'day_progress_sin',
    'day_progress_cos','cos_latitude',
    'cos_longitude','sin_longitude'])

    arco = arco.rename({'total_precipitation': 'total_precipitation_6hr'})

    arco = arco.expand_dims(batch=1)
    arco = arco.rename({'latitude': 'lat', 'longitude': 'lon'})

    datetime_array = arco['time'].values
    # Calculate the time coordinate in 6-hour increments (in nanoseconarco)
    time_array = np.arange(0, len(datetime_array) * 21600000000000, 21600000000000, dtype='timedelta64[ns]')

    # Add the new 'time' coordinate to the dataset
    arco1 = arco.assign_coords(datetime=('time', time_array))

    temp_time = arco1.coords["time"].copy()
    temp_datetime = arco1.coords["datetime"].copy()

    # Reassign the coordinates, swapping their values
    arco1 = arco1.assign_coords(
        time=temp_datetime,
        datetime=temp_time
    )

    logger.info("Coordinates first time")
    logger.info(arco1)
    logger.info("\n")
    logger.info(arco1.coords)
    logger.info("\n")


    arco1['geopotential_at_surface'] = arco1['geopotential_at_surface'].isel(batch=0, time=0)
    arco1['land_sea_mask'] = arco1['land_sea_mask'].isel(batch=0, time=0)

    old_datetime = arco1["datetime"].values  # shape (1489,)

    # For our purposes, we want the coordinate to have shape (batch, time). Since the batch
    # dimension is of length 1, we can simply add a new axis.
    new_datetime = old_datetime[np.newaxis, :]  # shape becomes (1, 1489)

    # Now, reassign the "datetime" coordinate to have dims ("batch", "time").
    arco1 = arco1.assign_coords(datetime=(("batch", "time"), new_datetime))

    print(f"Coordinates after reassigning: {arco1.coords}\n")


    logger.info(arco1.nbytes)

    select_time = arco1.isel(time=slice(0, 60))
    tik = datetime.now()



    example_batch = select_time.load()
    """

    



    train_steps = 2
    eval_steps = 4
    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
        **dataclasses.asdict(task_config))
    

    # print("All Examples:  ", example_batch.dims.mapping)
    # print("Train Inputs:  ", train_inputs.dims.mapping)
    # print("Train Targets: ", train_targets.dims.mapping)
    # print("Train Forcings:", train_forcings.dims.mapping)
    # print("Eval Inputs:   ", eval_inputs.dims.mapping)
    # print("Eval Targets:  ", eval_targets.dims.mapping)
    # print("Eval Forcings: ", eval_forcings.dims.mapping)


    with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/diffs_stddev_by_level.nc', 'rb') as f:
      diffs_stddev_by_level = xr.load_dataset(f).compute()
    with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/stddev_by_level.nc', 'rb') as f:
      stddev_by_level = xr.load_dataset(f).compute()
    with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/mean_by_level.nc', 'rb') as f:
      mean_by_level = xr.load_dataset(f).compute()


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
      

    # print(example_batch)
    # print(train_inputs)
    # print(eval_inputs)


    loss_fn_jitted = setup_jax_functions.drop_state(setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(setup_jax_functions.loss_fn.apply))))

    grads_fn_jitted = setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(setup_jax_functions.grads_fn)))

    run_forward_jitted = setup_jax_functions.drop_state(setup_jax_functions.with_params(jax.jit(setup_jax_functions.with_configs(
        setup_jax_functions.run_forward.apply))))


      
    def run_model(params, state, inputs, targets_template, forcings):
        predictions = run_forward_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            params=params,
            state=state
        )
        return predictions

    import numpy as np


    # print("Rolling out chunked prediction")
    # predictions = rollout.chunked_prediction(
    # run_forward_jitted,
    # rng=jax.random.PRNGKey(0),
    # inputs=eval_inputs,
    # targets_template=eval_targets * np.nan,
    # forcings=eval_forcings)


    # print(predictions)

    # print("Rolling out chunked prediction done")
    loss, diagnostics = loss_fn_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings)

    print(f"\n======== Loss: {loss:.4f}, Diagnostics: {diagnostics} ========= \n")


    regions = {
    # "USA": [220, 310, 25, 83],
    # "Europe": [0, 60, 35, 75],
    # "West Asia": [10, 65,10, 45],
    # "Australia": [100, 160, -44, -10],
    # "East Asia": [90, 180, 1, 55],
    # "South America": [275, 330, -56, 13],
    "India": [68, 98, 6, 38],
    }

    for region in regions.keys():
        coords = regions[region]

        lr = 1e-3
        optimiser = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)
        old_params = params
        opt_state = optimiser.init(old_params)

        print("Starting finetuning for region:", region)

        grads_fn_jitted = jax.jit(setup_jax_functions.with_configs(setup_jax_functions.grads_fn))
        loss, diagnostics, next_state, grads = grads_fn_jitted(old_params, state, train_inputs, train_targets, train_forcings)

        print(f"Gradients done for {region} with loss: {loss:.4f}, Diagnostics: {diagnostics}")


        updates, opt_state = optimiser.update(grads, opt_state)
        new_params = optax.apply_updates(old_params, updates)


    # print("Loss with old params:", float(loss))

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_params_utils.save_model_params(new_params, f'/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_old_params{current_date}.npz')

    targets_template = eval_targets * np.nan


    predictions_old = run_model(old_params, state, eval_inputs, targets_template, eval_forcings)

    predictions_finetuned = run_model(new_params, state, eval_inputs, targets_template, eval_forcings)

    variable_name = '2m_temperature'
    # temperature_pred_old = predictions_old[variable_name]
    temperature_pred_finetuned = predictions_finetuned[variable_name]
    temperature_targets = eval_targets[variable_name]


    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from sklearn.metrics import mean_squared_error


    def plot_comparisons_for_region(time_step, region_name, bounds, variable):
        west, east, south, north = bounds
        extent = [west, east, south, north]

        fig, axs = plt.subplots(3, 1, figsize=(12, 18), subplot_kw={'projection': ccrs.PlateCarree()})

        # Difference between base prediction and ground truth
        ax = axs[0]
        diff_base_gt = predictions_old[variable].isel(time=time_step) - eval_targets[var].isel(time=time_step)
        diff_base_gt.squeeze().plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
            cbar_kwargs={'label': f'{var} Difference'})
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(f"{region_name} - Base Prediction vs Ground Truth (Time Step {time_step})  for {variable}")
        lon_min, lon_max, lat_min, lat_max = bounds[0], bounds[1], bounds[2], bounds[3]
        # rmse_base = np.sqrt(mean_squared_error(eval_targets[var].isel(time=time_step).values.flatten(), predictions_old[variable].isel(time=time_step).values.flatten()))
        rmse_base = np.sqrt(mean_squared_error(eval_targets[var].isel(time=time_step).sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.flatten(), predictions_old[variable].isel(time=time_step).sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.flatten()))


        # Difference between finetuned prediction and ground truth
        ax = axs[1]
        diff_finetuned_gt = predictions_finetuned[variable].isel(time=time_step) - eval_targets[var].isel(time=time_step)
        diff_finetuned_gt.squeeze().plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
            cbar_kwargs={'label': f'{var} Difference (K)'})
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(f"{region_name} - Finetuned Prediction vs Ground Truth (Time Step {time_step}) for {variable}")

        rmse_finetuned = np.sqrt(mean_squared_error(eval_targets[var].isel(time=time_step).sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.flatten(), predictions_finetuned[variable].isel(time=time_step).sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.flatten()))


        # Difference between finetuned and base predictions
        ax = axs[2]
        diff_finetuned_base = predictions_finetuned[variable].isel(time=time_step) - predictions_old[variable].isel(time=time_step)
        diff_finetuned_base.squeeze().plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm',
            cbar_kwargs={'label': f'{var} Difference (K)'})
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(f"{region_name} - Finetuned vs Base Prediction (Time Step {time_step})")

        plt.tight_layout()
        # plt.show()
        return rmse_base, rmse_finetuned

# Loop over each region and time step

    for region_name, bounds in regions.items():
        for var in ['2m_temperature']:
            base_rmses = []
            finetuned_rmses = []
            for time_step in [0,1,2]:
                base_rmses.append(plot_comparisons_for_region(time_step, region_name, bounds, var)[0])
                finetuned_rmses.append(plot_comparisons_for_region(time_step, region_name, bounds, var)[1])


        plt.figure(figsize=(8, 6))

        time_steps = [0,1,2]  
        time_steps_list = list(time_steps)
        plt.plot(time_steps_list,base_rmses, marker='o', linestyle='-', color='b', label='Base')
        plt.plot(time_steps_list,finetuned_rmses, marker='s', linestyle='-', color='g', label='Finetuned')

        # Adding labels and title
        plt.xlabel('List 1')
        plt.ylabel('Values')
        plt.title(f'{region_name} Plot')
        plt.legend()
        plt.grid(True)

        # Display the plot
        plt.savefig(f'./finetune_comparisons_{region_name}_{var}_{current_date}.png')


    
    # print(f"Optax apply updates done at {current_date}")


    print("FINISHED")

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

#</run_graphcast_train_one_step.py>