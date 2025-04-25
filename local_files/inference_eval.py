#<run_graphcast_train_one_step.py>
import logging
import argparse
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'
import dataclasses
import xarray as xr
import numpy as np
import pandas as pd
import jax
import optax
import save_params_utils
import setup_jax_functions
from graphcast import checkpoint, data_utils, rollout, graphcast
from datetime import datetime

from plotting import scale, select, plot_data, save_animation, save_static_plot

from metrics import compute_rmse, compute_mae, compute_bias, compute_acc


jax.config.update('jax_disable_jit', True)

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")



mean_by_level = None
stddev_by_level = None
diffs_stddev_by_level = None
model_config = None
task_config = None
params = None
state = None


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

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")

    log_file = f"printing_logs{current_date}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger()
    logger.info("Starting the script")
    


    args = parser.parse_args()
    if args.model_levels == 37:
      filename = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_0.25_37.npz'
            
    elif args.model_resolution == 0.25:
      filename = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_0.25_13.npz'
      
      with open(f'/Datastorage/saptarishi.dhanuka_asp25/era5_data/graphcast_dataset_source-era5_date-2022-01-01_res-{args.model_resolution}_levels-13_steps-12.nc', 'rb') as f:
        logger.info("Loading Dataset")
        tik = datetime.now()
        training_trial_batch = xr.load_dataset(f).compute()
        tok = datetime.now()
        logger.info(f"Dataset loaded in {tok - tik}")

    else:
        filename = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13.npz'
        dataset_name = "/Datastorage/saptarishi.dhanuka_asp25/era5_data/arco_era5_1.0_formatted.nc"

        arco = xr.open_zarr("/Datastorage/divij.khaitan_asp25/arco_era5.zarr")
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

        select_time = arco1.isel(time=slice(120, 132))
        tik = datetime.now()

        eval_batch = select_time.load()

        tok = datetime.now()
        # training_trial_batch = training_trial_batch.rename({'time': 'datetime'})
        logger.info("Eval batch time")
        logger.info(eval_batch.coords)
        logger.info(f"Dataset loaded in {tok - tik}")

    
    
    with open(filename, 'rb') as f:
      ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params

    new_params = save_params_utils.load_model_params("/home/saptarishi.dhanuka_asp25/weather/graphcast_dir/graphcast/local_files/params/params_finetune_test2025-04-14_03-05.npz")

    params = new_params


    with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/diffs_stddev_by_level.nc', 'rb') as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/stddev_by_level.nc', 'rb') as f:
        stddev_by_level = xr.load_dataset(f).compute()
    with open('/Datastorage/saptarishi.dhanuka_asp25/gc_norms/mean_by_level.nc', 'rb') as f:
        mean_by_level = xr.load_dataset(f).compute()


    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    logger.info(model_config)
    logger.info(task_config)
    setup_jax_functions.configs['model_config'] = model_config
    setup_jax_functions.configs['task_config'] = task_config
    setup_jax_functions.configs['state'] = state
    setup_jax_functions.configs['params'] = params
    setup_jax_functions.configs['stddev_by_level'] = stddev_by_level
    setup_jax_functions.configs['diffs_stddev_by_level'] = diffs_stddev_by_level
    setup_jax_functions.configs['mean_by_level'] = mean_by_level

    setup_jax_functions.update_configs({
        'params': params,
        'state': {},
        'model_config': ckpt.model_config,
        'task_config': ckpt.task_config,
        'mean_by_level': mean_by_level,
        'stddev_by_level': stddev_by_level,
        'diffs_stddev_by_level': diffs_stddev_by_level
    })

    init_jitted = jax.jit(setup_jax_functions.with_configs(setup_jax_functions.run_forward.init))
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
  

    # training_trial_batch = generate_sample_era5_dataset(model_config=model_config, task_config=task_config, time_steps = 10)

    # assert training_trial_batch.sizes["time"] >= 3

    # logger.info(f"Training batch time dimensions: {training_trial_batch.sizes['time']}")

    task_config_dict =  dataclasses.asdict(task_config)




    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    eval_batch, target_lead_times=slice("6h", f"{72}h"),
    **dataclasses.asdict(task_config))

    targets_template = eval_targets * np.nan
    predictions_finetuned = run_model(new_params, state, eval_inputs, targets_template, eval_forcings)

    print("saving predictions")
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    predictions_finetuned.to_netcdf(f'./predictions_finetuned_{current_date}.nc')





if __name__=="__main__":
  main()

