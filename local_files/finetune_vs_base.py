#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys

import logging

# graphcast is in the parent directory so insert it into the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import argparse
import dataclasses
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import jax
import optax

from graphcast import checkpoint, data_utils, rollout, graphcast, normalization
import save_params_utils
import setup_jax_functions
from plotting import scale, select, plot_data, save_animation, save_static_plot
from metrics import compute_rmse, compute_mae, compute_bias, compute_acc
import matplotlib.pyplot as plt
# import pynvml
import time

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")




# jax.config.update('jax_disable_jit', True)

# for memory efficiency
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0' 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

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


# modify the gradients function signature (needed for finetuning with optax)
def grads_fn(params, state, inputs, targets, forcings, model_config, task_config):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = setup_jax_functions.loss_fn.apply(params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

def finetuning(train_inputs,train_targets,train_forcings,params):
    lr = 1e-4
    optimiser = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)
    opt_state = optimiser.init(params)

    grads_fn_jitted = jax.jit(setup_jax_functions.with_configs(setup_jax_functions.grads_fn))
    
    print("Setting up grads function")
    
    state = {}
    loss, diagnostics, next_state, grads = grads_fn_jitted(params, state, train_inputs, train_targets, train_forcings)

    logger = logging.getLogger()

    print("Losses calculated, now updating")

    updates, opt_state = optimiser.update(grads, opt_state)

    print("Applying updates")

    params = optax.apply_updates(params, updates)

    return params, loss


def create_input_data(example_batch_, task_config_dict, index=0, lookahead=4):

    minibatch = example_batch_.isel(time=slice(index, index + lookahead))

    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(minibatch, target_lead_times=slice('6h', f'{12}h'), **task_config_dict)
    return train_inputs, train_targets, train_forcings


def daily_mean(data, input_data=False):
    return data.mean('time', keepdims=True)

def combinedata(first,second):
    return first.merge(second)


def train_graphcast(data, params, task_config_dict, epochs=10):

    device = jax.devices()[0]

    log_file = f"training_logs_{current_date}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger()

    logger.info(f"JAX is running on: {device}")
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    gpu_utilization = []
    gpu_memory = []


    params_path = os.path.join('/home/saptarishi.dhanuka_asp25/weather/graphcast_dir/graphcast/local_files/params', f'params_finetune_test{current_date}.npz')

    loss_tracker = []
    lookahead = 5

    epoch_batch_loss = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        logger.info(f"Epoch number {epoch}")

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        gpu_utilization.append(util.gpu)           # in percent era5_temp_ppt2022_wb_1.0_regrid_all_vars.zarr
        gpu_memory.append(mem.used / 1024**2)      # in MB

        print(gpu_utilization)
        print(gpu_memory)
        
        time.sleep(1)

        for i in tqdm(range(data.dims.mapping['time']-3), desc="Training Batches"):
            logger.info(f"Time Batch number: {i}")
            
            # batches
            train_inputs, train_targets, train_forcings = create_input_data(data, task_config_dict=task_config_dict, index=i, lookahead = lookahead + 1)
            # combined_data = combiningData(train_targets, train_forcings)

            # train_inputs_mean_1_day = daily_mean(train_inputs)
            # train_targets_mean_1_day = daily_mean(train_targets)
            # train_forcings_mean_1_day = daily_mean(train_forcings)

            train_inputs_mean_1_day = train_inputs
            train_targets_mean_1_day = train_targets
            train_forcings_mean_1_day = train_forcings

            print("Train Inputs:  ", train_inputs_mean_1_day.sizes.mapping)
            print("Train Targets: ", train_targets_mean_1_day.sizes.mapping)
            print("Train Forcings:", train_forcings_mean_1_day.sizes.mapping)

            params, loss = finetuning(train_inputs_mean_1_day,train_targets_mean_1_day,train_forcings_mean_1_day, params)
            logger.info(f'\n =========== Loss for time batch number: {i} = {loss} =========== \n')
            print(f'\n =========== Loss for time batch number: {i} = {loss} =========== \n')
            loss_tracker.append(loss)
            epoch_batch_loss.append((epoch, i, loss)) 
            if i % 20 == 0 and i > 0:
                save_params_utils.save_model_params(params, f'{params_path}_{i}')

        logger.info(f'\n =========== Saving model after epoch {epoch} =========== \n')
        save_params_utils.save_model_params(params, params_path)

    pynvml.nvmlShutdown()
    plot_loss(epoch_batch_loss, current_date)
    plot_gpu_util(gpu_utilization, gpu_memory, current_date)

def plot_loss(epoch_batch_loss, date):
    """
    Plots the loss with epoch and batch number.
    """
    epochs = [entry[0] for entry in epoch_batch_loss]
    batches = [entry[1] for entry in epoch_batch_loss]
    losses = [entry[2] for entry in epoch_batch_loss]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label="Loss")
    plt.xlabel("Epoch and Batch Number (Combined Index)")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch and Batch Number")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/training/loss_plot{date}.png")
    plt.close()

def plot_gpu_util(gpu_utilization, gpu_memory, date):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(gpu_utilization, label='GPU Utilization (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Utilization')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(gpu_memory, label='GPU Memory (MB)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Memory Usage')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/training/gpu_utilization_memory_plot{date}.png")
    plt.close()


# In[3]:


import xarray as xr


# In[24]:


get_ipython().run_line_magic('reset', '-f')


# In[4]:


# 


# In[5]:


imd_targets_temp_ppt = xr.load_dataset("/Datastorage/saptarishi.dhanuka_asp25/imd_july2022_temp_ppt.grib", engine='cfgrib')
imd_targets_temp_ppt


# In[10]:


imd_targets_temp_ppt.isel(time=0, step=2)['tp']


# 

# In[4]:


wb_data = xr.open_zarr("/Datastorage/saptarishi.dhanuka_asp25/era5_data/wb_era5_jan2016_temp_ppt.zarr/")
wb_data


# In[5]:



# parser = argparse.ArgumentParser()
# parser.add_argument('--model_levels', default=13, type=int, choices=[13, 37], help='Number of Pressure Levels')
# parser.add_argument('--model_resolution', default=1.0, type=float, choices=[1.0, 0.25], help='Model Resolution')
# parser.add_argument('--data_type', default='fake', type=str, choices=['fake', 'era5_1', 'era5_0.25'])
# parser.add_argument('--data_path', default=None, help='Path to load era5 data from if necessary')
# parser.add_argument('--means_path', default='./', help='Path to load mean and stdev for scaling from')
global mean_by_level
global stddev_by_level
global diffs_stddev_by_level
global model_config
global task_config
global params
global state

log_file = f"loggers/printing_logs{current_date}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
logger.info("Starting the script")



# args = parser.parse_args()
model_levels = 13
model_resolution = 1.0
if model_levels == 37:
    filename = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_0.25_37.npz'
        
elif model_resolution == 0.25:
    filename = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_0.25_13.npz'
    
    with open(f'/Datastorage/saptarishi.dhanuka_asp25/era5_data/graphcast_dataset_source-era5_date-2022-01-01_res-{args.model_resolution}_levels-13_steps-12.nc', 'rb') as f:
        logger.info("Loading Dataset")
        tik = datetime.now()
        training_trial_batch = xr.load_dataset(f).compute()
        tok = datetime.now()
        logger.info(f"Dataset loaded in {tok - tik}")

else:

    
    filename = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/origs/graphcast_1_13.npz'
    dataset_name = "/Datastorage/saptarishi.dhanuka_asp25/era5_data/arco_era5_1.0_formatted.nc"

    arco = wb_data
    
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
    
    # arco = arco.drop_vars(['toa_incident_solar_radiation',
    # 'year_progress_sin',
    # 'year_progress_cos',
    # 'day_progress_sin',
    # 'day_progress_cos','cos_latitude',
    # 'cos_longitude','sin_longitude'])

    # arco = arco.rename({'total_precipitation': 'total_precipitation_6hr'})

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


    # arco1['geopotential_at_surface'] = arco1['geopotential_at_surface'].isel(batch=0, time=0)
    # arco1['land_sea_mask'] = arco1['land_sea_mask'].isel(batch=0, time=0)

    old_datetime = arco1["datetime"].values  # shape (1489,)

    # For our purposes, we want the coordinate to have shape (batch, time). Since the batch
    # dimension is of length 1, we can simply add a new axis.
    new_datetime = old_datetime[np.newaxis, :]  # shape becomes (1, 1489)

    # Now, reassign the "datetime" coordinate to have dims ("batch", "time").
    arco1 = arco1.assign_coords(datetime=(("batch", "time"), new_datetime))

    print(f"Coordinates after reassigning: {arco1.coords}\n")


    logger.info(arco1.nbytes)

    select_time = arco1.isel(time=slice(0, 240))
    tik = datetime.now()

    select_time_eval = arco1.isel(time=slice(7,71))

    eval_trial_batch = select_time_eval.load()

    print(f"Eval batch first time: {eval_trial_batch.time.values}")
    # training_trial_batch = select_time.load()

    tok = datetime.now()
    # training_trial_batch = training_trial_batch.rename({'time': 'datetime'})
    logger.info("Training batch time")
    # logger.info(training_trial_batch.coords)
    logger.info(f"Dataset loaded in {tok - tik}")

    # with open("/Datastorage/saptarishi.dhanuka_asp25/era5_data/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc", 'rb') as f:
    #     print("Loading Dataset")
    #     tik = datetime.now()
    #     training_trial_batch = xr.load_dataset(f).compute()
    #     # training_trial_batch = training_trial_batch.isel(time=slice(0, 12))
    #     # training_trial_batch = training_trial_batch.rename({'time': 'datetime'})
    #     print("Training batch time")
    #     print(training_trial_batch.datetime)
    #     tok = datetime.now()
    #     print(f"Dataset loaded in {tok - tik}")


with open(filename, 'rb') as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

params = ckpt.params


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
    'params': ckpt.params,
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


# In[6]:


eval_trial_batch


# In[7]:


eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
eval_trial_batch, target_lead_times=slice("6h", "168h"),
**dataclasses.asdict(task_config))


# In[6]:


eval_targets


# In[17]:


# since we did target lead times as 6h 60h we used 6h as inputs and from 6h onwards till 60 h later. So since the eval_trial_batch is from 2022-06-01, the eval targets start from 2022-06-15 06:00:00 and go till 2022-06-17 12:00:00
eval_targets


# In[8]:


jax.config.update("jax_enable_x64", True)


# In[10]:



# training_trial_batch = generate_sample_era5_dataset(model_config=model_config, task_config=task_config, time_steps = 10)

assert eval_trial_batch.sizes["time"] >= 3

logger.info(f"Eval batch time dimensions: {eval_trial_batch.sizes['time']}")

task_config_dict =  dataclasses.asdict(task_config)
# task_config_dict.pop('input_duration')

logger.info("Starting training")

# train_graphcast(training_trial_batch, params, task_config_dict, epochs=3)


print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

# print("Running chunked predictions")

# predictions = rollout.chunked_prediction(
# run_forward_jitted,
# rng=jax.random.PRNGKey(0),
# inputs=eval_inputs,
# targets_template=eval_targets * np.nan,
# forcings=eval_forcings)

# print("Predictions:   ", predictions.dims.mapping)
# predictions.to_netcdf("/Datastorage/saptarishi.dhanuka_asp25/predictions.nc")



new_params_path = "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig.npz"

f = open(new_params_path, "rb")
new_ckpt = checkpoint.load(f, graphcast.CheckPoint)
new_params = new_ckpt.params

# new_params = save_params_utils.load_model_params(new_params_path)

print("Loaded New Params: ", new_params.keys())

targets_template = eval_targets * np.nan

print("Running old params")
predictions_old = run_model(params, state, eval_inputs, targets_template, eval_forcings)

print("Predictions Old: ", predictions_old.dims.mapping)
# predictions_old.to_netcdf("/Datastorage/saptarishi.dhanuka_asp25/predictions_old.nc")


# In[11]:


print("Running new params")

predictions_finetuned = run_model(new_params, state, eval_inputs, targets_template, eval_forcings)
print("Predictions Finetuned: ", predictions_finetuned.dims.mapping)
# predictions_finetuned.to_netcdf("/Datastorage/saptarishi.dhanuka_asp25/predictions_finetuned.nc")


# In[13]:


predictions_old.to_netcdf("/Datastorage/saptarishi.dhanuka_asp25/predictions_old_india_jan2016.nc")
predictions_finetuned.to_netcdf("/Datastorage/saptarishi.dhanuka_asp25/predictions_finetuned_india2015_full_jan2016.nc")


# In[ ]:


# import xarray as xr 
# import numpy as np

# predictions_old = xr.open_dataset("/Datastorage/saptarishi.dhanuka_asp25/predictions_old.nc")
# predictions_finetuned = xr.open_dataset("/Datastorage/saptarishi.dhanuka_asp25/predictions_finetuned.nc")
# predictions_old


# In[14]:


import cartopy.crs as ccrs

# Extract the variable of interest (e.g., '2m_temperature')
variable_name = '2m_temperature'
temperature_pred_old = predictions_old[variable_name]
temperature_pred_finetuned = predictions_finetuned[variable_name]
temperature_targets = eval_targets[variable_name]


# In[20]:


temperature_targets


# In[27]:


import xarray as xr
import numpy as np
imd_targets = xr.load_dataset("/Datastorage/saptarishi.dhanuka_asp25/_mars-bol-webmars-private-svc-blue-006-7a527896970b09a4fc90fa37bf98d3ff-ax1w8d.grib", engine='cfgrib')
old_lats = imd_targets['latitude'].values
old_lons = imd_targets['longitude'].values
new_lats = np.arange(-90.0, 90.0 + 1e-8, 1.0)
new_lats = np.flip(new_lats)
new_lons = np.arange(0, 359.75 + 1e-8, 1.0)
imd_targets = imd_targets.interp({'latitude': new_lats, 'longitude': new_lons}, 
                        method='linear',
                        kwargs={'fill_value': None})
imd_targets


# In[ ]:


imd_target_init_15th = imd_targets.sel(time='2022-06-15T00:00:00.000000000')['t2m']
imd_target_init_15th = imd_target_init_15th.rename({'latitude': 'lat', 'longitude': 'lon'})
imd_target_init_15th


# In[16]:


temperature_pred_old.isel(time=0).plot()


# In[15]:


temperature_pred_finetuned.isel(time=0).squeeze().plot()


# In[51]:


temperature_pred_finetuned.isel(time=0)


# In[19]:


latmax = 38
latmin = 6
lonmin = 68
lonmax = 98

temperature_pred_finetuned.isel(time=0).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze().plot()


# In[17]:


diff = temperature_pred_finetuned.isel(time=0).squeeze() - temperature_targets.isel(time=0).squeeze()
diff


# In[18]:


diff.plot()


# In[50]:


fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
extent = [68, 98, 6, 38]  # Focus on India

# Difference between finetuned predictions and evaluation targets
temp_old = temperature_pred_old.isel(time=0).squeeze()
temp_targets = imd_target_init_15th.isel(step=0).squeeze()
difference = temp_old - temp_targets

difference.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
    cbar_kwargs={'label': 'Temperature Difference (K)'})
ax.set_extent(extent)
ax.coastlines()
ax.set_title(f"Base Prediction vs IMD at Time Step {0}")

plt.tight_layout()
plt.show()


# In[51]:


fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
extent = [68, 98, 6, 38]  # Focus on India

# Difference between finetuned predictions and evaluation targets
temp_finetuned = temperature_pred_old.isel(time=0).squeeze()
temp_targets = imd_target_init_15th.isel(step=0).squeeze()
difference = temp_finetuned - temp_targets

difference.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
    cbar_kwargs={'label': 'Temperature Difference (K)'})
ax.set_extent(extent)
ax.coastlines()
ax.set_title(f"Finetuned Prediction vs IMD at Time Step {0}")

plt.tight_layout()
plt.show()


# In[19]:


from tqdm import tqdm
# Define the time steps to visualize
time_steps = list(range(7))  # Adjust as needed

extent = [68, 98, 6, 38]  # Focus on India
latmax = 38
latmin = 6
lonmax = 98
lonmin = 68


# Function to plot predictions over India
def plot_predictions(time_step):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [68, 98, 6, 38]  # Focus on India

    # Old predictions
    ax = axs[0]
    temp_old = temperature_pred_old.isel(time=time_step).squeeze()
    temp_old.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm',
        cbar_kwargs={'label': 'Temperature (K)'})
    ax.set_extent(extent)
    ax.coastlines()
    ax.set_title(f"Old Predictions at Time Step {time_step}")

    # Finetuned predictions
    ax = axs[1]
    temp_finetuned = temperature_pred_finetuned.isel(time=time_step).squeeze()
    temp_finetuned.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm',
        cbar_kwargs={'label': 'Temperature (K)'})
    ax.set_extent(extent)
    ax.coastlines()
    ax.set_title(f"Finetuned Predictions at Time Step {time_step}")

    # Difference between finetuned and old predictions
    ax = axs[2]
    difference = temp_finetuned - temp_old
    difference.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
        cbar_kwargs={'label': 'Temperature Difference (K)'})
    ax.set_extent(extent)
    ax.coastlines()
    ax.set_title(f"Difference at Time Step {time_step}")

    plt.tight_layout()
    plt.savefig(f'plots/training/predictions_comparison{time_step}.png')
    plt.close()


# Function to plot difference between finetuned predictions and ground truth
def finetuned_diff_imd(time_step):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [68, 98, 6, 38]  # Focus on India

    # Difference between finetuned predictions and evaluation targets
    temp_finetuned = temperature_pred_finetuned.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    temp_targets = imd_target_init_15th.isel(step=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    difference = temp_finetuned - temp_targets

    difference.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
        cbar_kwargs={'label': 'Temperature Difference (K)'})
    ax.set_extent(extent)
    ax.coastlines()
    ax.set_title(f"Finetuned Prediction vs IMD at Time Step {time_step}")

    plt.tight_layout()
    plt.savefig(f'plots/training/difference_finetuned{time_step}_imd.png')
    plt.close()

    return (difference)



def plot_difference_with_targets_fine(time_step):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [68, 98, 6, 38]  # Focus on India

    # Difference between finetuned predictions and evaluation targets
    temp_finetuned = temperature_pred_finetuned.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    temp_targets = temperature_targets.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    difference = temp_finetuned - temp_targets

    difference.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
        cbar_kwargs={'label': 'Temperature Difference (K)'})
    ax.set_extent(extent)
    ax.coastlines()
    ax.set_title(f"Finetuned Prediction vs Ground Truth at Time Step {time_step}")

    plt.tight_layout()
    plt.savefig(f'plots/training/difference_finetuned{time_step}.png')
    plt.close()

    return (difference)



def plot_difference_with_targets_base(time_step):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [68, 98, 6, 38]  # Focus on India

    # Difference between finetuned predictions and evaluation targets
    temp_base = temperature_pred_old.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    temp_targets = temperature_targets.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    difference = temp_base - temp_targets

    difference.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
        cbar_kwargs={'label': 'Temperature Difference (K)'})
    ax.set_extent(extent)
    ax.coastlines()
    ax.set_title(f"Finetuned Prediction vs Ground Truth at Time Step {time_step}")

    plt.tight_layout()
    plt.savefig(f'plots/training/difference_finetuned{time_step}.png')
    plt.close()

    return (difference)




# Function to plot difference between base predictions and ground truth
def base_diff_imd(time_step):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [68, 98, 6, 38]  # Focus on India

    # Difference between finetuned predictions and evaluation targets
    temp_old = temperature_pred_old.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    temp_targets = imd_target_init_15th.isel(step=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    difference = temp_old - temp_targets

    difference.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
        cbar_kwargs={'label': 'Temperature Difference (K)'})
    ax.set_extent(extent)
    ax.coastlines()
    ax.set_title(f"Base Prediction vs IMD at Time Step {time_step}")

    plt.tight_layout()
    plt.savefig(f'plots/training/difference_base{time_step}_imd.png')
    plt.close()

    return (difference)


# Function to plot difference between base predictions and ground truth
def plot_diff_imd_era5(time_step):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [68, 98, 6, 38]  # Focus on India

    # Difference between finetuned predictions and evaluation targets
    imd_pred = imd_target_init_15th.isel(step=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    temp_targets = temperature_targets.isel(time=time_step).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)).squeeze()
    difference = imd_pred - temp_targets

    # difference.plot.pcolormesh(
    #     ax=ax, transform=ccrs.PlateCarree(), cmap='bwr',
    #     cbar_kwargs={'label': 'Temperature Difference (K)'})
    # ax.set_extent(extent)
    # ax.coastlines()
    # ax.set_title(f"ERA5 vs IMD at Time Step {time_step}")

    # plt.tight_layout()
    # plt.savefig(f'plots/training/difference_era5{time_step}_imd.png')
    # plt.close()

    return (difference)



# Plot differences with ground truth for each time step
differences_finetuned = []
differences_base = []
differences_era5_imd = []
for time_step in tqdm(time_steps):
    differences_finetuned.append(plot_difference_with_targets_fine(time_step))
    differences_base.append(plot_difference_with_targets_base(time_step))
    # differences_era5_imd.append(plot_diff_imd_era5(time_step))

# print(differences_finetuned)
# print(differences_base)


def compute_mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def compute_mse_diffs(diff):
    return (diff ** 2).mean(dim=["lat", "lon"]).values

# Compute the MSE for each time step and aggregate
mse_finetuned = []
mse_base = []
mse_imd_era5 = []

# need to define the extent for india and then calculate metric


# In[20]:


# Compute the mean absolute difference for each timestep and append to the lists

mse_finetuned = []
mse_base = []
for i in tqdm(range(len(differences_finetuned))):
    mse_diff_finetuned = compute_mse_diffs(differences_finetuned[i])
    mse_diff_base = compute_mse_diffs(differences_base[i])
    # mse_diff_imd_era5 = compute_mse_diffs(differences_era5_imd[i])
    
    mse_finetuned.append(mse_diff_finetuned)
    mse_base.append(mse_diff_base)
    # mse_imd_era5.append(mse_diff_imd_era5)


# In[40]:


# Compute the mean absolute difference for each timestep and append to the lists

mse_finetuned = []
mse_base = []
mse_imd_era5 = []
for i in tqdm(range(len(differences_finetuned))):
    mean_abs_diff_finetuned = abs(differences_finetuned[i]).mean(dim=["lat", "lon"]).values
    mean_abs_diff_base = abs(differences_base[i]).mean(dim=["lat", "lon"]).values
    mean_abs_diff_imd_era5 = abs(differences_era5_imd[i]).mean(dim=["lat", "lon"]).values
    
    mse_finetuned.append(mean_abs_diff_finetuned)
    mse_base.append(mean_abs_diff_base)
    mse_imd_era5.append(mean_abs_diff_imd_era5)


# In[21]:


mse_finetuned


# In[22]:


mse_base


# In[41]:


mse_imd_era5


# In[42]:


import matplotlib.pyplot as plt

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(time_steps, mse_imd_era5, label='Mean IMD ERA5', marker='o')


# Add labels, title, and legend
plt.xlabel('Time Step')
plt.ylabel('Mean ABS Diff')
plt.title('Mean ABS Diff IMD vs ERA5')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# In[23]:


import matplotlib.pyplot as plt

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(time_steps, mse_base, label='MSE Base', marker='o')
plt.plot(time_steps, mse_finetuned, label='MSE Finetuned', marker='o')
# plt.plot(time_steps, mse_imd_era5, label='MSE IMD ERA5', marker='x')

# Add labels, title, and legend
plt.xlabel('Time Step')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison: Base vs Finetuned vs IMD relative to ERA5')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# In[ ]:


# predictions_old['total_precipitation_6hr']


# In[ ]:


# 


# In[ ]:


# mse_base


# In[ ]:


# list(mse_finetuned)


# In[11]:



# latmin = 8
# latmax = 37
# lonmin = 68
# lonmax = 97

# temperature_pred_finetuned


# In[ ]:


# temperature_pred_finetuned.isel(time=1).sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax))


# In[ ]:






# print("Starting subsetting:")
# for time_step in time_steps:    
#     temp_finetuned = temperature_pred_finetuned.isel(time=time_step).sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
#     if time_step == 1:
#         print(temp_finetuned.dims)
#     temp_finetuned = temp_finetuned.squeeze()
    
#     temp_old = temperature_pred_old.isel(time=time_step).sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
#     if time_step == 1:
    
#         print(temp_old.dims)
#     temp_old = temp_old.squeeze()

#     temp_targets = temperature_targets.isel(time=time_step).sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
#     if time_step == 1:
    
#         print(temp_targets.dims)
#     temp_targets = temp_targets.squeeze()

#     mse_finetuned.append(compute_mse(temp_finetuned, temp_targets))
#     mse_base.append(compute_mse(temp_old, temp_targets))

# # Compute the overall loss by averaging the MSE over all time steps
# overall_loss_finetuned = np.mean(mse_finetuned)
# overall_loss_base = np.mean(mse_base)

# print(f"Overall Loss (Finetuned Predictions): {overall_loss_finetuned}")
# print(mse_finetuned)
# print(f"Overall Loss (Base Predictions): {overall_loss_base}")
# print(mse_base)


# logger.info("Finished inference")

# # can add evaluation and comparison code here after testing everything for a few timesteps




# # </finetuning_cleaned.py>

