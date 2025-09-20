# <eval_forecast.py>
"""
python /home/saptarishi.dhanuka_asp25/weather/graphcast_dir/graphcast/local_files/eval_forecast.py \ 
--eval_start "2024-08-01" \ 
--eval_end "2024-09-15" \ 
--eval_dataset_choice "imerg" \ 
--vars_to_eval "total_precipitation_6hr" \ 
--params_path_new1 "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2024-06-01_2024-09-15_FORECAST28.npz" \ 
--params_path_new2 "/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2024-06-01_2024-07-30_FORECAST28.npz"
"""

"""
Complete evaluation of forecast against different datasets
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

import jax
import optax

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from graphcast import checkpoint, data_utils, rollout, graphcast, normalization
import save_params_utils
import setup_jax_functions
from plotting import scale, select, plot_data, save_animation, save_static_plot, compute_difference_with_targets_sims, plot_sample_from_ds
from metrics import compute_rmse, compute_mae, compute_bias, compute_acc
from utils import regrid_hres_fine_to_coarse, generate_sample_era5_dataset, grads_fn, parse_args, process_to_graphcast_format, compute_mse, compute_mse_diffs


sys.path.append('/home/saptarishi.dhanuka_asp25/weather/graphcast_dir/gc_dist')
import trainer.dataloader
from dist_utils import construct_era5_imerg

args = parse_args()

eval_start = args.eval_start
eval_end = args.eval_end
dataset_choice = args.eval_dataset_choice
eval_vars = args.vars_to_eval
apath = args.eval_data_path
params_path_old = args.params_path_old
params_path_new1 = args.params_path_new1
params_path_new2 = args.params_path_new2
norms_dir = args.norms_dir
plots_dir = args.plots_dir
latmin, latmax, lonmin, lonmax = args.latmin, args.latmax, args.lonmin, args.lonmax
plot_timesteps = args.plot_timesteps
output_pred_old_dir = args.output_pred_old_dir
output_pred_finetuned_dir = args.output_pred_finetuned_dir



if dataset_choice == "imerg":
    apath = '/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/'
    dbase,_ = trainer.dataloader.open_databases(apath,None) # Note no need for a separate verification dbase
    dbase = construct_era5_imerg(dbase)
    eval_time_ds = dbase.sel(time=slice(eval_start, eval_end))
    del dbase


select_time_eval = process_to_graphcast_format(eval_time_ds)
print(f"Time range: {select_time_eval.time.values[0]} - {select_time_eval.time.values[-1]}")
data_len = len(select_time_eval.time.values)

select_time_evals = []
for time_slice in range(0, data_len, 28):
    select_time_evals.append(select_time_eval.isel(time=slice(time_slice, time_slice+28)))


print('Sims initialisations')
for ds in select_time_evals:
    print(ds.datetime.values[0][0])


sims = len(select_time_evals)
print(f"Num sims: {sims}")



eval_trial_batches = []
from dask.diagnostics import ProgressBar
for i in tqdm(range(sims-1), desc="Computing sims"):
    with ProgressBar():
        eval_trial_batches.append(select_time_evals[i].load())


with open(params_path_old, 'rb') as f:
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

f = open(params_path_new1, "rb")
new_ckpt = checkpoint.load(f, graphcast.CheckPoint)
new_params1 = new_ckpt.params
f.close()


f = open(params_path_new2, "rb")
new_ckpt = checkpoint.load(f, graphcast.CheckPoint)
new_params2 = new_ckpt.params
f.close()




params_keys_list = list(new_params1.keys())
any_not_nan = any(not np.isnan(x) for x in list(new_params1[f'{params_keys_list[0]}'].values())[1])
if not any_not_nan:
    print("All values are NaN in params!")
    exit()

jax.config.update("jax_enable_x64", True)

def compute_mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def compute_mse_diffs(diff):
    return (diff ** 2).mean(dim=["lat", "lon"]).values




# Combine list of xarray Datasets into a single xarray Dataset via summing
def accumulate_diff(diff_list, use_abs=False):
    accumulated = None
    for ds in diff_list:
        var = ds
        if use_abs:
            var = np.abs(var)
        if accumulated is None:
            accumulated = var.copy(deep=True)
        else:
            accumulated += var
    return accumulated

# Plotting function
def plot_accumulated_diff(data_array, title, simnum, absolute=True, cmap='RdBu', vmin=None, vmax=None):
    # Set fixed vmin and vmax based on `absolute` flag, if not provided explicitly
    if absolute:
        vmin = 0 if vmin is None else vmin
        vmax = 0.6 if vmax is None else vmax
    else:
        vmin = -0.3 if vmin is None else vmin
        vmax = 0.3 if vmax is None else vmax

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    im = data_array.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': 'Accumulated Error (mm)'}
    )

    ax.coastlines()
    ax.set_title(title)
    ax.set_extent([65, 100, 5, 40])  # Set extent over India

    plt.savefig(f'./output/20240601_20240730_train/shapefile_accumulated_diff_map_{title}_sim_{simnum}')
    plt.close()


# def plot_accumulated_diff_side_by_side(truth, pred1, pred2, data_arrays, titles, simnum, absolute=True, cmap='RdBu', vmin=None, vmax=None):
#     """
#     Plot 3 data arrays side by side on a map with fixed colorbar range depending on the `absolute` flag.

#     Parameters:
#         data_arrays (list of xarray.DataArray): Three data arrays to plot.
#         titles (list of str): Titles for each subplot.
#         simnum (int): Simulation number to use in filename.
#         absolute (bool): Whether to fix color range to absolute values (0 to 0.6) or relative (-0.3 to 0.3).
#         cmap (str): Colormap.
#         vmin (float): Minimum value for colorbar (overridden by absolute flag if None).
#         vmax (float): Maximum value for colorbar (overridden by absolute flag if None).
#     """
#     # Set fixed vmin/vmax depending on 'absolute'

#     lon_min, lon_max = 65, 95
#     lat_min, lat_max = 5, 40

#     assert(truth.lat.values[0] - truth.lat.values[1] < 0)
#     assert(pred1.lat.values[0] - pred1.lat.values[1] < 0)
#     assert(pred2.lat.values[0] - pred2.lat.values[1] < 0)

#     # Slice to Indian region
#     truth_india  = truth.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))['total_precipitation_6hr'].sum(dim="time").squeeze(dim='batch')
#     pred1_india  = pred1.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))['total_precipitation_6hr'].sum(dim="time").squeeze(dim='batch')
#     pred2_india  = pred2.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))['total_precipitation_6hr'].sum(dim="time").squeeze(dim='batch')


#     # Set vmin and vmax based on absolute flag (for diff maps only)
#     if absolute:
#         vmin_diff = 0 if vmin is None else vmin
#         vmax_diff = 0.7 if vmax is None else vmax
#     else:
#         vmin_diff = -0.4 if vmin is None else vmin
#         vmax_diff = 0.4 if vmax is None else vmax

#     # Setup figure with 2 rows and 3 columns
#     fig, axes = plt.subplots(2, 3, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})


#     data_arrays = [truth_india, pred1_india, pred2_india, data_arrays[0], data_arrays[1], data_arrays[2]]
#     titles = [
#         "Ground Truth",
#         "Prediction 1",
#         "Prediction 2",
#         "Pred1 - Truth",
#         "Pred2 - Truth",
#         "Pred1 - Pred2"
#     ]

#     for i, ax in enumerate(axes.flat):
#         da = data_arrays[i]

#         # Only apply fixed vmin/vmax to difference plots (last 3)
#         if i < 3:
#             im = da.plot.pcolormesh(
#                 ax=ax,
#                 transform=ccrs.PlateCarree(),
#                 cmap=cmap,
#                 add_colorbar=(i == 2),  # Add colorbar only to last of top row
#                 cbar_kwargs={'label': 'Precipitation (mm)'} if i == 2 else {}
#             )
#         else:
#             im = da.plot.pcolormesh(
#                 ax=ax,
#                 transform=ccrs.PlateCarree(),
#                 cmap=cmap,
#                 vmin=vmin_diff,
#                 vmax=vmax_diff,
#                 add_colorbar=(i == 5),  # Add colorbar only to last of bottom row
#                 cbar_kwargs={'label': 'Difference (mm)'} if i == 5 else {}
#             )

#         ax.set_title(titles[i], fontsize=12)
#         ax.set_extent([lon_min, lon_max, lat_min, lat_max])
#         ax.coastlines()

#     plt.tight_layout()
#     plt.savefig(f'./output/train_compare/sim_{simnum}_{"absolute" if absolute else "diff"}_6panel_plot.png')
#     plt.close()

#     # if absolute:
#     #     vmin = 0 if vmin is None else vmin
#     #     vmax = 0.7 if vmax is None else vmax
#     # else:
#     #     vmin = -0.4 if vmin is None else vmin
#     #     vmax = 0.4 if vmax is None else vmax

#     # # Set up the figure with 3 subplots
#     # fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})

#     # for i in range(3):
#     #     ax = axes[i]
#     #     im = data_arrays[i].plot.pcolormesh(
#     #         ax=ax,
#     #         transform=ccrs.PlateCarree(),
#     #         cmap=cmap,
#     #         vmin=vmin,
#     #         vmax=vmax,
#     #         add_colorbar=(i == 2),  # Only add colorbar to the last subplot
#     #         cbar_kwargs={'label': 'Accumulated Error (mm)'} if i == 2 else {}
#     #     )
#     #     ax.set_title(titles[i])
#     #     ax.set_extent([65, 100, 5, 40])  # over India
#     #     ax.coastlines()

#     # # Save the figure
#     # plt.tight_layout()
#     # plt.savefig(f'./output/train_compare/{absolute}_accumulated_diff_sim_{simnum}_side_by_side.png')
#     # plt.close()


def plot_comparison_and_diffs(truth, pred1, pred2, old, simnum, 
                              cmap_precip='Blues', cmap_diff='RdBu', 
                              lon_min=65, lon_max=95, lat_min=5, lat_max=40):
    """
    Plot precipitation and difference maps for:
      • Ground truth
      • Prediction 1
      • Prediction 2
      • Difference: pred1 - truth
      • Difference: pred2 - truth
      • Difference: old - truth
      • Difference: pred1 - old

    Arranged in a 2×4 grid (only 7 panels used).
    """

    # 1. Slice to India and sum over time & batch
    def prep(da):
        return (da
                .sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                ['total_precipitation_6hr']
                .sum(dim="time")
                .squeeze(dim='batch'))

    t = prep(truth)
    p1 = prep(pred1)
    p2 = prep(pred2)
    po = prep(old)

    # 2. Compute difference fields
    d1 = p1 - t
    d2 = p2 - t
    d3 = po - t
    d4 = p1 - po

    # 3. Determine a common vmax for precipitation plots
    vmax_precip = float(max(t.max(), p1.max(), p2.max()))

    # 4. Determine symmetric vmin/vmax for diffs
    vmax_diff = max(abs(d1).max(), abs(d2).max(), abs(d3).max(), abs(d4).max())
    vlim = float(vmax_diff)

    # 5. Set up figure: 2 rows × 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(24, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    panels = [
        (t,  'Ground Truth',       cmap_precip, 0,        vmax_precip, {'label':'Precip (mm)'}),
        (p1, 'Prediction 1',       cmap_precip, 0,        vmax_precip, {}),
        (p2, 'Prediction 2',       cmap_precip, 0,        vmax_precip, {}),
        (d1, 'Pred1 − Truth',      cmap_diff,  -vlim,     vlim,        {}),
        (d2, 'Pred2 − Truth',      cmap_diff,  -vlim,     vlim,        {}),
        (d3, 'Old − Truth',        cmap_diff,  -vlim,     vlim,        {}),
        (d4, 'Pred1 − Old',        cmap_diff,  -vlim,     vlim,        {'label':'Difference (mm)'}),
    ]

    for idx, (da, title, cmap, vmin, vmax, cbar_kwargs) in enumerate(panels):
        ax = axes.flat[idx]
        im = da.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            add_colorbar=False
        )
        ax.set_title(title, fontsize=12)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.coastlines()

        # only the last panel gets a colorbar, with its own label
        if 'label' in cbar_kwargs:
            cb = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.03)
            cb.set_label(cbar_kwargs['label'])

    # turn off the unused 8th subplot
    axes.flat[-1].axis('off')

    plt.tight_layout()
    plt.savefig(f'./output/train_compare/shapefile_sim_{simnum}_comparison_7panel.png')
    plt.close()

def plot_accumulated_diff_side_by_side(truth, pred1, pred2, data_arrays, titles, simnum,
                                       absolute=True, cmap='RdBu', vmin=None, vmax=None):
    """
    Plot 3 precipitation fields and 4 difference fields side by side on a map.

    Parameters:
        truth, pred1, pred2        : xarray.Dataset of 'total_precipitation_6hr'
        data_arrays (list of xarray.DataArray): [pred1−truth, pred2−truth, old−truth]
        titles      (list of str)  : titles for those 3 diff panels, e.g. ["Raw Finetuned 1", "Finetuned 2", "Old"]
        simnum      (int)          : simulation number for filename
        absolute    (bool)         : ignored now (all diffs get symmetric range)
        cmap        (str)          : colormap for diffs (we’ll override precip→Blues)
        vmin, vmax  (float|None)   : ignored now (we derive fixed ranges)
    """
    # India bounds
    lon_min, lon_max = 65, 95
    lat_min, lat_max = 5, 40

    # --- 1. Prep precip arrays ---
    def prep_precip(ds):
        return (ds
                .sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                ['total_precipitation_6hr']
                .sum(dim="time")
                .squeeze(dim="batch"))

    t = prep_precip(truth)
    p1 = prep_precip(pred1)
    p2 = prep_precip(pred2)

    # get common vmax for precip
    vmax_precip = float(max(t.max(), p1.max(), p2.max()))

    # --- 2. Prep diff arrays (already time-summed) ---
    da1 = data_arrays[0].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    da2 = data_arrays[1].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    da3 = data_arrays[2].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # compute pred1−old = (pred1−truth) - (old−truth)
    da4 = da1 - da3

    # symmetric vlim for all diffs
    limit = float(max(abs(da1).max(), abs(da2).max(), abs(da3).max(), abs(da4).max()))

    # --- 3. Build panels & titles ---
    top = [
        (t,  "Ground Truth",   'Blues', 0,         vmax_precip, {'label': 'Precip (mm)'}),
        (p1, "Prediction 1",   'Blues', 0,         vmax_precip, {}),
        (p2, "Prediction 2",   'Blues', 0,         vmax_precip, {}),
    ]
    bot_titles = titles + ["Prediction 1 − Old"]
    bot = [
        (da1, bot_titles[0], cmap, -limit, limit, {}),
        (da2, bot_titles[1], cmap, -limit, limit, {}),
        (da3, bot_titles[2], cmap, -limit, limit, {}),
        (da4, bot_titles[3], cmap, -limit, limit, {'label': 'Difference (mm)'}),
    ]

    panels = top + bot

    # --- 4. Plot ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    for idx, (da, title, cm, mn, mx, cbar_kw) in enumerate(panels):
        ax = axes.flat[idx]
        im = da.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cm,
            vmin=mn,
            vmax=mx,
            add_colorbar=False
        )
        ax.set_title(title, fontsize=12)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.coastlines()
        if 'label' in cbar_kw:
            cb = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.03)
            cb.set_label(cbar_kw['label'])

    # hide unused panel
    axes.flat[-1].axis('off')

    plt.tight_layout()
    plt.savefig(f'./output/train_compare/shapefile_sim_{simnum}_comparison_7panel.png')
    plt.close()


















import glob
file_list = glob.glob("/Datastorage/saptarishi.dhanuka_asp25/forecasts_hres/raw_hres/2024/*.grib")

# Helper function to extract (month, day) and convert to datetime
def extract_month_day(path):
    parts = path.split('_')
    # print(parts)
    year = 2024  # fixed, or extract if you want dynamic
    month = int(parts[5])
    day = int(parts[6])
    return datetime(year, month, day)

# Sort based on extracted datetime
sorted_paths = sorted(file_list, key=extract_month_day)



for i, hres_file in tqdm(enumerate(sorted_paths[5:11]), desc="Sims"):

    print(f"Using hres file: {hres_file}")

    if i > sims - 1:
        break
    if i > len(eval_trial_batches) - 1:
        break

    time_len = len(eval_trial_batches[i].time.values)
    eval_sim_data = eval_trial_batches[i].isel(time=slice(0, time_len-1))
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    eval_sim_data, target_lead_times=slice("6h", "100h"),
    **dataclasses.asdict(task_config))
    forecast_len = len(eval_targets.time.values)

    assert eval_trial_batches[i].sizes["time"] >= 3

    print(f"Eval batch time dimensions: {eval_trial_batches[i].sizes['time']}")

    task_config_dict =  dataclasses.asdict(task_config)

    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)

    targets_template = eval_targets * np.nan

    print("Running old params")
    predictions_old = run_model(params, state, eval_inputs, targets_template, eval_forcings)

    print("Predictions Old: ", predictions_old.dims.mapping)

    print("Running new params")

    predictions_finetuned1 = run_model(new_params1, state, eval_inputs, targets_template, eval_forcings)
    print("Predictions Finetuned1: ", predictions_finetuned1.dims.mapping)

    predictions_finetuned2 = run_model(new_params2, state, eval_inputs, targets_template, eval_forcings)

    print("Predictions Finetuned2: ", predictions_finetuned2.dims.mapping)

    hres = xr.open_dataset(hres_file)
    regridded_hres = regrid_hres_fine_to_coarse(hres, variable='tp', coarse_resolution=1.0)


    diff_fine1, diff_fine2, diff_old, diff_hres = [], [], [], []
    for time_step in tqdm(range(forecast_len), desc='Diff compute over timesteps'):
        diff_fine1.append(compute_difference_with_targets_sims(predictions_finetuned1,eval_targets,time_step, 'total_precipitation_6hr', plot=True))
        diff_fine2.append(compute_difference_with_targets_sims(predictions_finetuned2,eval_targets,time_step, 'total_precipitation_6hr', plot=True))
        diff_old.append(compute_difference_with_targets_sims(predictions_old,eval_targets,time_step, 'total_precipitation_6hr', plot=True))
        diff_hres.append(regridded_hres.isel(step=time_step) - eval_targets['total_precipitation_6hr'].isel(time=time_step).squeeze(dim='batch'))


        # ---- Accumulate and plot raw diffs ----

    accum_fine_raw1 = accumulate_diff(diff_fine1, use_abs=False)
    accum_fine_raw2 = accumulate_diff(diff_fine2, use_abs=False)
    accum_old_raw = accumulate_diff(diff_old, use_abs=False)
    accum_hres = accumulate_diff(diff_hres, use_abs=False)

    # plot_accumulated_diff(accum_fine_raw1, 'Accumulated Raw Difference (Fine-Tuned 1)', simnum=i, absolute=False)
    # plot_accumulated_diff(accum_fine_raw2, 'Accumulated Raw Difference (Fine-Tuned 2)', simnum=i, absolute=False)
    # plot_accumulated_diff(accum_old_raw, 'Accumulated Raw Difference (Old)', simnum=i, absolute=False)

    # plot_accumulated_diff_side_by_side(eval_targets, predictions_finetuned1, predictions_finetuned2, data_arrays=[accum_fine_raw1, accum_fine_raw2, accum_old_raw], titles=["Raw Finetuned 1", "Finetuned 2", "Old"],simnum=i, absolute=False )



    # ---- Accumulate and plot absolute diffs ----

    accum_fine_abs1 = accumulate_diff(diff_fine1, use_abs=True)
    accum_fine_abs2 = accumulate_diff(diff_fine2, use_abs=True)

    accum_old_abs = accumulate_diff(diff_old, use_abs=True)


    # TODO Change all the scales in all the plots to be of fixed ranges 

    # plot_accumulated_diff(accum_fine_abs1, 'Accumulated Absolute Difference (Fine-Tuned 1)',simnum=i, cmap='Reds', vmin=0)
    # plot_accumulated_diff(accum_fine_abs2, 'Accumulated Absolute Difference (Fine-Tuned 2)',simnum=i, cmap='Reds', vmin=0)

    # plot_accumulated_diff(accum_old_abs, 'Accumulated Absolute Difference (Old)',simnum=i, cmap='Reds', vmin=0)
    print("Plotting side by side")

    plot_accumulated_diff_side_by_side(eval_targets, predictions_finetuned1, predictions_finetuned2, data_arrays=[accum_fine_abs1, accum_fine_abs2, accum_old_abs], titles=["Absolute Finetuned 1", "Finetuned 2", "Old"],simnum=i )


    # plot_comparison_and_diffs(eval_targets, predictions_finetuned1, predictions_finetuned2, data_arrays=[accum_fine_raw1, accum_fine_raw2, accum_old_raw], titles=["Raw Finetuned 1", "Finetuned 2", "Old"],simnum=i, absolute=False )


    # Compute the MSE for each time step and aggregate
    mse_finetuned1 = []
    mse_finetuned2 = []

    mse_base = []
    mse_hres =[]
    for diff_counter in range(min(len(diff_fine1), len(diff_hres))):
        mse_diff_fine1tuned = compute_mse_diffs(diff_fine1[diff_counter])
        mse_diff_fine2tuned = compute_mse_diffs(diff_fine2[diff_counter])
        
        mse_diff_base = compute_mse_diffs(diff_old[diff_counter])

        mse_diff_hres = compute_mse_diffs(diff_hres[diff_counter])
        # mse_diff_imd_era5 = compute_mse_diffs(differences_era5_imd[diff_counter])
        
        mse_finetuned1.append(mse_diff_fine1tuned)
        mse_finetuned2.append(mse_diff_fine2tuned)
        mse_base.append(mse_diff_base)
        mse_hres.append(mse_diff_hres)
        # mse_imd_era5.append(mse_diff_imd_era5)

    print(mse_base)
    print(f"Mse finetuned 1")
    print(mse_finetuned1)
    print(f"Mse finetuned 2")
    print(mse_finetuned2)
    print(f"Mse hres")
    # print(mse_hres)
    mse_hres_new = [np.array(x.item(), dtype=np.float32) for x in mse_hres]
    print(mse_hres_new)

    print(f"\n Forecast length is: {forecast_len} \n")


    # Plot the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(forecast_len)), mse_base, label='MSE Base', marker='o')
    plt.plot(list(range(forecast_len)), mse_finetuned1, label='MSE Finetuned 1', marker='o')
    plt.plot(list(range(forecast_len)), mse_finetuned2, label='MSE Finetuned 2', marker='o')

    plt.plot(list(range(forecast_len)), mse_hres_new, label='HRES', marker='o')
    # plt.plot(list(range(4)), mse_imd_era5, label='MSE IMD', marker='o')
    # plt.plot(time_steps, mse_imd_era5, label='MSE IMD ERA5', marker='x')

    # Annotate the first and last points for MSE Base

    # plt.annotate(f"{mse_base[i]:.4f}", (0, mse_base[i]), textcoords="offset points", xytext=(-10, 10), ha='center')
    # plt.annotate(f"{mse_base[-1]:.4f}", (6, mse_base[-1]), textcoords="offset points", xytext=(-10, 10), ha='center')


    # Add labels, title, and legend
    plt.xlabel('Time Step')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'MSE Comparison: Base vs Finetuned relative to IMERG init at {eval_targets.time.values[0]}_{hres_file[-25:-5]} ')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig(f"./output/compare/shapefile_sims_base_finetuned_test_imerg_init_2024_09_01_{eval_trial_batches[i].time.values[0]}_{hres_file[-25:-5]}_{i}")
    plt.clf()
    # diff_imd.append(plot_diff_imd_era5_sims(imd_6h_converted,eval_targets,time_step, 'total_precipitation_6hr'))







"""

time_len = len(eval_trial_batch.time.values)
eval_sim_data = eval_trial_batch.isel(time=slice(0, time_len-1))
eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
eval_sim_data, target_lead_times=slice("6h", "168h"),
**dataclasses.asdict(task_config))

assert eval_trial_batch.sizes["time"] >= 3

print(f"Eval batch time dimensions: {eval_trial_batch.sizes['time']}")

task_config_dict =  dataclasses.asdict(task_config)

print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

targets_template = eval_targets * np.nan

print("Running old params")
predictions_old = run_model(params, state, eval_inputs, targets_template, eval_forcings)

print("Predictions Old: ", predictions_old.dims.mapping)

print("Running new params")

predictions_finetuned1 = run_model(new_params1, state, eval_inputs, targets_template, eval_forcings)
print("Predictions Finetuned: ", predictions_finetuned1.dims.mapping)



forecast_len = len(eval_targets.time.values)

diff_fine1, diff_old = [], []
for time_step in tqdm(range(forecast_len)):
    diff_fine1.append(compute_difference_with_targets_sims(predictions_finetuned1,eval_targets,time_step, 'total_precipitation_6hr'))
    diff_old.append(compute_difference_with_targets_sims(predictions_old,eval_targets,time_step, 'total_precipitation_6hr'))
    # diff_imd.append(plot_diff_imd_era5_sims(imd_6h_converted,eval_targets,time_step, 'total_precipitation_6hr'))


mse_finetuned1 = []
mse_base = []
mse_imd_era5 = []
for diff_counter in range(len(diff_fine1)):
    mse_diff_fine1tuned = compute_mse_diffs(diff_fine1[diff_counter])
    mse_diff_base = compute_mse_diffs(diff_old[diff_counter])
    # mse_diff_imd_era5 = compute_mse_diffs(differences_era5_imd[diff_counter])

    mse_finetuned1.append(mse_diff_fine1tuned)
    mse_base.append(mse_diff_base)
    # mse_imd_era5.append(mse_diff_imd_era5)

# print(mse_base)
# print(mse_finetuned1)

plt.figure(figsize=(10, 6))
plt.plot(list(range(forecast_len)), mse_base, label='MSE Base', marker='o')
plt.plot(list(range(forecast_len)), mse_finetuned1, label='MSE Finetuned', marker='o')
# plt.plot(list(range(4)), mse_imd_era5, label='MSE IMD', marker='o')
# plt.plot(time_steps, mse_imd_era5, label='MSE IMD ERA5', marker='x')

plt.xlabel('Time Step')
plt.ylabel('Mean Squared Error (MSE)')
plt.title(f'MSE Comparison: Init at {eval_targets.time.values[0]} ')
plt.legend()
plt.grid(True)

plt.savefig('output/mse_comp.jpeg')
print(sum(mse_base))
print(sum(mse_finetuned1))


"""
# </eval_forecast.py>