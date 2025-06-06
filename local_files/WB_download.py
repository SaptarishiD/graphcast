import weatherbench2
import xarray as xr
import zarr
import time
import numpy as np

obs_path_new = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
full_obs = xr.open_zarr(obs_path_new)
full_obs2022 = full_obs.sel(time=slice('2015-06-01', '2015-09-30'))
print(full_obs2022.nbytes)
selected_our_vars = ['2m_temperature','total_precipitation_6hr']

remaining_vars = ['10m_u_component_of_wind',
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




full_obs2022_gc_input_vars = full_obs2022[remaining_vars]


print(full_obs2022_gc_input_vars.nbytes)


latitude_min = float(full_obs2022_gc_input_vars.latitude.min().values)
latitude_max = float(full_obs2022_gc_input_vars.latitude.max().values)
longitude_min = float(full_obs2022_gc_input_vars.longitude.min().values)
longitude_max = float(full_obs2022_gc_input_vars.longitude.max().values)

# Create new latitudeitude and longitudegitude arrays at 1.0° resolution.
latitude_new = np.arange(latitude_min, latitude_max + 1.0, 1.0)
longitude_new = np.arange(longitude_min, longitude_max + 1.0, 1.0)

# Step 3: Interpolatitudee the dataset to the new grid.
# xarray.interp performs 1-D interpolatitudeion alongitudeg the specified coordinates.
full_obs2022_gc_input_vars_interp = full_obs2022_gc_input_vars.interp(latitude=latitude_new, longitude=longitude_new)


print(full_obs2022_gc_input_vars_interp.nbytes)

print("Interpolated to 1.0 degree grid")

print("Saving to disk...")
tik = time.time()
full_obs2022_gc_input_vars_interp.to_zarr("/Datastorage/saptarishi.dhanuka_asp25/era5_data/wb_era5_jun_sept_2015_temp_ppt.zarr")
tok = time.time()
print("Time taken to save to disk: ", tok - tik)
print("Done!")