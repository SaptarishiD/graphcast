import weatherbench2
import xarray as xr
import zarr
import time

obs_path_new = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr/"
full_obs = xr.open_zarr(obs_path_new)
full_obs2022 = full_obs.sel(time=slice('2022-01-01', '2022-12-31'))
print(full_obs2022.nbytes)
selected_our_vars = ['2m_temperature','total_precipitation_6hr']
full_obs2022_gc_vars = full_obs2022[selected_our_vars]

print("Saving to disk...")
tik = time.time()
full_obs2022_gc_vars.to_zarr("/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_temp_ppt2022_wb_0.25.zarr")
tok = time.time()
print("Time taken to save to disk: ", tok - tik)
print("Done!")