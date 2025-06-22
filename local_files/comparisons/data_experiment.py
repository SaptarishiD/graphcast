import xarray as xr

regridded_2022_era5_wb = xr.open_dataset("./era5_temp_ppt2022_wb_1.0_all_vars.nc")


regridded_2022_era5_wb = regridded_2022_era5_wb.drop_sel(longitude=360)


regridded_2022_era5_wb.to_netcdf("./era5_temp_ppt2022_wb_1.0_all_vars.nc")