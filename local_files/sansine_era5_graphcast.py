import xarray
import xesmf
import dask
import numpy as np
import pandas as pd
from processing_outputs import generating_dates_ecmwf


def data_graphcast(default_settings : bool,
                   start : str,
                   end : str,
                   year : int,
                   example_batch : xarray.Dataset) -> xarray.Dataset:
    
    """Creating new batches of data for graphcast inference and fine-tuning, 
    Data resolution is coarsened to 1° with xesmf package with bilinear interpolation.
    
  Args:
    Defaults settings : boolean to know whether or not to use defaults settings for graphcast.

    start : string of date with format %YYYY-%mm-%dd represents start date to extract data

    end : string of date with format %YYYY-%mm-%dd represents end date to extract data

    example_batch : example_batch given in demo but with a 1° resolution needed for regridding 

  Returns:
    batches : with shape (time: ?, step: 47 isobaricInhPa: 10, latitude: 121, longitude: 240)

"""
    #downloading data from weatherbench2
    data = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')

    variables = ['geopotential_at_surface', 'land_sea_mask', '2m_temperature','mean_sea_level_pressure', '10m_v_component_of_wind',\
                 '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential',\
                 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity','specific_humidity']
                 
    
    #Choosing mondays and thursdays (by default)
    #dates = generating_dates_ecmwf(year)
    
    #to follow dates of reforecasts 
    dates = np.load('/home/vsansi01/dates_reforecasts_2020_2021.npy')
    dates = [pd.to_datetime(date).date().strftime('%Y-%m-%d') for date in dates]
    
    if default_settings:
        processed_data = data[variables].sel(time=slice(start,end)).where(data.time.dt.hour.isin([0,6,12]),drop=True)
    else:
        processed_data = data[variables].sel(time=slice(start,end))
        #processed_data = processed_data.where(processed_data.time.dt.dayofyear.isin(dates.dayofyear),drop=True) 

    #regridding to 1° from 0.25° as it is the low memory graphcast that we are using 
    ds_in = processed_data
    ds_out = example_batch
    regridder = xesmf.Regridder(ds_in, ds_out, "bilinear")
    regridded_data = regridder(ds_in,keep_attrs=True)
    
    #creating batches, if default_settings = True normal settings for graphcast
    if default_settings:
        daily_regridded_data = list(regridded_data.groupby('time.dayofyear'))
    else:
    #creating batches of 22 days but only shifted by 7 days
    #88 to change if needed 90 timestamps is approximately equal to 22 days
    #28 timestamps is equal to 7 days 
    #4 timestamps per day
        daily_regridded_data = []
        for i in range(0,regridded_data['time'].size-88,4):#,28):
            data = regridded_data.isel(time=slice(i,-1))
            print(pd.to_datetime(data.time.values[0]).date())
            if pd.to_datetime(data.time.values[0]).date().strftime('%Y-%m-%d') in dates:
                print('Good')
                temp_data = list(data.resample(time='30d'))[0]
                daily_regridded_data.append(temp_data)
                
    daily_regridded_data = [element[1] for element in daily_regridded_data]
    #conserving times for new coordinates in batches
    day_time = [dataset['time'].values for dataset in daily_regridded_data]
    
    #storing timedeltas for conversion of datasets' times to timedeltas format required by graphcast
    timedeltas = []
    for i,date in enumerate(day_time[0]):
        first_date = day_time[0][0] 
        timedeltas.append(day_time[0][i] - first_date)
    
    #sanity check
    assert len(timedeltas) == len(day_time[0])
    
    #building batches
    batches = []
    for dataset,coords in zip(daily_regridded_data,day_time):
        if default_settings:
            dataset['time'] = example_batch['time']
            batches.append(dataset.assign_coords(coords={'datetime':(('batch','time'),coords.reshape(1,-1))}))
        
        else:
            if len(dataset['time']) == len(daily_regridded_data[0]['time']):
                dataset['time'] =  timedeltas
                batches.append(dataset.assign_coords(coords={'datetime':(('batch','time'),coords.reshape(1,-1))}))
            
    with dask.config.set(**{'array.slicing.split_large_chunks':True}):
        return xarray.concat(batches,dim='batch')

if __name__ == "__main__":
    example_batch = xarray.open_dataset('/home/vsansi01/exemple_batch.nc')
    example_batch_ = data_graphcast(default_settings=False,start='2021-01-01',end='2021-12-31',year=2021,example_batch=example_batch)
    print('---> loading data')
    example_batch_ = example_batch_.load()
    example_batch_ = example_batch_.to_netcdf('/home/vsansi01/data_graphcast_2021_era5_30_days.nc')
    
    #example_batch_ = data_graphcast(default_settings=False,start='2021-01-01',end='2021-12-31',year=2021,example_batch=example_batch)
    #print('---> loading data')
    #example_batch_ = example_batch_.load()
    #example_batch_ = example_batch_.to_netcdf('/home/vsansi01/data_graphcast_2021_era5_S2S.nc')

