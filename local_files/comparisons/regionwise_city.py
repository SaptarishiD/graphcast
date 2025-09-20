from tqdm.auto import tqdm
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def compute_regional_rmse(forecast_file, target_file, regions_dict, model_name='model'):
    """
    Compute RMSE for precipitation over specified regions for all lead times.
    
    Parameters:
    -----------
    forecast_file : str
        Path to forecast NetCDF file
    target_file : str
        Path to target/truth NetCDF file
    regions_dict : dict
        Dictionary with region names as keys and (lat_min, lat_max, lon_min, lon_max) as values
    model_name : str
        Name of the model for recording
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: init_date, lead_time, model, region, rmse
    """
    
    # Extract initialization date from filename
    init_date_str = Path(forecast_file).stem.replace('fine_init_', '')
    init_date = pd.to_datetime(init_date_str)
    
    # Load datasets
    forecast = xr.open_dataset(forecast_file, decode_timedelta=True)
    target = xr.open_dataset(target_file, decode_timedelta=True)
    
    # Get precipitation variable
    precip_var = 'total_precipitation_6hr'
    
    # Ensure consistent dimension order - forecast has (time, batch, lat, lon)
    # target has (batch, time, lat, lon) - transpose to match forecast
    target_precip = target[precip_var].transpose('time', 'batch', 'lat', 'lon')
    forecast_precip = forecast[precip_var]
    
    results = []
    
    # Iterate over regions
    for region_name, bounds in regions_dict.items():
        lat_min, lat_max, lon_min, lon_max = bounds
        
        # Select region - handle longitude wrapping if necessary
        # Assuming longitude is in 0-360 format based on the data shown
        if lon_min < lon_max:
            region_forecast = forecast_precip.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
            region_target = target_precip.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
        else:
            # Handle wrapping around 0/360 boundary
            region_forecast_1 = forecast_precip.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, 360)
            )
            region_forecast_2 = forecast_precip.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(0, lon_max)
            )
            region_forecast = xr.concat([region_forecast_1, region_forecast_2], dim='lon')
            
            region_target_1 = target_precip.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, 360)
            )
            region_target_2 = target_precip.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(0, lon_max)
            )
            region_target = xr.concat([region_target_1, region_target_2], dim='lon')
        
        # Compute RMSE for each lead time
        for i, lead_time in enumerate(forecast.time.values):
            # Extract data for this lead time
            forecast_lead = region_forecast.isel(time=i, batch=0).values
            target_lead = region_target.isel(time=i, batch=0).values
            
            # Compute RMSE
            mse = np.mean((forecast_lead - target_lead) ** 2)
            rmse = np.sqrt(mse)
            
            # Convert lead time to hours
            lead_hours = pd.Timedelta(lead_time).total_seconds() / 3600
            
            results.append({
                'init_date': init_date,
                'lead_time': lead_hours,
                'model': model_name,
                'region': region_name,
                'rmse': rmse
            })
    
    forecast.close()
    target.close()
    
    return pd.DataFrame(results)


def process_all_forecasts(forecast_dir, target_dir, regions_dict, output_file, 
                         model_name='model', forecast_pattern='fine_init_2014*.nc',
                         target_pattern='target_init_2014*.nc'):
    """
    Process all forecast files in a directory and compute regional RMSE.
    
    Parameters:
    -----------
    forecast_dir : str
        Directory containing forecast files
    target_dir : str
        Directory containing target files
    regions_dict : dict
        Dictionary with region names as keys and (lat_min, lat_max, lon_min, lon_max) as values
    output_file : str
        Path to output CSV file
    model_name : str
        Name of the model
    forecast_pattern : str
        Glob pattern for forecast files
    target_pattern : str
        Glob pattern for target files
    """
    
    # Find all forecast files
    forecast_files = sorted(glob.glob(str(Path(forecast_dir) / forecast_pattern)))
    
    if not forecast_files:
        raise ValueError(f"No forecast files found in {forecast_dir} with pattern {forecast_pattern}")
    
    all_results = []
    
    for forecast_file in tqdm(forecast_files):
        # Extract date from forecast filename and construct target filename
        forecast_path = Path(forecast_file)
        init_date_str = forecast_path.stem.replace('fine_init_', '')
        target_file = Path(target_dir) / f"target_init_{init_date_str}.nc"
        
        if not target_file.exists():
            print(f"Warning: Target file not found for {init_date_str}, skipping...")
            continue
        
        print(f"Processing initialization date: {init_date_str}")
        
        try:
            df_results = compute_regional_rmse(
                forecast_file, 
                str(target_file), 
                regions_dict, 
                model_name
            )
            all_results.append(df_results)
        except Exception as e:
            print(f"Error processing {init_date_str}: {e}")
            continue
    
    if all_results:
        # Combine all results
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Sort by init_date, lead_time, region
        final_results = final_results.sort_values(['init_date', 'lead_time', 'region'])
        
        # Save to CSV
        final_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        return final_results
    else:
        print("No results to save")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Define regions with their bounding boxes (lat_min, lat_max, lon_min, lon_max)
    regions = {
        'North_America': (20, 70, 230, 300),  # Adjusted for 0-360 longitude
        'Europe': (35, 70, 350, 40),  # Wraps around 0/360
        'East_Asia': (20, 50, 100, 145),
        'South_America': (-55, 15, 280, 330),
        'Africa': (-35, 35, 340, 55),
        'Australia': (-45, -10, 110, 155),
        'Arctic': (70, 90, 0, 360),  # All longitudes
        'Tropics': (-20, 20, 0, 360),  # All longitudes
    }
    
    # Process all forecasts
    results_df = process_all_forecasts(
        forecast_dir='/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds',
        target_dir='/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds',
        regions_dict=regions,
        output_file='precipitation_rmse_results_regions.csv',
        model_name='my_weather_model'
    )
    
    # Display summary statistics
    if not results_df.empty:
        print("\nSummary Statistics:")
        print("=" * 50)
        
        # Average RMSE by region
        print("\nAverage RMSE by Region:")
        print(results_df.groupby('region')['rmse'].mean().sort_values())
        
        # Average RMSE by lead time
        print("\nAverage RMSE by Lead Time (hours):")
        print(results_df.groupby('lead_time')['rmse'].mean().sort_values())
        
        # Sample of results
        print("\nSample of results:")
        print(results_df.head(10))