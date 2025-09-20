import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_india_bounds():
    """Define bounding box for India (approximate)"""
    return {
        'lat_min': 8.0,
        'lat_max': 37.0,
        'lon_min': 68.0,
        'lon_max': 97.0
    }

def subset_to_india(ds, bounds):
    """Subset dataset to India bounding box"""
    if 'latitude' in ds.coords:
        # HRES format
        subset = ds.sel(
            latitude=slice(bounds['lat_max'], bounds['lat_min']),  # Note: reversed for descending coords
            longitude=slice(bounds['lon_min'], bounds['lon_max'])
        )
    elif 'lat' in ds.coords:
        # Target format
        subset = ds.sel(
            lat=slice(bounds['lat_min'], bounds['lat_max']),
            lon=slice(bounds['lon_min'], bounds['lon_max'])
        )
    return subset

def interpolate_to_target_grid(forecast_data, target_data):
    """Interpolate forecast data to match target grid"""
    # Get target coordinates
    if 'lat' in target_data.coords and 'lon' in target_data.coords:
        target_lats = target_data.lat
        target_lons = target_data.lon
    else:
        raise ValueError("Target data must have 'lat' and 'lon' coordinates")
    
    # Interpolate forecast to target grid
    # Handle the case where forecast might have different coordinate names
    if 'latitude' in forecast_data.coords and 'longitude' in forecast_data.coords:
        forecast_interp = forecast_data.interp(
            latitude=target_lats,
            longitude=target_lons,
            method='linear'
        )
    else:
        raise ValueError("Forecast data must have 'latitude' and 'longitude' coordinates")
    
    return forecast_interp

def parse_target_filename(filename):
    """Extract initialization date from target filename"""
    # Expected format: target_init_2012-08-01 00:00:00.nc
    try:
        # Extract the date part from filename
        parts = filename.stem.split('_')
        date_part = parts[2]  # Should be like '2012-08-01'
        time_part = parts[3] + '_' + parts[4] + '_' + parts[5]  # '00:00:00'
        
        # Combine and parse
        datetime_str = f"{date_part} {time_part.replace('_', ':')}"
        return pd.to_datetime(datetime_str)
    except:
        # Alternative parsing if format is different
        filename_str = str(filename.stem)
        # Look for date pattern YYYY-MM-DD
        import re
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename_str)
        if date_match:
            return pd.to_datetime(date_match.group(1))
        else:
            raise ValueError(f"Could not parse date from filename: {filename}")

def calculate_mse(forecast, target):
    """Calculate Mean Squared Error between forecast and target"""
    # Handle NaN values
    valid_mask = ~(np.isnan(forecast) | np.isnan(target))
    if valid_mask.sum() == 0:
        return np.nan
    
    diff = forecast - target
    mse = np.mean(diff**2)
    return mse

def process_hres_forecasts(hres_file, target_dir, output_file):
    """Main function to process HRES forecasts and calculate MSE"""
    
    print(f"Loading HRES data from {hres_file}...")
    hres_ds = xr.open_dataset(hres_file)
    
    # Get India bounds
    india_bounds = get_india_bounds()
    
    # Subset HRES data to India
    print("Subsetting HRES data to India...")
    hres_india = subset_to_india(hres_ds, india_bounds)
    
    # Get target files
    target_files = list(Path(target_dir).glob("target_init_*.nc"))
    print(f"Found {len(target_files)} target files")
    
    # Filter for dates from August 1st onward
    august_cutoff = pd.to_datetime('2014-08-01')  # Adjust year as needed
    
    results = []
    
    for target_file in target_files:
        try:
            # Parse initialization date from filename
            init_date = parse_target_filename(target_file)
            
            # Skip if before August 1st
            if init_date < august_cutoff:
                continue
                
            print(f"Processing initialization date: {init_date}")
            
            # Check if this initialization date exists in HRES data
            if init_date not in hres_india.time.values:
                print(f"  Skipping - initialization date not found in HRES data")
                continue
            
            # Load target data
            target_ds = xr.open_dataset(target_file)
            
            # Debug: Print target data structure
            print(f"  Target data shape: {target_ds.total_precipitation_6hr.shape}")
            print(f"  Target time dimension: {len(target_ds.time)}")
            
            # Subset target to India
            target_india = subset_to_india(target_ds, india_bounds)
            
            # Get HRES data for this initialization date
            hres_init = hres_india.sel(time=init_date)
            
            # Debug: Print HRES data structure
            print(f"  HRES data shape: {hres_init.tp.shape}")
            print(f"  HRES steps: {len(hres_init.step)}")
            
            # Process each lead time/step
            for step_idx, step in enumerate(hres_init.step.values):
                step_hours = step / np.timedelta64(1, 'h')
                
                # Get forecast data for this step
                forecast_data = hres_init.tp.isel(step=step_idx)
                
                # Debug: Check forecast data dimensions
                print(f"    Forecast data shape for step {step_idx}: {forecast_data.shape}")
                
                # Calculate corresponding time index in target data
                # Assuming target time represents hours from initialization
                target_time_hours = target_india.time / np.timedelta64(1, 'h')
                
                # Find closest time in target data
                closest_time_idx = np.argmin(np.abs(target_time_hours - step_hours))
                
                print(f"    Matching step {step_hours:.0f}h with target time index {closest_time_idx}")
                
                if closest_time_idx < len(target_india.time):
                    try:
                        # Get target precipitation data
                        # Note: target has 'total_precipitation_6hr' variable
                        # Use .values to get numpy array directly
                        target_slice = target_india.total_precipitation_6hr.isel(
                            batch=0, 
                            time=closest_time_idx
                        )
                        
                        print(f"    Target slice shape: {target_slice.shape}")
                        
                        # Create a DataArray for interpolation with proper coordinates
                        target_coords = {
                            'lat': target_india.lat,
                            'lon': target_india.lon
                        }
                        target_data = xr.DataArray(
                            target_slice.values,
                            coords=target_coords,
                            dims=['lat', 'lon']
                        )
                        
                        # Interpolate forecast to target grid
                        forecast_interp = interpolate_to_target_grid(forecast_data, target_data)
                        
                        # Calculate MSE
                        mse = calculate_mse(
                            forecast_interp.values.flatten(), 
                            target_data.values.flatten()
                        )
                        
                        # Store results
                        results.append({
                            'initialization_date': init_date,
                            'lead_time_hours': step_hours,
                            'lead_time_days': step_hours / 24,
                            'mse': mse,
                            'target_file': target_file.name
                        })
                        
                        print(f"    Step {step_idx + 1}/{len(hres_init.step)}: {step_hours:.0f}h, MSE: {mse:.6f}")
                        
                    except Exception as step_error:
                        print(f"    Error processing step {step_idx}: {step_error}")
                        continue
                
        except Exception as e:
            print(f"Error processing {target_file}: {e}")
            continue
    
    # Convert results to DataFrame and save
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(['initialization_date', 'lead_time_hours'])
        
        # Save to CSV
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(df_results)} forecast-target pairs")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Date range: {df_results['initialization_date'].min()} to {df_results['initialization_date'].max()}")
        print(f"Lead times: {df_results['lead_time_hours'].min():.0f}h to {df_results['lead_time_hours'].max():.0f}h")
        print(f"Mean MSE: {df_results['mse'].mean():.6f}")
        print(f"Std MSE: {df_results['mse'].std():.6f}")
        
        return df_results
    else:
        print("No valid results found!")
        return None










# Example usage
if __name__ == "__main__":
    # Set your file paths here
    hres_file = '/Datastorage/divij.khaitan_asp25/forecasts_2014/hres_forecasts_20140601_20140831.nc'  # Path to your HRES file
    target_dir = "/Datastorage/saptarishi.dhanuka_asp25/rolled_out_preds"          # Directory containing target files
    output_file = "hres_mse_results.csv"        # Output CSV file
    
    # Run the analysis
    results = process_hres_forecasts(hres_file, target_dir, output_file)
    
    if results is not None:
        # Optional: Create some summary plots
        try:
            import matplotlib.pyplot as plt
            
            # MSE vs lead time
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            results.groupby('lead_time_hours')['mse'].mean().plot()
            plt.xlabel('Lead Time (hours)')
            plt.ylabel('Mean MSE')
            plt.title('MSE vs Lead Time')
            plt.grid(True)
            
            # MSE vs initialization date
            plt.subplot(1, 2, 2)
            daily_mse = results.groupby('initialization_date')['mse'].mean()
            daily_mse.plot()
            plt.xlabel('Initialization Date')
            plt.ylabel('Mean MSE')
            plt.title('MSE vs Initialization Date')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('mse_analysis_hres20140801_20140831.png', dpi=150, bbox_inches='tight')
            # plt.show()
            
            print("Summary plots saved as 'mse_analysis.png'")
            
        except ImportError:
            print("Matplotlib not available for plotting")