import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

def get_files_for_timestep(base_dir, timestamp):
    """
    Get list of files that should be aggregated for a given 6-hour timestep.
    
    Args:
        base_dir (str): Directory containing IMERG files
        timestamp (datetime): Target timestamp for 6-hour period
    
    Returns:
        list: List of file paths to be aggregated
    """
    files = []
    base_dir = Path(base_dir)
    
    # For each 30-minute interval in the 6-hour period
    for i in range(12):  # 12 half-hour periods in 6 hours
        current_time = timestamp + timedelta(minutes=30 * i)
        
        # Construct file pattern based on IMERG naming convention
        file_pattern = f"3B-HHR.MS.MRG.3IMERG.{current_time.strftime('%Y%m%d-S%H%M%S')}-*.nc"
        matching_files = list(base_dir.glob(file_pattern))
        
        if matching_files:
            files.append(matching_files[0])
    
    return sorted(files)

def aggregate_6hourly(input_dir, start_date, end_date, output_dir):
    """
    Aggregate IMERG half-hourly data to 6-hourly timesteps.
    
    Args:
        input_dir (str): Directory containing input IMERG files
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_dir (str): Directory for output files
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define aggregation methods for different variables
    aggregation_methods = {
        'precipitation': 'sum',           # Sum for precipitation
        'randomError': 'mean',           # Mean for random error
        'precipitationQualityIndex': 'mean',  # Mean for quality index
        'probabilityLiquidPrecipitation': 'max'  # Max for probability
    }
    
    # Generate list of 6-hourly timestamps
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    timestamps = pd.date_range(start, end, freq='6H')
    
    for timestamp in timestamps:
        # Get files for this 6-hour period
        files = get_files_for_timestep(input_dir, timestamp)
        
        if not files:
            print(f"No files found for period starting at {timestamp}")
            continue
            
        print(f"Processing {timestamp}: Found {len(files)} files")
        
        # Load and concatenate all files for this period
        datasets = []
        for file in files:
            try:
                ds = xr.open_dataset(file)
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if not datasets:
            continue
            
        # Concatenate along time dimension
        combined = xr.concat(datasets, dim='time')
        
        # Apply appropriate aggregation method for each variable
        aggregated_vars = {}
        for var, method in aggregation_methods.items():
            if var in combined:
                if method == 'sum':
                    aggregated_vars[var] = combined[var].sum(dim='time')
                elif method == 'mean':
                    aggregated_vars[var] = combined[var].mean(dim='time')
                elif method == 'max':
                    aggregated_vars[var] = combined[var].max(dim='time')
        
        # Create new dataset with aggregated variables
        aggregated = xr.Dataset(aggregated_vars)
        
        # Save aggregated file
        output_file = output_dir / f"IMERG.6H.{timestamp.strftime('%Y%m%d.%H%M')}.nc"
        aggregated.to_netcdf(output_file)
        print(f"Saved aggregated file: {output_file}")
        
        # Clean up
        for ds in datasets:
            ds.close()

# Example usage
if __name__ == "__main__":
    input_dir = "imerg_files"
    output_dir = "imerg_files"
    start_date = "2022-01-01"
    end_date = "2022-01-31"
    
    aggregate_6hourly(input_dir, start_date, end_date, output_dir)
