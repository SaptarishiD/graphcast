import jax
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import argparse
import setup_jax_functions


import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from shapely.geometry import mapping



def mask_dbase_india_buffer(dbase, buffer=True, buffer_deg=2.0, crs="EPSG:4326", lat_thresh=23.5):


    


    if "latitude" in dbase.coords:
        dbase = dbase.rename({"latitude": "lat", "longitude": "lon"})
    elif "lat" in dbase.coords:
        dbase = dbase.rename({"lat": "latitude", "lon": "longitude"})

    # Load world boundaries from Natural Earth (built-in to geopandas)
    world = gpd.read_file("/Datastorage/saptarishi.dhanuka_asp25/ne_110m_admin_0_countries.shp")
    india = world[world.SOVEREIGNT == 'India']
    india_gdf = india.geometry

    """
    return a GeoSeries with the original land unioned with a 1° coastal buffer
    everywhere south of lat_thresh.
    """
    # 1. Ensure it's in lat/lon
    india_gdf = india_gdf.to_crs(crs)
    
    # 2. Merge all parts into one Polygon/MultiPolygon
    land = unary_union(india_gdf.geometry)
    
    # 3. Build a global mask for lat < lat_thresh
    lat_mask = box(-180, -90, 180, lat_thresh)
    
    # 4. Buffer the land by buffer_deg (in degrees)
    land_buffered = land.buffer(buffer_deg)
    
    # 5. Restrict the buffer to lat < lat_thresh, then subtract land to get just the new strip
    coastal_extension = (land_buffered
                         .intersection(lat_mask)
                         .difference(land))
    
    # 6. Union land + extension
    extended = land.union(coastal_extension)

    extended_series = gpd.GeoSeries([extended], crs=crs)

    mask = geometry_mask(
    [mapping(extended_series.iloc[0])],     # List of geometries
    out_shape=(len(dbase.latitude), len(dbase.longitude)),
    transform=dbase.rio.transform(),
    invert=True,                           # Areas *inside* geometry == True
    all_touched=False
)

    # Convert mask to xarray DataArray (True inside India, False outside)
    mask_xr = xr.DataArray(mask, coords={"latitude": dbase.latitude, "longitude": dbase.longitude}, dims=("latitude", "longitude"))

    # Apply mask using xarray.where: keep data inside India, zero out rest
    dbase_masked = xr.where(mask_xr, dbase, 0)  # or use `np.nan` instead of 0 if you prefer
    dbase_masked = dbase_masked.assign_coords(longitude=((dbase_masked.longitude + 360) % 360))
    dbase_masked = dbase_masked.sortby("longitude")


    return dbase_masked



def construct_era5_imerg(dbase, printyes=False):
        era_ds = dbase.sel(time=slice('2023-01-01', None))
        del dbase
        imerg_ds_initial = xr.open_dataset('/Datastorage/saptarishi.dhanuka_asp25/imerg_data/combined_imerg_dir/combined_imerg_20240101_20241231.nc4')
        imerg_ds_initial = imerg_ds_initial.where(imerg_ds_initial.lon != 180.0, drop=True)
        imerg_ds = imerg_ds_initial

        imerg_ds = imerg_ds_initial.assign_coords(lon=(imerg_ds_initial.lon + 360) % 360)
        imerg_ds = imerg_ds.sortby("lon")
        imerg_ds = imerg_ds.sortby("lat")

        era5_times = era_ds.time
        imerg_6hr = xr.Dataset(
            data_vars={
                "precipitation_6hr": (("time", "latitude", "longitude"), 
                                    np.zeros((len(era5_times), len(era_ds.latitude), len(era_ds.longitude))))
            },
            coords={
                "time": era5_times,
                "longitude": era_ds.longitude,
                "latitude": era_ds.latitude
            }
        )

        imerg_times = imerg_ds.time


        def find_nearest_imerg_day(ds, target_day, max_search=7):
            """
            Look up to ±max_search days for the closest available IMERG time slice.
            Returns (imerg_day_data, actual_day) or (None, None) if not found.
            """
            # check zero offset first
            for offset in range(0, max_search + 1):
                for sign in (+1, -1) if offset > 0 else (+1,):
                    candidate = target_day + pd.Timedelta(days=sign * offset)
                    try:
                        data = ds.sel(time=candidate)
                        return data, candidate
                    except KeyError:
                        continue
            # if we exit the loops, nothing was found
            return None, None


        # Loop through each unique ERA5 day
        for day in tqdm(pd.DatetimeIndex(era5_times.values).normalize().unique(), desc="Looping through era5 days for conversion"):
            if day > imerg_times.values[-1]:
                print(f"Exceeded IMERG timespan, stopped replacing with IMERG")
                break
            next_day = day + pd.Timedelta(days=1)

            # Try to get the best available IMERG slice:
            imerg_day_data, found_day = find_nearest_imerg_day(imerg_ds, day, max_search=7)

            if imerg_day_data is None or 'precipitation' not in imerg_day_data:
                if printyes:
                    print(f"Warning: No IMERG data within ±7 days of {day}")
                # ────────────────────────────────────────────────────────────────────────
                # The `continue` here jumps straight to the next iteration of the
                # outer `for day in …` loop, skipping all processing for this `day`.
                # Without it, you'd fall through into the precipitation‐conversion code
                # even though you don't have valid input.
                continue

            # (optional) let user know if it fell back to a nearby day
            if found_day != day:
                print(f"Info: used IMERG data from {found_day} for target {day!r}")

            # Extract and convert the precipitation
            daily_precip = imerg_day_data.precipitation
            sixhour_precip = daily_precip * 0.25 * 0.001  # m per 6 hours

            # Mask the ERA5 times for this day
            day_mask = (era5_times >= day) & (era5_times < next_day)
            era5_day_times = era5_times.where(day_mask, drop=True)

            # Assign into your 6‑hr IMERG array with NaN check
            for t in era5_day_times.values:
                # Get the corresponding ERA5 precipitation for this time
                era5_precip_at_time = era_ds["total_precipitation_6hr"].sel(time=t)
                
                # Create a mask for non-NaN IMERG values
                valid_imerg_mask = ~np.isnan(sixhour_precip.values)
                
                # Start with ERA5 data as the base
                combined_precip = era5_precip_at_time.values.copy()
                
                # Replace with IMERG data only where IMERG is not NaN
                # Note: Need to transpose sixhour_precip to match ERA5 dimension order
                imerg_transposed = sixhour_precip.values.T
                combined_precip[valid_imerg_mask.T] = imerg_transposed[valid_imerg_mask.T]
                
                # Assign the combined data
                imerg_6hr["precipitation_6hr"].loc[dict(time=t)] = combined_precip

        # Merge back and save as before...
        updated_era5 = era_ds.drop_vars("total_precipitation_6hr")
        updated_era5["total_precipitation_6hr"] = imerg_6hr.precipitation_6hr
        updated_era5.total_precipitation_6hr.attrs.update({
            "source": "IMERG daily precipitation converted to 6‑hourly accumulation, with ERA5 fallback for NaN values",
            "original_source": "GPM IMERG Final Precipitation L3 1 day 0.1°×0.1° V07",
            "conversion_method": "Daily rate (mm/day) → 6‑hr accum. (m), ERA5 used where IMERG is NaN",
            "units": "m",
        })

        updated_era5 = updated_era5.where(updated_era5.time < imerg_times[-1], drop=True)


        # Averaging for missing longitude

        
        var = updated_era5['total_precipitation_6hr']  # shape: (time, lat, lon)

        # The longitude to be replaced
        target_lon = 180

        # Find the index of that longitude
        target_lon_idx = np.argmin(np.abs(updated_era5.longitude - target_lon).values)
        target_lon_val = updated_era5.longitude[target_lon_idx].values

        # Get adjacent longitudes (left and right)
        left_lon_idx = target_lon_idx - 1
        right_lon_idx = target_lon_idx + 1

        # For all latitudes, we want to average the 4 points:
        # (lat+1, lon-1), (lat+1, lon+1), (lat-1, lon-1), (lat-1, lon+1)

        # We'll loop over latitudes and calculate this for each
        lats = updated_era5.latitude
        lat_len = len(lats)

        new_data = var.copy(deep=True)

        for lat_idx in range(1, lat_len - 1):  # skip edge lats
            # Pick 4 neighboring points for this latitude band
            points = [
                var.isel(latitude=lat_idx - 1, longitude=left_lon_idx),
                var.isel(latitude=lat_idx - 1, longitude=right_lon_idx),
                var.isel(latitude=lat_idx + 1, longitude=left_lon_idx),
                var.isel(latitude=lat_idx + 1, longitude=right_lon_idx),
            ]
            
            # Stack and take the mean across the 4 points
            mean_val = xr.concat(points, dim='dummy').mean(dim='dummy')

            # Replace the target longitude value at this latitude
            new_data.loc[dict(latitude=lats[lat_idx], longitude=target_lon_val)] = mean_val

            updated_era5['total_precipitation_6hr'] = new_data

        imerg_renamed = imerg_ds.rename({'lat': 'latitude', 'lon': 'longitude'})
        del imerg_ds_initial
        del imerg_ds
        del era_ds

        def verify_era5_matches_imerg_6hrly(updated_era5, imerg_ds, bbox=None, atol=1e-6):
                """
                For each 6-hourly ERA5 timestamp, check if the value equals the IMERG daily
                value converted to m/6hr over a bounding box.
                Assumes each 6-hourly time in a day should have the same value.
                """
                if bbox is None:
                    bbox = {'lat_min': 5.0, 'lat_max': 35.0, 'lon_min': 65.0, 'lon_max': 100.0}

                # Slice both datasets to bounding box
                era5 = updated_era5.sel(
                    latitude=slice(bbox['lat_min'], bbox['lat_max']),
                    longitude=slice(bbox['lon_min'], bbox['lon_max'])
                )
                imerg = imerg_ds.sel(
                    latitude=slice(bbox['lat_min'], bbox['lat_max']),
                    longitude=slice(bbox['lon_min'], bbox['lon_max'])
                )

                report = []

                # Loop through all 6-hour timestamps
                for t in pd.DatetimeIndex(era5.time.values):
                    day = t.normalize()

                    try:
                        imerg_day = imerg.sel(time=day).precipitation
                    except KeyError:
                        if printyes:
                            print(f"Skipping {t}: no IMERG data for {day}")
                        continue

                    # Convert IMERG to m/6hr
                    expected_val = imerg_day * 0.25 * 0.001  # mm/day → m/6hr

                    # Get ERA5 value at that timestamp
                    era5_val = era5["total_precipitation_6hr"].sel(time=t)

                    # Compare the two (only where IMERG is not NaN)
                    valid_mask = ~np.isnan(expected_val)
                    if valid_mask.sum() > 0:  # Only compare if there are valid IMERG values
                        diff = np.abs(era5_val - expected_val).where(valid_mask)
                        max_diff = float(diff.max().values)

                        if max_diff > atol:
                            report.append({
                                "timestamp": str(t),
                                "max_diff": max_diff,
                                "mean_diff": float(diff.mean().values),
                            })

                if not report:
                    print("✅ All ERA5 6-hourly values match the converted IMERG values (where IMERG is valid).")
                else:
                    print("❌ Mismatches found at these timestamps:")
                    for r in report:
                        print(f"{r['timestamp']}: max diff = {r['max_diff']:.6e}, mean diff = {r['mean_diff']:.6e}")
                        
                        
                return report


        report = verify_era5_matches_imerg_6hrly(updated_era5, imerg_renamed)

        updated_era5_cp = updated_era5
        updated_era5_cp['geopotential_at_surface'] = updated_era5['geopotential_at_surface'].isel(time=0)
        updated_era5_cp['land_sea_mask'] = updated_era5['land_sea_mask'].isel(time=0)

        updated_era5_cp['total_precipitation_6hr'] = updated_era5_cp['total_precipitation_6hr'].astype('float32')
        

        if abs(updated_era5_cp.latitude.values[0] - updated_era5_cp.latitude.values[1]) == 1:
            updated_era5_cp['total_precipitation_6hr'] = (
            updated_era5_cp['total_precipitation_6hr']
            .astype('float32')
            .chunk({'time': 1, 'latitude': 181, 'longitude': 360})
            )

        return updated_era5_cp




def compute_mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def compute_mse_diffs(diff):
    return (diff ** 2).mean(dim=["lat", "lon"]).values


def process_to_graphcast_format(eval_time_ds):
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

    eval_time_ds = eval_time_ds.expand_dims(batch=1)
    eval_time_ds = eval_time_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    datetime_array = eval_time_ds['time'].values
    # Calculate the time coordinate in 6-hour increments (in nanoseconds)
    time_array = np.arange(0, len(datetime_array) * 21600000000000, 21600000000000, dtype='timedelta64[ns]')

    # Add the new 'time' coordinate to the dataset
    final_eval_ds = eval_time_ds.assign_coords(datetime=('time', time_array))

    temp_time = final_eval_ds.coords["time"].copy()
    temp_datetime = final_eval_ds.coords["datetime"].copy()

    # Reassign the coordinates, swapping their values
    final_eval_ds = final_eval_ds.assign_coords(
        time=temp_datetime,
        datetime=temp_time
    )

    # final_eval_ds['geopotential_at_surface'] = final_eval_ds['geopotential_at_surface'].isel(batch=0, time=0)
    # final_eval_ds['land_sea_mask'] = final_eval_ds['land_sea_mask'].isel(batch=0, time=0)

    old_datetime = final_eval_ds["datetime"].values  # shape (1489,)

    # For our purposes, we want the coordinate to have shape (batch, time). Since the batch
    # dimension is of length 1, we can simply add a new axis.
    new_datetime = old_datetime[np.newaxis, :]  # shape becomes (1, 1489)

    # Now, reassign the "datetime" coordinate to have dims ("batch", "time").
    final_eval_ds = final_eval_ds.assign_coords(datetime=(("batch", "time"), new_datetime))

    # print(f"Coordinates after reassigning: {final_eval_ds.coords}\n")
    return final_eval_ds



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GraphCast forecasts against datasets like ERA5/IMERG")

    # Evaluation range
    parser.add_argument('--eval_start', type=str, required=True,
                        help='Start date for evaluation period (format: YYYY-MM-DD)')
    parser.add_argument('--eval_end', type=str, required=True,
                        help='End date for evaluation period (format: YYYY-MM-DD)')

    # Dataset choice
    parser.add_argument('--eval_dataset_choice', type=str, choices=['era5', 'imerg', 'imd', 'stations'], default='era5',
                        help='Dataset choice for evaluation')
    
    parser.add_argument('--vars_to_eval', type=str, required=True, help='Which variables to evaluate')

    # Data paths
    parser.add_argument('--eval_data_path', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/',
                        help='Path to Eval dataset')
    
    parser.add_argument('--params_path_old', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/origs/graphcast_1_13.npz',
                        help='Path to original GraphCast parameters (.npz)')
    
    parser.add_argument('--params_path_new1', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig.npz',
                        help='Path to fine-tuned GraphCast parameters (.npz)')


    parser.add_argument('--params_path_new2', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig.npz',
                        help='Path to fine-tuned GraphCast parameters (.npz)')
    

    parser.add_argument('--norms_dir', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_norms/',
                        help='Directory containing normalization .nc files')

    # Output paths
    parser.add_argument('--output_pred_old_dir', type=str,
                        default='/Datastorage/saptarishi.dhanuka_asp25/preds_dir/',
                        help='Output path for original GraphCast predictions')
    parser.add_argument('--output_pred_finetuned_dir', type=str,
                        default='/Datastorage/saptarishi.dhanuka_asp25/preds_dir/',
                        help='Output path for fine-tuned GraphCast predictions')
    parser.add_argument('--plots_dir', type=str, default='plots/evals',
                        help='Directory to save eval plots')

    parser.add_argument('--plot_timesteps', type=int, default=4,
                        help='Number of timesteps to generate plots for')

    # Spatial extent
    parser.add_argument('--latmin', type=float, default=6, help='Minimum latitude for plotting')
    parser.add_argument('--latmax', type=float, default=38, help='Maximum latitude for plotting')
    parser.add_argument('--lonmin', type=float, default=68, help='Minimum longitude for plotting')
    parser.add_argument('--lonmax', type=float, default=98, help='Maximum longitude for plotting')

    return parser.parse_args()




def regrid_hres_fine_to_coarse(hres_grid, variable, coarse_resolution=1.0, ):
    target_lats = np.arange(-90, 91, coarse_resolution)
    target_lons = np.arange(0, 360, coarse_resolution)
    target_grid = xr.Dataset({
        'lat': (['lat'], target_lats),
        'lon': (['lon'], target_lons),
    })
    
    print(f"Regridding {variable} from high resolution to coarse resolution {coarse_resolution} degrees")

    regridder = xe.Regridder(
        hres_grid, 
        target_grid, 
        'nearest_s2d',
        locstream_in=True # important for unstructured data as we got with hres
    )

    print(f"Regridder built")

    regridded_ds = regridder(hres_grid[variable], keep_attrs=True)

    return regridded_ds

    









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



def regrid(source, target):
    pass