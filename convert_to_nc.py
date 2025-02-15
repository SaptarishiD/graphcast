import h5py
import netCDF4 as nc
import numpy as np
import glob

def copy_attributes(in_object, out_object):
    """Copy attributes from input object to output object, excluding problematic attributes"""
    skip_attributes = {
        '_FillValue',
        'DIMENSION_LIST',
        'REFERENCE_LIST',
        'CLASS',
        'NAME'
    }
    
    for key, value in in_object.attrs.items():
        if key not in skip_attributes:
            try:
                out_object.setncattr(key, value)
            except Exception as e:
                print(f"Info: Skipping attribute {key}")

def create_dimensions(nc_file, h5_file):
    """Create dimensions based on IMERG data structure"""
    grid = h5_file['Grid']
    
    # Create time dimension
    time_data = grid['time'][:]
    nc_file.createDimension('time', len(time_data))
    
    # Create lat/lon dimensions
    nc_file.createDimension('lon', len(grid['lon'][:]))
    nc_file.createDimension('lat', len(grid['lat'][:]))
    
    # Create bounds dimension
    nc_file.createDimension('nv', 2)

def create_coordinate_variables(nc_file, h5_file):
    """Create coordinate variables"""
    grid = h5_file['Grid']
    
    # Time variable
    time_var = nc_file.createVariable('time', np.float64, ('time',))
    time_var[:] = grid['time'][:]
    copy_attributes(grid['time'], time_var)
    
    # Latitude
    lat_var = nc_file.createVariable('lat', np.float32, ('lat',))
    lat_var[:] = grid['lat'][:]
    copy_attributes(grid['lat'], lat_var)
    
    # Longitude
    lon_var = nc_file.createVariable('lon', np.float32, ('lon',))
    lon_var[:] = grid['lon'][:]
    copy_attributes(grid['lon'], lon_var)
    
    # Bounds variables
    lat_bnds = nc_file.createVariable('lat_bnds', np.float32, ('lat', 'nv'))
    lat_bnds[:] = grid['lat_bnds'][:]
    
    lon_bnds = nc_file.createVariable('lon_bnds', np.float32, ('lon', 'nv'))
    lon_bnds[:] = grid['lon_bnds'][:]

def create_data_variables(nc_file, h5_file):
    """Create data variables"""
    grid = h5_file['Grid']
    
    # Main variables
    var_names = ['precipitation', 'randomError', 'probabilityLiquidPrecipitation', 
                 'precipitationQualityIndex']
    
    for var_name in var_names:
        if var_name in grid:
            try:
                var_data = grid[var_name]
                var = nc_file.createVariable(var_name, var_data.dtype, ('time', 'lon', 'lat'))
                print(var_name, var_data[:])
                var[:] = var_data[:]
                copy_attributes(var_data, var)
            except Exception as e:
                print(f"Warning: Error processing {var_name}: {e}")

def convert_h5_to_nc(h5_path, nc_path):
    """Convert IMERG HDF5 file to NetCDF4"""
    with h5py.File(h5_path, 'r') as h5_file:
        with nc.Dataset(nc_path, 'w', format='NETCDF4') as nc_file:
            # Copy global attributes
            copy_attributes(h5_file, nc_file)
            
            # Create structure
            create_dimensions(nc_file, h5_file)
            create_coordinate_variables(nc_file, h5_file)
            create_data_variables(nc_file, h5_file)

if __name__ == '__main__':
    files = glob.glob('imerg_files/*.HDF5')
    print(files)
    for file in files:
        convert_h5_to_nc(file, file.replace('HDF5', 'nc'))
