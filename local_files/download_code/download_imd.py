import imdlib as imd

start_yr = 1951
end_yr = 2014
variable = 'rain' # other options are ('tmin'/ 'tmax')
# file_dir = (r'C:\Users\imdlib\Desktop\\') #Path to save the files
file_dir = '/Datastorage/saptarishi.dhanuka_asp25/imdlib_data/raw_imdlib'
imd.get_data(variable, start_yr, end_yr, fn_format='yearwise', file_dir=file_dir)