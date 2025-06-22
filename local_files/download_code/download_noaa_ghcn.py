from noaa_ghcn import GHCN
import shapely

ghcn = GHCN()

lat_min = 7.0
lat_max = 37.0
lon_min = 65.0
lon_max = 97.0


import datetime as dt
inventory_subset = ghcn.filter_inventory(start_date= dt.datetime(2010, 1, 1), end_date=dt.datetime(2025, 5, 31))


inventory_subset = inventory_subset[
    (inventory_subset['LATITUDE'] >= lat_min) & (inventory_subset['LATITUDE'] <= lat_max) &
    (inventory_subset['LONGITUDE'] >= lon_min) & (inventory_subset['LONGITUDE'] <= lon_max)
]


import time

tik = time.time()


print("Starting download")

df = ghcn.load_data(inventory_subset)


tok = time.time()

print(f"Download took: {tok - tik}")

print(f"We have: {df.head()}")


df.to_csv("/Datastorage/saptarishi.dhanuka_asp25/noaa_data_20100101_20250531.csv")

