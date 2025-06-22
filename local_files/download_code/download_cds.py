import os
import cdsapi
import datetime
import isodate
import math
import numpy as np
import pandas as pd
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude
import pytz
import xarray
from scipy import __name__ as scipy_name

client = cdsapi.Client() # Making a connection to CDS, to fetch data.


# The fields to be fetched from the single-level source.
singlelevelfields = [
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '2m_temperature',
                        'geopotential',
                        'land_sea_mask',
                        'mean_sea_level_pressure',
                        'toa_incident_solar_radiation',
                        'total_precipitation'
                    ]

# The fields to be fetched from the pressure-level source.
pressurelevelfields = [
                        'u_component_of_wind',
                        'v_component_of_wind',
                        'geopotential',
                        'specific_humidity',
                        'temperature',
                        'vertical_velocity'
                    ]

# The 13 pressure levels.
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Initializing other required constants.
pi = math.pi
gap = 6 # There is a gap of 6 hours between each graphcast prediction.
first_prediction = datetime.datetime(2022, 1, 1, 18, 0) # Timestamp of the first prediction.
watts_to_joules = 3600
predictions_steps = 12 # Predicting for 4 timestamps.
lat_range = range(-180, 181, 1) # Latitude range.
lon_range = range(0, 360, 1) # Longitude range.

smallModel = True

if smallModel == False:
    spatial_resolution = '0.25/0.25'
else:
    spatial_resolution = '1.0/1.0'


# A utility function used for ease of coding.
# Converting the variable to a datetime object.
def toDatetime(dt) -> datetime.datetime:
    if isinstance(dt, datetime.date) and isinstance(dt, datetime.datetime):
        return dt
    
    elif isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        return datetime.datetime.combine(dt, datetime.datetime.min.time())
    
    elif isinstance(dt, str):
        if 'T' in dt:
            return isodate.parse_datetime(dt)
        else:
            return datetime.datetime.combine(isodate.parse_date(dt), datetime.datetime.min.time())
        
_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219




def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
  """Computes year progress for times in seconds.

  Args:
    seconds_since_epoch: Times in seconds since the "epoch" (the point at which
      UNIX time starts).

  Returns:
    Year progress normalized to be in the [0, 1) interval for each time point.
  """

  # Start with the pure integer division, and then float at the very end.
  # We will try to keep as much precision as possible.
  years_since_epoch = (
      seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
  )
  # Note depending on how these ops are down, we may end up with a "weak_type"
  # which can cause issues in subtle ways, and hard to track here.
  # In any case, casting to float32 should get rid of the weak type.
  # [0, 1.) Interval.
  return np.mod(years_since_epoch, 1.0).astype(np.float32)


def get_day_progress(
    seconds_since_epoch: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
  """Computes day progress for times in seconds at each longitude.

  Args:
    seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
      point at which UNIX time starts).
    longitude: 1D array of longitudes at which day progress is computed.

  Returns:
    2D array of day progress values normalized to be in the [0, 1) inverval
      for each time point at each longitude.
  """

  # [0.0, 1.0) Interval.
  day_progress_greenwich = (
      np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY
  )

  # Offset the day progress to the longitude of each point on Earth.
  longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
  day_progress = np.mod(
      day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0
  )
  return day_progress.astype(np.float32)