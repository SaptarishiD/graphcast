import numpy as np
import xarray as xr


# SD
def compute_mse(forecast: xr.Dataset, target: xr.Dataset, var) -> xr.Dataset:
    """Compute Mean Squared Error over lat and longitude."""
    if var == 'all':
        diff = forecast - target
    else:
        diff = forecast[var] - target[var]
    mse = (diff ** 2).mean(dim=("lat", "lon"))
    return mse

def compute_rmse(forecast: xr.Dataset, target: xr.Dataset, var: str) -> xr.Dataset:
    """Compute Root Mean Squared Error over lat and longitude."""
    return np.sqrt(compute_mse(forecast, target, var))

def compute_mae(forecast: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Compute Mean Absolute Error over lat and longitude."""
    diff = forecast - target
    mae = abs(diff).mean(dim=("lat", "longitude"))
    return mae

def compute_bias(forecast: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Compute Bias over lat and longitude."""
    diff = forecast - target
    bias = diff.mean(dim=("lat", "longitude"))
    return bias

def compute_acc(forecast: xr.Dataset, target: xr.Dataset, climatology: xr.Dataset) -> xr.Dataset:
    """Compute Anomaly Correlation Coefficient over lat and longitude."""
    forecast_anom = forecast - climatology
    target_anom = target - climatology
    numerator = (forecast_anom * target_anom).mean(dim=("lat", "longitude"))
    denominator = np.sqrt(
        (forecast_anom ** 2).mean(dim=("lat", "longitude")) * (target_anom ** 2).mean(dim=("lat", "longitude"))
    )
    acc = numerator / denominator
    return acc