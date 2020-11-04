import numpy as np
import xarray as xr

def pacific_lon(longitude, to180=True):
    """
    Help converting Pacific longitudes to 180 or 360 coordinates.
    
    Args:
        longitude (float): Single longitude value.
        to180 (boolean): If coords are in 360 degrees. Defaults to ``True``.
    """
    if to180:
        if longitude <= 180:
            return longitude
        if longitude > 180:
            return longitude-360
    if not to180:
        if longitude <= 0:
            return longitude+360
        if longitude > 0:
            return longitude

def weighted_mean(da, lat_name):
    """
    Compute the weighted mean of data.
    Args:
        da (xarray data array): Some data. Make sure to reduce coordinates to ``time, lat, lon``.
        lat_name (str): Name of array containing latitudes for weights. Can be multi-dimensional.
    Returns:
        weighted (area averaged) mean of variable.
    """
    weights = np.cos(np.deg2rad(da.coords[lat_name]))
    weights.name = "weights"
    sst_weighted = da.weighted(weights)
    # make sure dims are in time, then lat/lons for next step!
    weighted_mean = sst_weighted.mean(dim=(da.dims[1:3]), skipna=True)
    # returns weighted mean with one data value per time
    return weighted_mean
