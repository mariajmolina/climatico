import numpy as np
import xarray as xr
import pandas as pd

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
        if longitude < 0:
            return longitude+360
        if longitude >= 0:
            return longitude

def nino_dict(nino='nino34'):
    """
    Help to grab Nino region coords for SST bounds.
    Note: Longitudes in CESM ocean model output are in 360 degree coords.
    Args:
        nino (str): Nino SST region in Pacific (all lowercase). Defaults to ``nino34``.
    """
    n = {
         'nino12': np.array([-10, 0, pacific_lon(-90, to180=False), pacific_lon(-80, to180=False)]),
         'nino3': np.array([-5, 5, pacific_lon(-150, to180=False), pacific_lon(-90, to180=False)]),
         'nino34': np.array([-5, 5, pacific_lon(-170, to180=False), pacific_lon(-120, to180=False)]),
         'nino4': np.array([-5, 5, pacific_lon(160, to180=False), pacific_lon(-150, to180=False)]),
        }
    try:
        out = n[nino]
        return out
    except:
        raise ValueError('Not a Nino SST region! Make sure input is all lowercase.')

def nino(data, nino='nino34', lats='TLAT', lons='TLONG'):
    """
    Using Nino regional bounds in the Pacific for SST extraction from
    nino_dict().
    Note: Ocean CESM model (pop) uses irregular grid. Therefore, 
          lat and lon coordinates have two dimensions, and should
          be specified in function input.
    Args:
        data (xarray dataset): SST data.
        nino (str): Nino SST region in Pacific (all lowercase). Defaults to ``nino34``.
        lats (str): Name of irregular grid latitudes (should have two dims). Defaults to ``TLAT``.
        lons (str): Name of irregular grid longitudes (should have two dims). Defaults to ``TLONG``.
    """
    nino_coords = nino_dict(nino=nino)
    return data.where((data[lats]<nino_coords[1]) & (data[lats]>nino_coords[0]) & \
                      (data[lons]>nino_coords[2]) & (data[lons]<nino_coords[3]), drop=True)

def check_nino(data, lats='TLAT', lons='TLONG'):
    """
    Quick sanity check function for regional bounds.
    Args:
        data (xarray dataset): SST data.
        lats (str): Name of irregular grid latitudes (should have two dims). Defaults to ``TLAT``.
        lons (str): Name of irregular grid longitudes (should have two dims). Defaults to ``TLONG``.
    """
    print('Check we have the correct spatial extent')
    print('Latitude range: {:.1f} - {:.1f}'.format(data[lats].min().values, data[lats].max().values))
    print('Longitude range: {:.1f} - {:.1f}'.format(pacific_lon(data[lons].min().values), 
                                                    pacific_lon(data[lons].max().values)))
