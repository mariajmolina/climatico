import numpy as np
import matplotlib.pyplot as plt
import copy
import xarray as xr
import scipy.spatial

def haversine(lat1, lon1, lat2, lon2):
    """
    Haversine distance formula.
    Longitude can be in 180 or 360 degrees.
    
    Args:
        lat1 (float): First latitude.
        lon1 (float): First longitude.
        lat2 (float): Second latitude.
        lon2 (float): Second longitude.
        
    Outputs distance measured in km.
    """
    R = 6372.8 #km
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def grl_mask(lat, lon, landseamask):
    """
    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        landseamask: Mask.
        
    Returns: 
        2d array (mask) of weighted values over region.
    """
    # grab ocean and land points
    jsea,isea = np.where(landseamask==1.0)
    jland,iland = np.where(landseamask==0.0)
    # copy mask
    dist = copy.deepcopy(landseamask)
    # points designated to ignore, make 301-km
    dist[np.where(dist == 2.0)]=301.
    
    # loop thru points
    for j,i in zip(jsea,isea):
        
        # compute distance to land points
        dist_to_land = haversine(lat[j,i],lon[j,i],lat[jland,iland],lon[jland,iland])
        
        # grab minimum distance between land distance or 301-km
        dist[j,i] = min(dist_to_land.min(),301.)
        
    # maximum distance
    lmax = 300.    
    # make a region for inputs
    region = copy.deepcopy(landseamask) * 0.0
    # get indices meeting conditions
    j,i = np.where((dist<lmax) & (dist>0.0))
    # create weights
    region[j,i] = np.exp(-dist[j,i] / lmax)
    return region   

def do_kdtree(lat, lon, point):
    """
    Find indices of nearest lat, lon coordinates using KD tree.
    
    Args:
        lat (array): latitude array.
        lon (array): longitude array.
        point (tuple): lat, lon tuple (float).
    
    Returns:
        indices of nearest lat lon coordinate.
    """
    # unravel lats and lons into coordinate pairs
    combined_x_y_arrays = np.dstack([lat.ravel(), lon.ravel()])[0]
    # use KD tree algorithm (Maneewongvatana and Mount 1999) to find nearest points
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    _, indexes = mytree.query(point)
    # return only the indices
    return indexes

def print_nearest_indx(lat, lon, point):
    """
    For use during search for indices needed to ignore regions in grid.
    
    Args:
        lat (array): latitude array.
        lon (array): longitude array.
        point (tuple): lat, lon tuple (float).
        
    ```
    E.g.:
    print_nearest_indx(lat, lon, point=(70.,275.))
    
    Returns:
    Index of nearest latitude: 363
    Index of nearest longitude: 263
    ```
    """
    # extract the indices from the KD tree flattened arrays
    indx = np.unravel_index(indices=do_kdtree(lat, lon, (point)), shape=np.shape(lat))
    print(f"Index of nearest latitude: {indx[0]}")
    print(f"Index of nearest longitude: {indx[1]}")
    
def make_hosing_nc(hosing_rate, fileout, return_file = False,
                   mesh_file = '/glade/scratch/ahu/archive/b.e22.B1850LENS.hosingCoast/ocn/hist/b.e22.B1850LENS.hosingCoast.pop.h.0054-10.nc',
                   author = 'Maria J. Molina for Aixue Hu',
                   description = 'Greenland coast hosing'):
    """
    Create a netcdf file to input fresh water hosing for CESM grid.
    
    ```
    For example:
    
    # 0.1 Sv, 0.3 Sv, and 0.5 Sv, and 0.0 Sv.
    create_hosing.make_hosing_nc(hosing_rate=0.1, fileout='/glade/scratch/molina/hosing_greenland_0.1_Sv.nc')
    create_hosing.make_hosing_nc(hosing_rate=0.3, fileout='/glade/scratch/molina/hosing_greenland_0.3_Sv.nc')
    create_hosing.make_hosing_nc(hosing_rate=0.5, fileout='/glade/scratch/molina/hosing_greenland_0.5_Sv.nc')
    create_hosing.make_hosing_nc(hosing_rate=0.0, fileout='/glade/scratch/molina/hosing_greenland_0.0_Sv.nc')
    ```
    
    Args:
       hosing_rate (float):   Hosing rate in Sv.
       fileout (str):         File name for output file.
       meshfile (str):        Directory and file for mesh grid.
    """
    # open grid
    ds = xr.open_dataset(mesh_file)
    
    # grab variables from ds
    tmaskutil = ds['REGION_MASK'].where(ds['REGION_MASK']==0,1).values # land mask
    lat = ds.coords['TLAT'].values
    lon = ds.coords['TLONG'].values
    dx = ds['DXT'].values
    dy = ds['DYT'].values
    
    # Make hosing array
    landseamask = copy.deepcopy(tmaskutil)  # deep copy of mask
    # add 2's to get mask of Greenland region (along parts of coast) using data indices for speed
    # indices grabbed using iteration and ``print_nearest_index``
    landseamask[:339,280:]=2.               # ignore south of greenland
    landseamask[339:351,280:300]=2.         # ignore parts of north america
    landseamask[:369,220:293]=2.            # ignore parts of north america
    landseamask[:373,190:284]=2.
    landseamask[300:376,193:247]=2.
    landseamask[:,110:275]=2.
    landseamask[:359,19:130]=2.
    landseamask[355:367,19:32]=2.           # iceland
    landseamask[:220,:100]=2.               # parts of africa and antarctica
    landseamask[330:371,84:105]=2.          # island south of greenland
    
    region = grl_mask(lat, lon, landseamask)  # get weighted mask of region to hose
    hflux = np.ma.masked_equal(tmaskutil, 0.0) * 0.0  # Created masked array
    hflux.data[:]=region[:]    # put region array into masked array
    area_of_hosing = (dx * dy * region).sum() # calc area of hosing region
    
    hosing_per_unit_area = hosing_rate * 1.0e6 * 1000.0 / (area_of_hosing * 1e-4)  # rate_in_Sv * 1.0e6 * rho_fw / (converted) area = kg/m2/s 
    
    hflux *= hosing_per_unit_area  # hosing flux in kg/m2/s
    
    # assemble netcdf file
    data_assemble = xr.Dataset({
                         'hosing': (['nlat','nlon'], hflux),
                        },
                         coords =
                        {'DXT'   : (['nlat','nlon'], dx),
                         'DYT'   : (['nlat','nlon'], dy),
                         'TLAT'  : (['nlat','nlon'], lat),
                         'TLONG' : (['nlat','nlon'], lon)
                        },
                        attrs = 
                        {'Author' : author,
                         'File' : description,
                         'Hosing rate' : str(hosing_rate),
                         'Original pop mesh' : mesh_file,
                         'Hosing flux units' : 'kg/m2/s'})
    
    # save netcdf file
    data_assemble.to_netcdf(fileout)
    
    if return_file:
        return data_assemble
