import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import cftime
import subprocess
import time
import copy
from config import directory_data
import argparse
from datetime import timedelta

#############################################################################
#############################################################################

parser = argparse.ArgumentParser(description='Computing global spectral analysis in parallel.')

parser.add_argument("--block",    required=True, type=int, help="This is the global block index to compute.")
parser.add_argument("--fileindx", required=True, type=int, help="This is the filename index to compute.")

args = parser.parse_args()

#############################################################################
#############################################################################

def assign_latlon(block_value):
    """
    Set block lat/lon range for parallelization.
    
    Args:
        block_value (int): The chosen block as integer.
    
    Returns:
        str(block_value), lats1, lats2, lons1, lons2
    
    """
    if block_value == 1:
        lats1 = 0; lats2 = 60; lons1 = 0; lons2 = 60
    
    if block_value == 2:
        lats1 = 0; lats2 = 60; lons1 = 60; lons2 = 120
    
    if block_value == 3:
        lats1 = 0; lats2 = 60; lons1 = 120; lons2 = 180

    if block_value == 4:
        lats1 = 0; lats2 = 60; lons1 = 180; lons2 = 240
    
    if block_value == 5:
        lats1 = 0; lats2 = 60; lons1 = 240; lons2 = 300
    
    if block_value == 6:
        lats1 = 0; lats2 = 60; lons1 = 300; lons2 = 360
    
    if block_value == 7:
        lats1 = 60; lats2 = 120; lons1 = 0; lons2 = 60
    
    if block_value == 8:
        lats1 = 60; lats2 = 120; lons1 = 60; lons2 = 120
    
    if block_value == 9:
        lats1 = 60; lats2 = 120; lons1 = 120; lons2 = 180
    
    if block_value == 10:
        lats1 = 60; lats2 = 120; lons1 = 180; lons2 = 240
    
    if block_value == 11:
        lats1 = 60; lats2 = 120; lons1 = 240; lons2 = 300
    
    if block_value == 12:
        lats1 = 60; lats2 = 120; lons1 = 300; lons2 = 360
    
    if block_value == 13:
        lats1 = 120; lats2 = 180; lons1 = 0; lons2 = 60
    
    if block_value == 14:
        lats1 = 120; lats2 = 180; lons1 = 60; lons2 = 120
    
    if block_value == 15:
        lats1 = 120; lats2 = 180; lons1 = 120; lons2 = 180
    
    if block_value == 16:
        lats1 = 120; lats2 = 180; lons1 = 180; lons2 = 240
    
    if block_value == 17:
        lats1 = 120; lats2 = 180; lons1 = 240; lons2 = 300
    
    if block_value == 18:
        lats1 = 120; lats2 = 180; lons1 = 300; lons2 = 360
        
    return str(block_value), lats1, lats2, lons1, lons2

def oisst_latlon(block_value):
    """
    Set block lat/lon range for parallelization.
    
    Args:
        block_value (int): The chosen block as integer.
    
    Returns:
        str(block_value), lats1, lats2, lons1, lons2
    
    """
    if block_value == 1:
        lats1 = 0; lats2 = 240; lons1 = 0; lons2 = 240
    
    if block_value == 2:
        lats1 = 0; lats2 = 240; lons1 = 240; lons2 = 480
    
    if block_value == 3:
        lats1 = 0; lats2 = 240; lons1 = 480; lons2 = 720

    if block_value == 4:
        lats1 = 0; lats2 = 240; lons1 = 720; lons2 = 960
    
    if block_value == 5:
        lats1 = 0; lats2 = 240; lons1 = 960; lons2 = 1200
    
    if block_value == 6:
        lats1 = 0; lats2 = 240; lons1 = 1200; lons2 = 1440
    
    if block_value == 7:
        lats1 = 240; lats2 = 480; lons1 = 0; lons2 = 240
    
    if block_value == 8:
        lats1 = 240; lats2 = 480; lons1 = 240; lons2 = 480
    
    if block_value == 9:
        lats1 = 240; lats2 = 480; lons1 = 480; lons2 = 720
    
    if block_value == 10:
        lats1 = 240; lats2 = 480; lons1 = 720; lons2 = 960
    
    if block_value == 11:
        lats1 = 240; lats2 = 480; lons1 = 960; lons2 = 1200
    
    if block_value == 12:
        lats1 = 240; lats2 = 480; lons1 = 1200; lons2 = 1440
    
    if block_value == 13:
        lats1 = 480; lats2 = 720; lons1 = 0; lons2 = 240
    
    if block_value == 14:
        lats1 = 480; lats2 = 720; lons1 = 240; lons2 = 480
    
    if block_value == 15:
        lats1 = 480; lats2 = 720; lons1 = 480; lons2 = 720
    
    if block_value == 16:
        lats1 = 480; lats2 = 720; lons1 = 720; lons2 = 960
    
    if block_value == 17:
        lats1 = 480; lats2 = 720; lons1 = 960; lons2 = 1200
    
    if block_value == 18:
        lats1 = 480; lats2 = 720; lons1 = 1200; lons2 = 1440
        
    return str(block_value), lats1, lats2, lons1, lons2

def assign_yrrange(filename_indx):
    """
    Set year range based on file under consideration.
    
    Args:
        filename_indx (int): Index of file name string.
        
    Returns:
        filename as string, year1, year2
    
    """
    if filename_indx == 0: 
        filename = f'{directory_data}b1d.e11.B1850LENS.f09_g16.FWAtSalG02Sv.pop.h.SST.*.nc'
        year1 = 201; year2 = 500
    
    if filename_indx == 1:
        filename = f'{directory_data}b1d.e11.B1850LENS.f09_g16.FWAtSalG04Sv.pop.h.SST.*.nc'
        year1 = 201; year2 = 500
        
    if filename_indx == 2:
        filename = f'{directory_data}b1d.e11.B1850LENS.f09_g16.FWAtSalP02Sv.pop.h.SST.*.nc'
        year1 = 201; year2 = 500
        
    if filename_indx == 3:
        filename = f'{directory_data}b1d.e11.B1850LENS.f09_g16.FWAtSalP04Sv.pop.h.SST.*.nc'
        year1 = 201; year2 = 500
    
    if filename_indx == 4:
        filename = f'{directory_data}b1d.e11.B1850LENS.f09_g16.FWPaSalP04Sv.pop.h.SST.*.nc'
        year1 = 101; year2 = 250
        
    if filename_indx == 5:
        filename = f'{directory_data}b1d.e11.B1850C5CN.f09_g16.005.pop.h.SST.*.nc'
        year1 = 1001; year2 = 1300
        
    if filename_indx == 6:
        filename = f'/gpfs/fs1/collections/rda/data/ds277.7/avhrr_v2.1/*/oisst-avhrr-v02r01.*.nc'
        year1 = 1982; year2 = 2020
        
    return filename, year1, year2
    
#############################################################################
#############################################################################

if args.fileindx <= 5:
    block, lats1, lats2, lons1, lons2 = assign_latlon(args.block)
    
if args.fileindx == 6:
    block, lats1, lats2, lons1, lons2 = oisst_latlon(args.block)

filename, year1, year2 = assign_yrrange(args.fileindx)

#############################################################################
#############################################################################

# location and name of dummy sst file (sst time series at a grid cell)
tmpfile = f'~/python_scripts/climatico/ncl/dummy{block}.nc'

# open all files associated with filename
ds = xr.open_mfdataset(f'{filename}')

# time fix
if args.fileindx <= 5:
    ds = ds.assign_coords(time=ds.coords['time'] - timedelta(days=17))
    datmp = ds.isel(z_t=0).sel(time=slice(cftime.DatetimeNoLeap(year1, 1, 1, 0, 0),cftime.DatetimeNoLeap(year2, 12, 31, 0, 0)))

if args.fileindx == 6:
    datmp = ds.isel(zlev=0).sel(time=slice(f'{year1}-01-01',f'{year2}-12-31')).resample(time='MS').mean(skipna=True)

# slice the region belonging to block
datmp = datmp.isel(lat=slice(lats1,lats2), lon=slice(lons1,lons2))

# grab sst variable
if args.fileindx <= 5:
    datmp = datmp['SST'].fillna(0.0)
    
if args.fileindx == 6:
    datmp = datmp['sst'].fillna(0.0)

# eagerly load all values
thessts = datmp.values

# create enumeration arrays for loop that follows of shape of the cut block
array1 = np.arange(0,thessts.shape[1])
array2 = np.arange(0,thessts.shape[2])

# create a dummy mask
dummy_regionmask = np.sum(thessts, axis=0)

# helper index 
indx = 0
# begin loop -- go thru each grid cell, save time series, compute fft, stick fft in array
for num, (i, j) in enumerate(product(array1, array2)):
    
    if dummy_regionmask[i,j] == 0:
        continue
    
    if dummy_regionmask[i,j] != 0:
        da_tmp = thessts[:,i,j]
        xr.Dataset({"SST": (["time"], da_tmp)}).to_netcdf(tmpfile, engine="scipy", compute=True)
        subprocess.call([f'ml intel/18.0.5; ml ncl; ncl /glade/u/home/molina/python_scripts/climatico/ncl/specx_anal{block}.ncl'], shell=True)
        spcx = xr.open_dataset(f"~/python_scripts/climatico/ncl/spcx{block}.nc") 

        if indx == 0:
            spcx_array = np.zeros((dummy_regionmask.shape[0]*dummy_regionmask.shape[1], spcx['spcx'].values.shape[0]))
            frqx = xr.open_dataset(f"~/python_scripts/climatico/ncl/frq{block}.nc")
            frqx_array = copy.deepcopy(frqx['frq'].squeeze().values)
            del frqx
            
        spcx_array[num] = copy.deepcopy(spcx['spcx'].squeeze().values)
        del spcx
        indx += 1
        
# put data in a dataset (xarray)
new_ds = xr.Dataset(
    {"specx": (["nlat x nlon", "frq"], spcx_array),
     "frq": (['frq'], frqx_array),
     "lat": (['lat'], datmp['lat'].values),
     "lon": (['lon'], datmp['lon'].values),
    },)

# save the dataset
new_ds.to_netcdf(f"{directory_data}specx_glbl{block}_{filename.split('.')[4]}_{year1}_{year2}.nc")
