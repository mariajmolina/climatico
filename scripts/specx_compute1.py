import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import cftime
import subprocess
import time
import copy
from config import directory_data

# chosen block (options 1 thru 12)
block = str(1)                         ####################################  edit place
lats1 = 0
lats2 = 60
lons1 = 0
lons2 = 60

# location and name of dummy sst file (sst time series at a grid cell)
tmpfile = f'~/python_scripts/climatico/ncl/dummy{block}.nc'

#list of filenames to do this for
filename = 'b1d.e11.B1850LENS.f09_g16.FWAtSalG02Sv.pop.h.SST.*.nc'; year1 = 201; year2 = 500
#filename = 'b1d.e11.B1850LENS.f09_g16.FWAtSalG04Sv.pop.h.SST.*.nc'; year1 = 201; year2 = 500
#filename = 'b1d.e11.B1850LENS.f09_g16.FWAtSalP02Sv.pop.h.SST.*.nc'; year1 = 201; year2 = 500
#filename = 'b1d.e11.B1850LENS.f09_g16.FWAtSalP04Sv.pop.h.SST.*.nc'; year1 = 201; year2 = 500
#filename = 'b1d.e11.B1850LENS.f09_g16.FWPaSalP04Sv.pop.h.SST.*.nc'; year1 = 101; year2 = 250
#filename = 'b1d.e11.B1850C5CN.f09_g16.005.pop.h.SST.*.nc'; year1 = 1001; year2 = 1300

# open all files associated with filename
ds = xr.open_mfdataset(f'{directory_data}{filename}')
# remove z_t extra dim and slice the amoc collapse/active period
datmp = ds.isel(z_t=0).sel(time=slice(cftime.DatetimeNoLeap(year1, 1, 1, 0, 0),cftime.DatetimeNoLeap(year2, 12, 1, 0, 0)))
# slice the region belonging to block
datmp = datmp.isel(lat=slice(lats1,lats2), lon=slice(lons1,lons2))   
# grab sst variable
datmp = datmp['SST'].fillna(0.0)
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
