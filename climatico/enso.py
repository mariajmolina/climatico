import numpy as np
import xarray as xr
import pandas as pd
from util import pacific_lon, weighted_mean
import matplotlib.pyplot as plt

class DefineNino:
    """
    Class instantiation of DefineNino.
    Args:
        nino (str): Nino SST region in Pacific (all lowercase). E.g., ``nino34``.
                    Options include nino12, nino3, nino34, nino4, atl3.
    """
    def __init__(self, nino):
        """
        Initialize.
        """
        self.ninodefined = nino

    def nino_dict(self):
        """
        Help to grab Nino region coords for SST bounds.
        Note: Longitudes in CESM ocean model output are in 360 degree coords.
        """
        n = {
             'nino12': np.array([-10, 0, pacific_lon(-90, to180=False), pacific_lon(-80, to180=False)]),
             'nino3': np.array([-5, 5, pacific_lon(-150, to180=False), pacific_lon(-90, to180=False)]),
             'nino34': np.array([-5, 5, pacific_lon(-170, to180=False), pacific_lon(-120, to180=False)]),
             'nino4': np.array([-5, 5, pacific_lon(160, to180=False), pacific_lon(-150, to180=False)]),
             'atl3': np.array([-3, 3, pacific_lon(-20, to180=False), pacific_lon(0, to180=False)]),
            }
        try:
            out = n[self.ninodefined]
            return out
        except:
            raise ValueError('Not a Nino SST region! Make sure input is all lowercase.')

    def nino(self, data, lats='TLAT', lons='TLONG'):
        """
        Using Nino regional bounds in the Pacific for SST extraction from
        nino_dict().
        Note: Ocean CESM model (pop) uses irregular grid. Therefore, 
              lat and lon coordinates have two dimensions, and should
              be specified in function input.
        Args:
            data (xarray dataset): SST data.
            lats (str): Name of irregular grid latitudes (should have two dims). Defaults to ``TLAT``.
            lons (str): Name of irregular grid longitudes (should have two dims). Defaults to ``TLONG``.
        """
        nino_coords = self.nino_dict()
        return data.where((data[lats]<nino_coords[1]) & (data[lats]>nino_coords[0]) & \
                          (data[lons]>nino_coords[2]) & (data[lons]<nino_coords[3]), drop=True)

    def roll_climo(self, data, month, yrsroll=30, centered=True):
        """
        Creating rolling 30-year mean climatology.
        Args:
            data (xarray dataarray): Data Array weighted mean already computed.
            month (int): Month for climatology.
            yrsroll (int): Number of years for climatology. Defaults to ``30``.
            centered (boolean): Whether the average is centered. Defaults to ``True``.
        """
        return data[data['time.month']==month].rolling(time=yrsroll, min_periods=1, center=centered).mean()
    
    def monthly_climo(self, data, yrsroll=30, centered=True):
        """
        Create rolling mean climatology. 
        Performs what xr.DataArray.groupby('time.month').rolling() would do.
        Args:
            data (xarray data array): Weighted mean variable.
            yrsroll (int): Number of years for climatology. Defaults to ``30``.
            centered (boolean): Whether the average is centered. Defaults to ``True``.
        Returns:
            nino_climo with rolling mean computed along months.
        """
        jan = self.roll_climo(data, month=1, yrsroll=yrsroll, centered=centered)
        feb = self.roll_climo(data, month=2, yrsroll=yrsroll, centered=centered)
        mar = self.roll_climo(data, month=3, yrsroll=yrsroll, centered=centered)
        apr = self.roll_climo(data, month=4, yrsroll=yrsroll, centered=centered)
        may = self.roll_climo(data, month=5, yrsroll=yrsroll, centered=centered)
        jun = self.roll_climo(data, month=6, yrsroll=yrsroll, centered=centered)
        jul = self.roll_climo(data, month=7, yrsroll=yrsroll, centered=centered)
        aug = self.roll_climo(data, month=8, yrsroll=yrsroll, centered=centered)
        sep = self.roll_climo(data, month=9, yrsroll=yrsroll, centered=centered)
        boo = self.roll_climo(data, month=10, yrsroll=yrsroll, centered=centered)
        nov = self.roll_climo(data, month=11, yrsroll=yrsroll, centered=centered)
        dec = self.roll_climo(data, month=12, yrsroll=yrsroll, centered=centered)
        nino_climo = xr.concat([jan,feb,mar,apr,may,jun,jul,aug,sep,boo,nov,dec], dim='time').sortby('time')
        return nino_climo
    
    def compute_index(self, data, climo, runningmean=3):
        """
        Compute nino index (sst based).
        Args:
            data (xarray data array): Weighted mean variable.
            climo (xarray data array): Monthly climatology.
            runningmean (int): Final running mean for index. Defaults to ``3``.
                               Select 5 for Nino regions and 3 for ONI.
        """
        anom = data - climo
        anom_rolling = anom.rolling(time=runningmean, min_periods=1, center=False).mean()
        std = anom_rolling.std()
        nino_index = (anom_rolling / std)
        return nino_index

    def check_nino(self, data, lats='TLAT', lons='TLONG'):
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

    def check_nino_percentages(self, index, cutoff=0.4):
        """
        Check percent of data set that contains nino and nina events.
        Args:
            index (xarray data array): Index.
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
        """
        print("Percentage of El Nino events = {:0.1f}%".format(100 * (index.where(index>=cutoff).count() / index.count()).values))
        print("Percentage of La Nina events = {:0.1f}%".format(100 * (index.where(index<=-cutoff).count() / index.count()).values))

    def fast_plot(self, index, cutoff=0.4):
        """
        Quick visualization of index.
        Args:
            index (xarray data array): Index.
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
        """
        index.plot(size=12)
        plt.margins(x=0)
        plt.axhline(0,color='black',lw=0.5)
        plt.axhline(cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.axhline(-cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.show()
        
    def shaded_plot(self, index, cutoff=0.4, title=None):
        """
        Quick shaded visualization of index.
        Args:
            index (xarray data array): Index.
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
            title (str): Title for the figure.
        """
        fig = plt.figure(figsize=(12, 8))
        plt.fill_between(index.time.values, index.where(index>=cutoff).values, cutoff, color='r', alpha=0.8)
        plt.fill_between(index.time.values, index.where(index<=-cutoff).values, -cutoff, color='b', alpha=0.3)
        index.plot(color='black',lw=0.2)
        plt.axhline(0,color='black',lw=0.5)
        plt.axhline(cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.axhline(-cutoff,color='black',linewidth=0.5,linestyle='dotted')
        if title:
            plt.title(title)
        plt.margins(x=0)
        plt.show()

    def nino_cumsum(self, index, cutoff=0.4, title=None):
        """
        Cumulative sum of El Nino, La Nino, and Neutral events over the index data.
        Args:
            index (xarray data array):
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
            title (str): Title for the figure.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes([0.,0.,1.,1.])
        index.where(index>=cutoff, 0).where(index<cutoff, 1).cumsum(dim='time').plot(ax=ax, c='r')
        index.where(index<=-cutoff, 0).where(index>-cutoff, 1).cumsum(dim='time').plot(ax=ax, c='b')
        index.where((index<cutoff)&(index>-cutoff), 0).where((index>=cutoff)|(index<=-cutoff), 1).cumsum(dim='time').plot(ax=ax, c='k')
        if title:
            plt.title(title)
        plt.margins(x=0)
        plt.show()
