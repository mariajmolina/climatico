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
        lats (str): Name of regular/irregular grid latitudes (should have two dims). Defaults to ``TLAT``.
        lons (str): Name of regular/irregular grid longitudes (should have two dims). Defaults to ``TLONG``.
    """
    def __init__(self, nino, lats='TLAT', lons='TLONG'):
        """
        Initialize.
        """
        self.ninodefined = nino
        self.lats = lats
        self.lons = lons

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

    def nino(self, data):
        """
        Using Nino regional bounds in the Pacific for SST extraction from
        nino_dict().
        Note: Ocean CESM model (pop) uses irregular grid. Therefore, 
              lat and lon coordinates have two dimensions, and should
              be specified in function input.
        Args:
            data (xarray dataset): SST data.
        """
        nino_coords = self.nino_dict()
        return data.where((data[self.lats]<nino_coords[1]) & (data[self.lats]>nino_coords[0]) & \
                          (data[self.lons]>nino_coords[2]) & (data[self.lons]<nino_coords[3]), drop=True)

    def roll_climo(self, data, month, yrsroll=30, centered=True, time='time'):
        """
        Creating rolling 30-year mean climatology.
        Args:
            data (xarray dataarray): Data Array weighted mean already computed.
            month (int): Month for climatology.
            yrsroll (int): Number of years for climatology. Defaults to ``30``.
            centered (boolean): Whether the average is centered. Defaults to ``True``.
            time (str): Time coordinate name. Defaults to ``time``.
        """
        return data[data[f'{time}.month']==month].rolling(time=yrsroll, min_periods=1, center=centered).mean()
    
    def monthly_climo(self, data, yrsroll=30, centered=True, time='time'):
        """
        Create rolling mean climatology. 
        Performs what xr.DataArray.groupby('time.month').rolling() would do.
        Args:
            data (xarray data array): Weighted mean variable.
            yrsroll (int): Number of years for climatology. Defaults to ``30``.
            centered (boolean): Whether the average is centered. Defaults to ``True``.
            time (str): Time coordinate name. Defaults to ``time``.
        Returns:
            nino_climo with rolling mean computed along months.
        """
        jan = self.roll_climo(data, month=1, yrsroll=yrsroll, centered=centered, time=time)
        feb = self.roll_climo(data, month=2, yrsroll=yrsroll, centered=centered, time=time)
        mar = self.roll_climo(data, month=3, yrsroll=yrsroll, centered=centered, time=time)
        apr = self.roll_climo(data, month=4, yrsroll=yrsroll, centered=centered, time=time)
        may = self.roll_climo(data, month=5, yrsroll=yrsroll, centered=centered, time=time)
        jun = self.roll_climo(data, month=6, yrsroll=yrsroll, centered=centered, time=time)
        jul = self.roll_climo(data, month=7, yrsroll=yrsroll, centered=centered, time=time)
        aug = self.roll_climo(data, month=8, yrsroll=yrsroll, centered=centered, time=time)
        sep = self.roll_climo(data, month=9, yrsroll=yrsroll, centered=centered, time=time)
        boo = self.roll_climo(data, month=10, yrsroll=yrsroll, centered=centered, time=time)
        nov = self.roll_climo(data, month=11, yrsroll=yrsroll, centered=centered, time=time)
        dec = self.roll_climo(data, month=12, yrsroll=yrsroll, centered=centered, time=time)
        nino_climo = xr.concat([jan,feb,mar,apr,may,jun,jul,aug,sep,boo,nov,dec], dim=time).sortby(time)
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
        if runningmean == 5:
            anom = anom.chunk({'time': 12})
        anom_rolling = anom.rolling(time=runningmean, min_periods=1, center=False).mean()
        std = anom_rolling.std()
        nino_index = (anom_rolling / std)
        return nino_index

    def check_nino(self, data):
        """
        Quick sanity check function for regional bounds.
        Args:
            data (xarray dataset): SST data.
        """
        print('Check we have the correct spatial extent')
        print('Latitude range: {:.1f} - {:.1f}'.format(data[self.lats].min().values, 
                                                       data[self.lats].max().values))
        print('Longitude range: {:.1f} - {:.1f}'.format(pacific_lon(data[self.lons].min().values), 
                                                        pacific_lon(data[self.lons].max().values)))

    def check_nino_percentages(self, index, cutoff=0.4):
        """
        Check percent of data set that contains nino and nina events.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
        """
        # previous code for xarray data array option
        # print("Percentage of El Nino events = {:0.1f}%".format(100 * (index.where(index>=cutoff).count() / index.count()).values))
        # print("Percentage of La Nina events = {:0.1f}%".format(100 * (index.where(index<=-cutoff).count() / index.count()).values))
        print("Percentage of El Nino events = {:0.1f}%".format(
            100 * np.around(np.count_nonzero((np.where(index>=cutoff, 1, 0))) / np.count_nonzero(~np.isnan(index)),2)))
        print("Percentage of La Nina events = {:0.1f}%".format(
            100 * np.around(np.count_nonzero((np.where(index<=-cutoff, 1, 0))) / np.count_nonzero(~np.isnan(index)),2)))

    def check_strong_nino(self, index, cutoff=0.4, strong_cutoff=1.5):
        """
        Check percent of data set that contains nino and nina events.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
            strong_cutoff (float): The +/- strong nino threshold. Defaults to ``1.5``.
        """
        # previous code for xarray data array option
        # print("Percentage of Strong El Nino events = {:0.1f}%".format(
        #     100 * (index.where(index>=strong_cutoff).count() / np.count_nonzero(~np.isnan(index))).values))
        # print("Percentage of Strong La Nina events = {:0.1f}%".format(
        #     100 * (index.where(index<=-strong_cutoff).count() / np.count_nonzero(~np.isnan(index))).values))
        # print("Percentage of Strong El Nino from All El Nino = {:0.1f}%".format(
        #     100 * (index.where(index>=strong_cutoff).count() / index.where(index>=cutoff).count()).values))
        # print("Percentage of Strong La Nina from All La Nina = {:0.1f}%".format(
        #     100 * (index.where(index<=-strong_cutoff).count() / index.where(index<=-cutoff).count()).values))
        print("Percentage of Strong El Nino events = {:0.1f}%".format(
            100 * np.around(np.count_nonzero((np.where(index>=strong_cutoff, 1, 0))) / np.count_nonzero(~np.isnan(index)),2)))
        print("Percentage of Strong La Nina events = {:0.1f}%".format(
            100 * np.around(np.count_nonzero((np.where(index<=-strong_cutoff, 1, 0))) / np.count_nonzero(~np.isnan(index)),2)))
        print("Percentage of Strong El Nino from All El Nino = {:0.1f}%".format(
            100 * (np.count_nonzero((np.where(index>=strong_cutoff, 1, 0))) / np.count_nonzero((np.where(index>=cutoff, 1, 0))))))
        print("Percentage of Strong La Nina from All La Nina = {:0.1f}%".format(
            100 * (np.count_nonzero((np.where(index<=-strong_cutoff, 1, 0))) / np.count_nonzero((np.where(index<=-cutoff, 1, 0))))))

    def fast_plot(self, index, cutoff=0.4):
        """
        Quick visualization of index.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
        """
        # previous code for xarray data array
        # index.plot(size=12)
        plt.plot(index)
        plt.margins(x=0)
        plt.axhline(0,color='black',lw=0.5)
        plt.axhline(cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.axhline(-cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.xlabel('Months')
        plt.ylabel('Index')
        plt.show()

    def shaded_plot(self, index, cutoff=0.4, strong_cutoff=1.5, title=None, savefig=None,
                    xticks=[0,1800,3600,5400,7200,9000], xticklabels=[0,150,300,450,600,750]):
        """
        Quick shaded visualization of index.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
            title (str): Title for the figure.
            savefig (str): Directory and figure name to save.
            xticks (list): X ticks. Defaults to freshwater hosing experiment time.
            xticklabels (list): X tick labels. Defaults to freshwater hosing experiment time.
        """
        fig = plt.figure(figsize=(12, 8))
        # plt.fill_between(index.time.values, index.where(index>=cutoff).values, cutoff, color='r', alpha=0.8)
        # plt.fill_between(index.time.values, index.where(index<=-cutoff).values, -cutoff, color='b', alpha=0.3)
        # index.plot(color='black', lw=0.2)
        plt.fill_between(range(index.shape[0]), np.where(index>=cutoff,index,np.nan), cutoff, color='r', alpha=0.8); 
        plt.fill_between(range(index.shape[0]), np.where(index<=-cutoff,index,np.nan), -cutoff, color='b', alpha=0.3)
        plt.plot(index, c='k', lw=0.15)
        plt.axhline(0,color='black',lw=0.5)
        plt.axhline(cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.axhline(-cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.axhline(strong_cutoff,color='black',linewidth=0.25,linestyle='dotted')
        plt.axhline(-strong_cutoff,color='black',linewidth=0.25,linestyle='dotted')
        plt.xticks(xticks, xticklabels)
        plt.xlabel('Years')
        plt.ylabel('Index')
        if title:
            plt.title(title)
        plt.margins(x=0)
        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=200)
            return plt.show()
        if not savefig:
            return plt.show()

    def nino_cumsum(self, index, cutoff=0.4, title=None, savefig=None, 
                    xticks=[0,1800,3600,5400,7200,9000], xticklabels=[0,150,300,450,600,750]):
        """
        Cumulative sum of El Nino, La Nino, and Neutral events over the index data.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
            cutoff (float): The +/- nino threshold. Defaults to ``0.4`` for nino sst regions.
                            Use ``0.5`` for oni.
            title (str): Title for the figure. Defaults to None.
            savefig (str): Directory and figure name to save. Defaults to None.
            xticks (list): X ticks. Defaults to freshwater hosing experiment time.
            xticklabels (list): X tick labels. Defaults to freshwater hosing experiment time.
        """
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes([0.,0.,1.,1.])
        # previous xarray data array
        # index.where(index>=cutoff, 0).where(index<cutoff, 1).cumsum(dim='time').plot(ax=ax, c='r')
        # index.where(index<=-cutoff, 0).where(index>-cutoff, 1).cumsum(dim='time').plot(ax=ax, c='b')
        # index.where((index<cutoff)&(index>-cutoff), 0).where((index>=cutoff)|(index<=-cutoff), 1).cumsum(dim='time').plot(ax=ax, c='k')
        ax.plot(np.cumsum(np.where(index>=cutoff, 1, 0)), c='r')
        ax.plot(np.cumsum(np.where(index<=-cutoff, 1, 0)), c='b')
        ax.plot(np.cumsum(np.where((index<cutoff)&(index>-cutoff), 1, 0)), c='k')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Years')
        ax.set_ylabel('Index')
        if title:
            ax.set_title(title)
        ax.margins(x=0)
        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=200)
            return plt.show()
        if not savefig:
            return plt.show()
