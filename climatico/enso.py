import numpy as np
import xarray as xr
import pandas as pd
from util import pacific_lon, weighted_mean
import matplotlib.pyplot as plt
import warnings

class DefineNino:
    """
    Class instantiation of DefineNino.
    Args:
        nino (str): Nino SST region in Pacific (all lowercase). E.g., ``nino34``.
                    Options include nino12, nino3, nino34, nino4, atl3.
        lats (str): Name of regular/irregular grid latitudes (should have two dims). Defaults to ``TLAT``.
        lons (str): Name of regular/irregular grid longitudes (should have two dims). Defaults to ``TLONG``.
        enso_event (str): ``nino``, ``nina``, or ``neutral`` events based on ``cutoff``. Defaults to ``None``.
        cutoff (float): The +/- nino threshold. Defaults to ``0.5`` for ONI.
                        Use ``0.5`` for nino sst regions.
        strong_cutoff (float): The +/- nino index threshold for strong events. Defaults to ``1.5``.
        runningmean (int): Final running mean for index. Defaults to ``3``.
                           Select 5 for Nino regions and 3 for ONI.
    """
    def __init__(self, nino, sst_name='SST', lats='TLAT', lons='TLONG', enso_event=None, 
                 cutoff=0.5, strong_cutoff=1.5, runningmean=3):
        """
        Initialize.
        """
        self.ninodefined = nino
        self.sst_name = sst_name
        self.lats = lats
        self.lons = lons
        self.enso_event = enso_event
        self.cutoff = cutoff
        self.strong_cutoff = strong_cutoff
        self.runningmean = runningmean

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    
    def compute_index(self, data, climo):
        """
        Compute nino index (sst based).
        Args:
            data (xarray data array): Weighted mean variable.
            climo (xarray data array): Monthly climatology.
        """
        anom = data - climo
        if self.runningmean == 5:
            anom = anom.chunk({'time': 12})
        anom_rolling = anom.rolling(time=self.runningmean, min_periods=1, center=False).mean()
        std = anom_rolling.std()
        nino_index = (anom_rolling / std)
        return nino_index

    def grab_nino(self, ds, squeeze=True):
        """
        Extract nino sst regions that were enso or not.
        Args:
            ds (xarray dataset): SST data.
            squeeze (boolean): Whether to squeeze out any len == 1 dims. Defaults to ``True``.
        """
        ds = self.nino(ds)
        if squeeze:
            da = ds.squeeze()
        # compute weighted mean of sst region
        nino_mean = weighted_mean(da[self.sst_name], lat_name=self.lats)
        # compute monthly climo (running centered 30-yr mean)
        climo = self.monthly_climo(nino_mean, yrsroll=30, centered=True)
        # compute the respective nino index
        index = self.compute_index(data=nino_mean, climo=climo)
        index = index.values
        # extract enso events
        if self.enso_event == "nino":
            enso_events = da.isel(time=np.where(index>=self.cutoff)[0])
        if self.enso_event == "nina":
            enso_events = da.isel(time=np.where(index<=-self.cutoff)[0])
        if self.enso_event == "neutral":
            enso_events = da.isel(time=np.where((index<self.cutoff)&(index>-self.cutoff))[0])
        return enso_events

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

    def check_nino_percentages(self, index):
        """
        Check percent of data set that contains nino and nina events.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
        """
        print("Percentage of El Nino events = {:0.1f}%".format(
            100 * np.around(np.count_nonzero((np.where(index>=self.cutoff, 1, 0))) / np.count_nonzero(~np.isnan(index)),2)))
        print("Percentage of La Nina events = {:0.1f}%".format(
            100 * np.around(np.count_nonzero((np.where(index<=-self.cutoff, 1, 0))) / np.count_nonzero(~np.isnan(index)),2)))

    def check_strong_nino(self, index):
        """
        Check percent of data set that contains nino and nina events.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
        """
        print("Percentage of Strong El Nino events = {:0.1f}%".format(
            100 * np.around(np.count_nonzero((np.where(index>=self.strong_cutoff, 1, 0))) / np.count_nonzero(~np.isnan(index)),2)))
        print("Percentage of Strong La Nina events = {:0.1f}%".format(
            100 * np.around(np.count_nonzero((np.where(index<=-self.strong_cutoff, 1, 0))) / np.count_nonzero(~np.isnan(index)),2)))
        print("Percentage of Strong El Nino from All El Nino = {:0.1f}%".format(
            100 * (np.count_nonzero((np.where(index>=self.strong_cutoff, 1, 0))) / np.count_nonzero((np.where(index>=self.cutoff, 1, 0))))))
        print("Percentage of Strong La Nina from All La Nina = {:0.1f}%".format(
            100 * (np.count_nonzero((np.where(index<=-self.strong_cutoff, 1, 0))) / np.count_nonzero((np.where(index<=-self.cutoff, 1, 0))))))

    def fast_plot(self, index):
        """
        Quick visualization of index.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
        """
        plt.plot(index)
        plt.margins(x=0)
        plt.axhline(0,color='black',lw=0.5)
        plt.axhline(self.cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.axhline(-self.cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.xlabel('Months')
        plt.ylabel('Index')
        plt.show()

    def shaded_plot(self, index, title=None, savefig=None,
                    xticks=[0,1800,3600,5400,7200,9000], xticklabels=[0,150,300,450,600,750],
                    add_yr_lines=None):
        """
        Quick shaded visualization of index.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
            title (str): Title for the figure.
            savefig (str): Directory and figure name to save.
            xticks (list): X ticks. Defaults to freshwater hosing experiment time.
            xticklabels (list): X tick labels. Defaults to freshwater hosing experiment time.
            add_yr_lines (float): Add vertical lines at locations of freshwater hosing addition or reduction.
                                  Defaults to ``None``. Should be e.g., ``[6000]``.
        """
        fig = plt.figure(figsize=(12, 8))
        plt.fill_between(range(index.shape[0]), np.where(index>=self.cutoff,index,np.nan), self.cutoff, color='r', alpha=0.8); 
        plt.fill_between(range(index.shape[0]), np.where(index<=-self.cutoff,index,np.nan), -self.cutoff, color='b', alpha=0.3)
        plt.plot(index, c='k', lw=0.15)
        plt.axhline(0,color='black',lw=0.5)
        plt.axhline(self.cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.axhline(-self.cutoff,color='black',linewidth=0.5,linestyle='dotted')
        plt.axhline(self.strong_cutoff,color='black',linewidth=0.25,linestyle='dotted')
        plt.axhline(-self.strong_cutoff,color='black',linewidth=0.25,linestyle='dotted')
        plt.xticks(xticks, xticklabels)
        plt.xlabel('Years')
        plt.ylabel('Index')
        if add_yr_lines:
            for flux in add_yr_lines:
                plt.axvline(flux, color='k', lw=1.5)
        if title:
            plt.title(title)
        plt.margins(x=0)
        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=200)
            return plt.show()
        if not savefig:
            return plt.show()

    def nino_cumsum(self, index, title=None, savefig=None, 
                    xticks=[0,1800,3600,5400,7200,9000], xticklabels=[0,150,300,450,600,750],
                    add_yr_lines=None):
        """
        Cumulative sum of El Nino, La Nino, and Neutral events over the index data.
        Args:
            index (numpy array): Index eagerly loaded as numpy array (for speed).
            title (str): Title for the figure. Defaults to None.
            savefig (str): Directory and figure name to save. Defaults to None.
            xticks (list): X ticks. Defaults to freshwater hosing experiment time.
            xticklabels (list): X tick labels. Defaults to freshwater hosing experiment time.
            add_yr_lines (float): Add vertical lines at locations of freshwater hosing addition or reduction.
                                  Defaults to ``None``. Should be e.g., ``[6000]``.
        """
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes([0.,0.,1.,1.])
        ax.plot(np.cumsum(np.where(index>=self.cutoff, 1, 0)), c='r')
        ax.plot(np.cumsum(np.where(index<=-self.cutoff, 1, 0)), c='b')
        ax.plot(np.cumsum(np.where((index<self.cutoff)&(index>-self.cutoff), 1, 0)), c='k')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Years')
        ax.set_ylabel('Index')
        if add_yr_lines:
            for flux in add_yr_lines:
                ax.axvline(flux, color='k', lw=1.5)
        if title:
            ax.set_title(title)
        ax.margins(x=0)
        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=200)
            return plt.show()
        if not savefig:
            return plt.show()
