"""Contains classes and functions needed for calculating correlations.
"""
import scipy
import xarray as xr
import matplotlib.pyplot as plt
from modules.misc import seaice_area_mean

################################################
#               Helper Functions               #
################################################

def _correlate(x,y):
    """Correlates two variables.
    
    Args:
        x (xarray DataArray): One time series.
        y (xarray DataArray): Another timeseries.
    
    Returns:
        xarray DataArray: Correlation of the two dataarrays.
    """
    return scipy.stats.pearsonr(x,y)[0]

def correlate_variables(x,y,dim='time'):
    """Correlates two variables.
    
    Args:
        x (xarray DataArray): One time series.
        y (xarray DataArray): Another timeseries.
        dim (str, optional): Dimension which we are comparing the datasets.
    
    Returns:
        xarray DataArray: Correlation of the two dataarrays.
    """

    return xr.apply_ufunc(_correlate, x , y,
                          input_core_dims=[[dim], [dim]],
                          vectorize=True, # !Important!
                          dask='parallelized',
                          output_dtypes=[float]
                          )

def _pcalculate(x,y):
    """Calculates the pvalue of two time seires.
    
    Args:
        x (xarray DataArray): One time series.
        y (xarray DataArray): Another timeseries.
    
    Returns:
        xarray DataArray: pvalue for correlation.
    """
    return scipy.stats.pearsonr(x,y)[1]

def calculate_pvalue(x,y,dim='time'):
    """Calculates the pvalue of two time seires.
    
    Args:
        x (xarray DataArray): One time series.
        y (xarray DataArray): Another time series.
        dim (str, optional): Dimension which we are comparing the datasets.
    
    Returns:
        xarray DataArray: pvalue for correlations on certain axis.
    """

    return xr.apply_ufunc(_pcalculate, x , y,
                          input_core_dims=[[dim], [dim]],
                          vectorize=True, # !Important!
                          dask='parallelized',
                          output_dtypes=[float]
                          )
    

################################################
#               Correlator Class               #
################################################

class correlator(object):
    """
    Attributes:
        anomlous (bool, optional): Wheather to load anomalous data.
        detrend (bool, optional): Wheather to load detrended data.
        index_data (dict): Index data.
        indicies (TYPE): Which indicies to process.
        input_folder (str): Where input data can be found.
        outputfolder (str, optional): Where to save output data.
        process_indicies (bool, optional): Decides if we should load seaice data.
        process_seaice (bool, optional): Decides if we should load index data.
        seaice_data (xarray DataArray): Sea ice data.
        spatial_resolution (int, optional): What spatial resolution to load.
        temporal_resolution (str, optional): What temporal resolution to load.
    """

    def __init__(self, process_seaice = True, process_indicies = True, indicies = ['SAM'], anomlous = False, temporal_resolution = 'monthly', spatial_resolution = 1, detrend = False, outputfolder = 'processed_data/correlations/', input_folder = 'processed_data/', seaice_source = 'nsidc', minyear = 1979, maxyear = 2020):
        """
        Args:
            process_seaice (bool, optional): Decides if we should load seaice data.
            process_indicies (bool, optional): Decides if we should load index data.
            indicies (list, optional): Which indicies to load as index data.
            anomlous (bool, optional): Wheather to load anomalous data.
            temporal_resolution (str, optional): What temporal resolution to load.
            spatial_resolution (int, optional): What spatial resolution to load.
            detrend (bool, optional): Wheather to load detrended data.
            outputfolder (str, optional): Where to save output data.
            input_folder (str, optional): Where input data can be found.
        """
        self.process_seaice      = process_seaice
        self.process_indicies    = process_indicies
        self.indicies            = indicies
        self.anomlous            = anomlous
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution  = spatial_resolution
        self.detrend             = detrend
        self.outputfolder        = outputfolder
        self.input_folder        = input_folder
        self.seaice_source       = seaice_source
        self.minyear             = minyear
        self.maxyear             = maxyear

        if self.seaice_source == 'ecmwf':
            self.outputfolder = 'processed_data/ERA5/correlations/'

        self.load_data()

    def load_data(self):
        """Loads data onto the corrolator.
        """
        if self.process_seaice:
            self.load_seaice()
        if self.process_indicies:
            self.load_indicies()

    def load_seaice(self):
        """Loads SIC data onto the correlator.
        """
        if self.anomlous:
            temp_decomp = 'anomalous'
        else:
            temp_decomp = 'raw'

        if self.detrend:
            dt = 'detrended'
        else:
            dt = 'raw'
            
        seaicename = f'{temp_decomp}_{self.temporal_resolution}_{self.spatial_resolution}_{dt}'
        if self.seaice_source == 'nsidc':
            self.seaice_data = xr.open_dataset(self.input_folder + 'SIC/' + seaicename +'.nc')
            self.seaice_data = self.seaice_data[seaicename]
        if self.seaice_source == 'ecmwf':
            self.seaice_data = xr.open_dataset(self.input_folder + 'ERA5/SIC/' + seaicename +'.nc')
            self.seaice_data = self.seaice_data[seaicename]

        self.seaice_data = self.seaice_data.sel(time=slice(f"{self.minyear}-01-01", f"{self.maxyear}-12-31"))


    def load_indicies(self):
        """Loads index data onto the correlator.
        """
        self.index_data = {}

        if self.anomlous:
            temp_decomp = 'anomalous'
        else:
            temp_decomp = 'raw'

        if self.detrend:
            dt = 'detrended'
        else:
            dt = 'raw'
            
        for indexname in self.indicies:
            filename = f'{indexname}_{temp_decomp}_{self.temporal_resolution}_{dt}'
            self.index_data[indexname] = xr.open_dataset(self.input_folder + 'INDICIES/' + filename +'.nc')[indexname]
            self.index_data[indexname] = self.index_data[indexname].sel(time=slice(f"{self.minyear}-01-01", f"{self.maxyear}-12-31"))

    def correlate_mean_sic_indicies(self):

        self.mean_correlations = {}
        self.mean_pvalue       = {}

        for index in self.indicies:
            sic = self.seaice_data.copy()
            ind = self.index_data[index].copy()

            times = list(set(set(sic.time.values) & set(ind.time.values)))
            if 'x' in sic.dims:
                sic = sic.sel(time=times)
                sic = seaice_area_mean(sic,1)
            else:
                sic = sic.sel(time=times).mean(dim=('longitude','latitude'))
            ind = ind.sel(time=times)

            sic = sic.sortby(sic.time)
            ind = ind.sortby(ind.time)

            self.mean_correlations[index] = correlate_variables(sic, ind).compute().values
            self.mean_pvalue[index]       =    calculate_pvalue(sic, ind).compute().values

    def correlate_spatial_sic_indicies(self):

        self.spatial_correlations = xr.Dataset()
        self.spatial_pvalue       = xr.Dataset()

        for index in self.indicies:
            sic = self.seaice_data.copy()
            ind = self.index_data[index].copy()

            times = list(set(set(sic.time.values) & set(ind.time.values)))
            sic = sic.sel(time=times)
            ind = ind.sel(time=times)

            sic = sic.sortby(sic.time)
            ind = ind.sortby(ind.time)

            # dict_spatial_pvalue[index]       =    calculate_pvalue(sic, ind).compute().values

            
            if self.seaice_source == 'ecmwf':
                sic = sic.stack(z=('latitude','longitude'))
                sic = sic.dropna(dim = 'z', how='all')

            correlation = sic.mean(dim = 'time').copy()
            correlation.values = correlate_variables(sic, ind).compute().values
            correlation.name = index

            pvalue = sic.mean(dim = 'time').copy()
            pvalue.values = calculate_pvalue(sic, ind).compute().values
            pvalue.name = index
            
            if self.seaice_source == 'ecmwf':
                sic = sic.unstack()
                correlation = correlation.unstack()
                pvalue = pvalue.unstack()

            self.spatial_correlations[index] = correlation
            self.spatial_pvalue[index] = pvalue





    def save_data(self):
        if self.anomlous:
            temp_decomp = 'anomalous'
        else:
            temp_decomp = 'raw'

        if self.detrend:
            dt = 'detrended'
        else:
            dt = 'raw'
            
        filename = f'{temp_decomp}_{self.temporal_resolution}_{dt}_{self.spatial_resolution}'
        xarray = xr.Dataset(self.mean_correlations)
        xarray.to_netcdf(self.outputfolder + 'single/corr_' + filename + '.nc')
        xarray = xr.Dataset(self.mean_pvalue)
        xarray.to_netcdf(self.outputfolder + 'single/pval_' + filename + '.nc')

        self.spatial_correlations.to_netcdf(self.outputfolder + 'spatial/corr_' + filename + '.nc')
        self.spatial_pvalue.to_netcdf(self.outputfolder + 'spatial/pval_' + filename + '.nc')