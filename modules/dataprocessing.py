"""Contains classes and functions required for data processing.
"""
# Loading relevant modules
import xarray as xr
import numpy  as np
import glob   as glob
import datetime
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.signal

# For printing headings
from modules.misc import print_heading


class dataprocessor(object):
    """Dataprocessor contains the data for processing and coordinates it's standardisation. It contains seaice data from NSIDC, ERA5 data of a variety of variables and index datasets.
    each of these need the following functions to work in this class. (Required if more data is added to the system at a later date)
    
    load_data()              - to load data in.
    temporal_decomposition() - to split into raw, seasonal cycle and anomalous data.
    save_data()              - to save data to folder.
    
    
    Attributes
    ----------
    index_data : TYPE
        Description
    indicies : list
        Which indicies to process.
    load_ERA5 : bool
        Should data from the ERA5 dataset be processed.
    load_indicies : bool
        Should data from the index datasets be processed.
    load_seaice : bool
        Are we processing seaice data.
    processeddatafolder : str
        File path for output processed data.
    rawdatafolder : str
        File path for source data.
    seaice_data : object
        Object containing seaice data.
    variables : list
        Which Era5 variables to load.
    """
    def __init__(self, rawdatafolder = 'data/', processeddatafolder = 'processed_data/'):
        """Generates a dataprocessor object.
        
        Parameters
        ----------
        rawdatafolder : str, optional
            Path to raw data.
        processeddatafolder : str, optional
            Path to output data.
        """
        heading = "Generating a data processor"
        print_heading(heading)

        # Saving datafolder paths to object
        self.rawdatafolder = rawdatafolder
        self.processeddatafolder = processeddatafolder

    def load_data(self, load_seaice = False, load_indicies = False, load_ERA5 = False, indicies = ['SAM'], variables = ['t2m'], minyear = 1979, maxyear = 2020):
        """Adds raw data to the processor object.
        
        Parameters
        ----------
        load_seaice : bool, optional
            Decides if we should load seaice data.
        load_indicies : bool, optional
            Decides if we should load index data.
        load_ERA5 : bool, optional
            Description
        indicies : list, optional
            Which indicies to load as index data.
        variables : list, optional
            which era5 variables to load.
        
        Deleted Parameters
        ------------------
        n : int, optional
            Spatial resolution parameter.
        """

        # Setting which datasets to load for processing
        self.load_seaice   = load_seaice
        self.load_indicies = load_indicies
        self.load_ERA5     = load_ERA5

        # For datasets with multiple variables, which should be loaded.
        self.indicies      = indicies
        self.variables     = variables


        if self.load_seaice:
            heading = "Loading seaice data from NSIDC"
            print_heading(heading)
            self.seaice_data = seaice_data(rawdatafolder       = self.rawdatafolder,
                                           processeddatafolder = self.processeddatafolder)
            self.seaice_data.load_data()
            self.seaice_data.data = self.seaice_data.data.where(self.seaice_data.data > 0.15*250, other = 0.0)
            self.seaice_data.data = self.seaice_data.data.sel(time=slice(f"{minyear}-01-01", f"{maxyear}-12-31"))

        if self.load_indicies:
            heading = f"Loading index data"
            print_heading(heading)
            self.index_data = index_data(rawdatafolder       = self.rawdatafolder,
                                         processeddatafolder = self.processeddatafolder,
                                         indicies = self.indicies)
            self.index_data.load_data()
            self.index_data.data = {index : self.index_data.data[index].sel(time=slice(f"{minyear}-01-01", f"{maxyear}-12-31")) for index in self.index_data.data.keys()}

        if self.load_ERA5:
            heading = f"Loading ECMWF ERA5 data"
            print_heading(heading)
            self.era5_data = era5_data(rawdatafolder       = self.rawdatafolder,
                                        processeddatafolder = self.processeddatafolder)
            self.era5_data.load_data()

    def decompose_and_save(self, resolutions = [1,5,10,20], temporal_resolution = ['monthly', 'seasonal', 'annual'], temporal_decomposition = ['raw', 'anomalous'], detrend = ['raw', 'detrended']):
        """Summary
        
        Parameters
        ----------
        resolutions : list, optional
            Description
        temporal_resolution : list, optional
            Description
        temporal_decomposition : list, optional
            Description
        detrend : list, optional
            Description
        
        Deleted Parameters
        ------------------
        temporal_decomp : list, optional
            Description
        """
        if self.load_seaice:
            self.seaice_data.decompose_and_save(resolutions = resolutions, temporal_resolution = temporal_resolution, temporal_decomposition = temporal_decomposition, detrend = detrend)
            
        if self.load_indicies:
            self.index_data.decompose_and_save(temporal_resolution = temporal_resolution, temporal_decomposition = temporal_decomposition, detrend = detrend)
            
        if self.load_ERA5:
            self.era5_data.decompose_and_save(resolutions = resolutions, temporal_resolution = temporal_resolution, temporal_decomposition = temporal_decomposition, detrend = detrend)
                
        


class seaice_data:
    """Class for seaice data.
    
    Attributes
    ----------
    data : xarray DataArray
        The data for seaice.
    files : list
        list of seaice raw data files.
    output_folder : str
        File path for output data folder.
    source_folder : str
        File path for source data folder.
    
    Deleted Attributes
    ------------------
    n : int
        spatial resolution parameter.
    """
    
    def __init__(self, rawdatafolder = 'data/', processeddatafolder = 'processeddata/', n = 5):
        """Loads the raw data.
        
        Parameters
        ----------
        rawdatafolder : str, optional
            File path for raw data.
        processeddatafolder : str, optional
            File path for processed data.
        n : int, optional
            Spatial resolution parameter.
        """

        self.source_folder = rawdatafolder + 'SIC-monthly/'
        self.output_folder = processeddatafolder + 'SIC/'
        self.files         = glob.glob(self.source_folder+'*.bin')

    def load_data(self):
        """Iterates over seaice files and loads as an object.
        """
        data      = []
        dates     = []
        errorlist = []
        sic_files = self.files
        n         = 1
        for file in sic_files:
            date  = file.split('_')[-4]
            try:
                data += [self.readfile(file)[::n,::n]]
            except ValueError:
                print(file)
                data += [data[-1]]
                errorlist += [(date,file)]

            # try:
            #     date = datetime.datetime.strptime(date, '%Y%m%d')
            # except:
            date = datetime.datetime.strptime(date, '%Y%m')
            dates += [date]
        for date, file in errorlist:
            i = int(np.where(np.array(files) == file)[0])
            data[i] = (data[i-1]+data[i+1])/2
        
        data = np.array(data, dtype = float)
        
        x = 10*np.arange(-395000,395000,2500)[::n]
        y = 10*np.arange(435000,-395000,-2500)[::n]
        x,y = np.meshgrid(x,y)
        
        sie = data[0]
        
        x_coastlines = x.flatten()[sie.flatten()==253]
        y_coastlines = y.flatten()[sie.flatten()==253]
        
        seaice = xr.DataArray(data, 
                              coords={'time': dates,
                                      'x': 10*np.arange(-395000, 395000, 2500)[::n],
                                      'y': 10*np.arange( 435000,-395000,-2500)[::n]},
                              dims=['time', 'y', 'x'])
        seaice.rename('seaice_concentration')

        self.data = seaice
        self.data = self.data.sortby('time')

    def decompose_and_save(self, resolutions = [1,5,10,20], temporal_resolution = ['monthly', 'seasonal', 'annual'], temporal_decomposition = ['raw', 'anomalous'], detrend = ['raw', 'detrended']):
        """Break the data into different temporal splits.
        
        Parameters
        ----------
        resolutions : list, optional
            Description
        temporal_resolution : list, optional
            Description
        temporal_decomposition : list, optional
            Description
        detrend : list, optional
            Description
        """
        dataset = xr.Dataset({'source':self.data.copy()})

        dataset.to_netcdf(self.output_folder+'source.nc')

        heading = 'Splitting the seaice data up'
        print_heading(heading) 

        for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
            print(n, temp_res, temp_decomp, dt)
            # Spatial resolution fix.
            new_data = dataset.source.loc[:,::n,::n].copy()

            # Temporal interpolation for missing data.
            new_data = new_data.resample(time = '1MS').fillna(np.nan)
            new_data = new_data.sortby(new_data.time)
            new_data = new_data.groupby('time.month').apply(lambda group: group.sortby(group.time).interp(method='linear'))

            if temp_res == 'seasonal':
                new_data = new_data[:-1]

            # If anomalous remove seasonal cycle
            if temp_decomp == 'anomalous':
                climatology = new_data.groupby("time.month").mean("time")
                new_data = new_data.groupby("time.month") - climatology


            # temporal averaging
            if temp_res == 'seasonal':
                new_data = new_data.resample(time="QS-DEC").mean()

            elif temp_res == 'annual':
                new_data = new_data.resample(time="YS").mean()
            # plt.plot(new_data.mean(dim = ('x','y')))
            # plt.show()

            # dataset = xr.Dataset({'source':self.data.copy()})
            # dataset[f'{temp_decomp}_{temp_res}_{n}'] = new_data


            # Detrend
            if 'detrended' == dt:
                new_data = new_data.sortby(new_data.time)
                new_data = detrend_data(new_data)


            new_data.name = f'{temp_decomp}_{temp_res}_{n}_{dt}'
            new_data.to_netcdf(self.output_folder + new_data.name +'.nc')

        # self.data = dataset
                
        print_heading('DONE')



    def readfile(self, file):
        """Reads a binary data file and returns the numpy data array.
        
        Parameters
        ----------
        file : str
            File path.
        
        Returns
        -------
        data = (numpy array) data contained in the file.
        """
        with open(file, "rb") as binary_file:
            # Seek a specific position in the file and read N bytes
            binary_file.seek(300, 0)  # Go to beginning of the file
            data = binary_file.read()         # data array
            data = np.array(list(data)).reshape(332, 316)
        return data

class index_data:

    """Class for index data.
    
    Attributes
    ----------
    data : dict
        Description
    indicies : list
        Which indicies to load.
    output_folder : str
        Path to output folder.
    source_folder : str
        Path to source folder.
    """

    def __init__(self, rawdatafolder = 'data/', processeddatafolder = 'processeddata/', indicies = ['SAM']):
        """Loads the raw data.
        
        Parameters
        ----------
        rawdatafolder : str, optional
            File path for raw data.
        processeddatafolder : str, optional
            File path for processed data.
        indicies : list, optional
            which indicies to load.
        """

        self.source_folder = rawdatafolder + 'indicies/'
        self.output_folder = processeddatafolder + 'INDICIES/'

        self.indicies = indicies

    def load_data(self):
        """Summary
        """
        self.data = {}
        if 'DMI' in self.indicies:
            dmi = xr.open_dataset('Data/Indicies/dmi.nc')
            self.data['DMI'] = dmi.DMI

        if 'SAM' in self.indicies:
            sam = np.genfromtxt('Data/Indicies/newsam.1957.2007.txt', skip_header =1, skip_footer = 1)[:,1:]

            index = range(1957, 2020)
            columns = range(1,13)

            sam = pd.DataFrame(data = sam, columns = columns, index = index)
            sam = sam.stack().reset_index()
            sam.columns = ['year', 'month', 'SAM']
            sam['time'] = pd.to_datetime(sam.year*100+sam.month,format='%Y%m')
            sam = sam.set_index('time').SAM
            sam = xr.DataArray(sam)
            self.data['SAM'] = sam

        if 'IPO' in self.indicies:
            ipo = np.genfromtxt('Data/Indicies/tpi.timeseries.ersstv5.data', skip_header = 1, skip_footer = 11)[:,1:]

            index = range(1854, 2021)
            columns = range(1,13)

            ipo = pd.DataFrame(data = ipo, columns = columns, index = index)
            ipo = ipo.stack().reset_index()
            ipo.columns = ['year', 'month', 'IPO']
            ipo['time'] = pd.to_datetime(ipo.year*100+ipo.month,format='%Y%m')
            ipo = ipo.set_index('time').IPO
            ipo = ipo[ipo>-10]
            ipo = xr.DataArray(ipo)
            self.data['IPO'] = ipo

        if 'ENSO' in self.indicies:
            ENSO = np.genfromtxt('Data/Indicies/soi.txt', usecols=np.arange(1,13))

            index = range(1951 , 2021)
            columns = range(1,13)

            ENSO = pd.DataFrame(data = ENSO, columns = columns, index = index)
            ENSO = ENSO.stack().reset_index()
            ENSO.columns = ['year', 'month', 'ENSO']
            ENSO['time'] = pd.to_datetime(ENSO.year*100+ENSO.month,format='%Y%m')
            ENSO = ENSO.set_index('time').ENSO
            ENSO = ENSO[ENSO>-10]
            ENSO = xr.DataArray(ENSO)
            self.data['ENSO'] = ENSO

    def decompose_and_save(self, temporal_resolution = ['monthly', 'seasonal', 'annual'], temporal_decomposition = ['raw', 'anomalous'], detrend = ['raw', 'detrended']):
        """Break the data into different temporal splits.
        
        Parameters
        ----------
        temporal_resolution : list, optional
            Description
        temporal_decomposition : list, optional
            Description
        detrend : list, optional
            Description
        """

        heading = 'Splitting the index data up'
        print_heading(heading) 

        for temp_res, temp_decomp, dt, index in itertools.product(temporal_resolution, temporal_decomposition, detrend, self.data):
            print(temp_res, temp_decomp, dt)
            # Spatial resolution fix.
            new_data = self.data[index].copy()

            # Temporal interpolation for missing data.
            new_data = new_data.resample(time = '1MS').fillna(np.nan)
            new_data = new_data.sortby(new_data.time)
            new_data = new_data.groupby('time.month').apply(lambda group: group.sortby(group.time).interp(method='linear'))

            new_data = new_data.loc[new_data.time.dt.year >= 1979]

            # If anomalous remove seasonal cycle
            if temp_decomp == 'anomalous':
                climatology = new_data.groupby("time.month").mean("time")
                new_data = new_data.groupby("time.month") - climatology


            # temporal averaging
            if temp_res == 'seasonal':
                new_data = new_data.resample(time="QS-DEC").mean()

            elif temp_res == 'annual':
                new_data = new_data.resample(time="YS").mean()
            # plt.plot(new_data.mean(dim = ('x','y')))
            # plt.show()

            # dataset = xr.Dataset({'source':self.data.copy()})
            # dataset[f'{temp_decomp}_{temp_res}_{n}'] = new_data


            # Detrend
            if 'detrended' == dt:
                subdata = new_data.copy()
                subdata = subdata.sortby(subdata.time)
                subdata = subdata.dropna(dim='time')
                subdata = detrend_data(subdata)
                new_data[index] = subdata

            new_dataname = f'{index}_{temp_decomp}_{temp_res}_{dt}'
            new_data.to_netcdf(self.output_folder + new_dataname +'.nc')

        # self.data = dataset
                
        print_heading('DONE')




class era5_data:

    """Class for index data.
    
    Attributes
    ----------
    output_folder : str
        Path to output folder.
    source_folder : str
        Path to source folder.
    variables : list
        Which variables to load.
    
    Deleted Attributes
    ------------------
    n : int
        Spatial resolution parameter.
    """

    def __init__(self, rawdatafolder = 'data/', processeddatafolder = 'processeddata/'):
        """Loads the raw data.
        
        Parameters
        ----------
        rawdatafolder : str, optional
            File path for raw data.
        processeddatafolder : str, optional
            File path for processed data.
        variables : list, optional
            which variables to laod.
        
        Deleted Parameters
        ------------------
        n : int, optional
            Spatial resolution parameter.
        """

        self.source_folder = rawdatafolder + 'ECMWF/'
        self.output_folder = processeddatafolder + 'ERA5/SIC/'


    def load_data(self):
        """Summary
        """
        self.data = xr.open_dataset('data/ERA5-SIC/download.nc')
        self.data = self.data.sel(latitude=slice(0, -90)).siconc
    
    def decompose_and_save(self, resolutions = [1,5,10,20], temporal_resolution = ['monthly', 'seasonal', 'annual'], temporal_decomposition = ['raw', 'anomalous'], detrend = ['raw', 'detrended']):
        """Break the data into different temporal splits.
        
        Parameters
        ----------
        resolutions : list, optional
            Description
        temporal_resolution : list, optional
            Description
        temporal_decomposition : list, optional
            Description
        detrend : list, optional
            Description
        """
        dataset = xr.Dataset({'source':self.data.copy()})

        dataset.to_netcdf(self.output_folder+'source.nc')

        heading = 'Splitting the seaice data up'
        print_heading(heading) 

        for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
            print(n, temp_res, temp_decomp, dt)
            # Spatial resolution fix.
            new_data = dataset.source.loc[:,::n,::n].copy()

            # Temporal interpolation for missing data.
            new_data = new_data.resample(time = '1MS').fillna(np.nan)
            new_data = new_data.sortby(new_data.time)
            new_data = new_data.groupby('time.month').apply(lambda group: group.sortby(group.time).interp(method='linear'))

            if temp_res == 'seasonal':
                new_data = new_data[:-1]

            # If anomalous remove seasonal cycle
            if temp_decomp == 'anomalous':
                climatology = new_data.groupby("time.month").mean("time")
                new_data = new_data.groupby("time.month") - climatology


            # temporal averaging
            if temp_res == 'seasonal':
                new_data = new_data.resample(time="QS-DEC").mean()

            elif temp_res == 'annual':
                new_data = new_data.resample(time="YS").mean()
            # plt.plot(new_data.mean(dim = ('x','y')))
            # plt.show()

            # dataset = xr.Dataset({'source':self.data.copy()})
            # dataset[f'{temp_decomp}_{temp_res}_{n}'] = new_data


            # Detrend
            if 'detrended' == dt:
                new_data = new_data.sortby(new_data.time)
                new_data = new_data.stack(z=('latitude','longitude'))
                new_data = new_data.dropna(dim = 'z', how='all')
                new_data = detrend_data(new_data)
                new_data = new_data.unstack()

            new_data.name = f'{temp_decomp}_{temp_res}_{n}_{dt}'
            new_data.to_netcdf(self.output_folder + new_data.name +'.nc')

        # self.data = dataset
                
        print_heading('DONE')


def detrend_data(t):
    """Summary
    
    Parameters
    ----------
    t : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    return xr.apply_ufunc(scipy.signal.detrend, t,
                          input_core_dims=[['time']],
                          vectorize=True, # !Important!
                          dask='parallelized',
                          output_core_dims=[['time']],
                          )

if __name__ == "__main__":
    import itertools

    # What data to load
    load_seaice   = True
    load_indicies = True
    load_ERA5     = False

    # What indicies and variables
    indicies  = ['SAM','DMI','IPO']
    variables = ['t2m']

    # Resolutions to save data as.
    resolutions = [1,5]
    n = 5

    # temporal averages
    temporal_resolution = ['monthly', 'seasonal', 'annual']

    # temporal_breakdown
    temporal_decomposition = ['raw', 'anomalous']

    # detrendin
    detrend = ['raw', 'detrended']

    # Generate a processor object
    processor = dataprocessor(rawdatafolder = 'data/', processeddatafolder = 'processed_data/')

    # Load in datasets
    processor.load_data(load_seaice   = load_seaice,
                        load_indicies = load_indicies,
                        load_ERA5     = load_ERA5,
                        indicies      = indicies,
                        variables     = variables)

    # Change resolution of data
    processor.decompose_and_save(resolutions            = resolutions,
                                 temporal_resolution    = temporal_resolution,
                                 temporal_decomposition = temporal_decomposition,
                                 detrend                = detrend)