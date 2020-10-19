import xarray as xr

class data:

	def __init__(self, type_               = 'seaice',
					   temp_decomp         = 'anomalous',
					   temporal_resolution = 'seasonal',
					   detrend             =  False,
					   seaice_source       = 'nsidc',
					   indicies            = ['SAM', 'ENSO', 'IPO', 'DMI'],
					   spatial_resolution  = 1,
					   minyear             = 1979,
					   maxyear             = 2019
					   ):
		# Setting the parameter values for identifying which data to use
		self.type_               = type_
		self.temp_decomp         = temp_decomp
		self.temporal_resolution = temporal_resolution
		self.detrend             = detrend
		self.spatial_resolution  = spatial_resolution
		self.seaice_source       = seaice_source
		self.indicies            = indicies
		self.input_folder        = 'processed_data/'
		self.minyear             = minyear
		self.maxyear             = maxyear

		# load in the required data
		self.data = self.load_data()

	def load_data(self):

		if self.detrend:
			dt = 'detrended'
		else:
			dt = 'raw'

		if self.type_ == 'seaice':
			seaicename = f'{self.temp_decomp}_{self.temporal_resolution}_{self.spatial_resolution}_{dt}'
			if self.seaice_source == 'nsidc':
				self.seaice_data = xr.open_dataset(self.input_folder + 'SIC/' + seaicename +'.nc')
				self.seaice_data = self.seaice_data[seaicename]
			if self.seaice_source == 'ecmwf':
				self.seaice_data = xr.open_dataset(self.input_folder + 'ERA5/SIC/' + seaicename +'.nc')
				self.seaice_data = self.seaice_data[seaicename]
			self.seaice_data = self.seaice_data.sel(time=slice(f"{self.minyear}-01-01", f"{self.maxyear}-12-31"))
			self.seaice_data = self.seaice_data/250


		if self.type_ == 'indicies':
			self.index_data = {}
			for indexname in self.indicies:
				filename = f'{indexname}_{self.temp_decomp}_{self.temporal_resolution}_{dt}'
				self.index_data[indexname] = xr.open_dataset(self.input_folder + 'INDICIES/' + filename +'.nc')[indexname]
				self.index_data[indexname] = self.index_data[indexname].sel(time=slice(f"{self.minyear}-01-01", f"{self.maxyear}-12-31"))


		if self.type_ == 'regressions':
			pass

		if self.type_ == 'correlations':
			pass

	def derrive_data(self):
		pass