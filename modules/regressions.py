"""Summary
"""
import xarray as xr
import scipy
from numba import njit,jit, cuda
import numpy as np

from modules.misc import seaice_area_mean

@njit
def fastlinregress(ind, sic, shape):
	if sic.ndim == 1:
		mean_ind = np.mean(ind)
		mean_sic = np.mean(sic)
		numerator = 0.0
		denominator = 0.0
		for n in range(shape[0]):
			numerator = numerator + (ind[n]-mean_ind) * (sic[n] - mean_sic)
			denominator = denominator + (ind[n]-mean_ind)**2
		slope = numerator / denominator
		b = mean_sic - slope * mean_ind
	if sic.ndim == 3:
		slope = np.empty(sic.shape[1:])
		b = np.empty(sic.shape[1:])

		mean_ind = np.mean(ind)
		denominator = 0.0

		for n in range(shape[0]):
			denominator = denominator + (ind[n]-mean_ind)**2
		for i in range(shape[1]):
			for j in range(shape[2]):
				mean_sic = np.mean(sic[:,i,j])
				numerator = 0.0
				for n in range(shape[0]):
					numerator = numerator + (ind[n]-mean_ind) * (sic[n,i,j] - mean_sic)
				slope[i,j] = numerator / denominator
				b[i,j] = mean_sic - slope[i,j] * mean_ind
	return slope, b


def linear_model(x, a, b, c, d, e):
			"""Summary
			
			Args:
				x (TYPE): Description
				a (TYPE): Description
				b (TYPE): Description
				c (TYPE): Description
				d (TYPE): Description
			
			Returns:
				TYPE: Description
			"""
			return e + a*x[0] + b*x[1] + c*x[2] + d*x[3]


class regressor(object):
	"""docstring for regressor
	
	Attributes:
		anomlous (TYPE): Description
		detrend (TYPE): Description
		index_data (dict): Description
		indicies (TYPE): Description
		input_folder (TYPE): Description
		mean_pvalue (dict): Description
		mean_regressions (dict): Description
		outputfolder (TYPE): Description
		process_indicies (TYPE): Description
		process_seaice (TYPE): Description
		seaice_data (TYPE): Description
		spatial_resolution (TYPE): Description
		temporal_resolution (TYPE): Description
	"""
	def __init__(self, process_seaice = True, process_indicies = True, indicies = ['SAM'], anomlous = False, temporal_resolution = 'monthly', spatial_resolution = 1, detrend = False, outputfolder = 'processed_data/correlations/', input_folder = 'processed_data/', seaice_source='nsidc', minyear = 1979, maxyear = 2020):
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
			self.outputfolder = 'processed_data/ERA5/regressions/'

		self.load_data()
		self.normalise_data()

	def load_data(self):
		"""Loads data onto the corrolator.
		"""
		if self.process_seaice:
			self.load_seaice()
		if self.process_indicies:
			self.load_indicies()


	def normalise_data(self):
		"""Summary
		"""
		for index in self.indicies:
			ind = self.index_data[index].copy()
			ind = (ind - ind.mean()) 
			ind =  ind / ind.std()
			self.index_data[index] = ind.copy()
			del ind

		# self.seaice_data = (self.seaice_data - self.seaice_data.mean()) 
		# self.seaice_data =  self.seaice_data / self.seaice_data.std()

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

	def regress_mean_sic_indicies(self):
		"""regresses mean sic against indicies.
		"""
		self.mean_regressions = {}
		self.mean_pvalue      = {}
		self.mean_rvalue      = {}
		self.mean_b           = {}
		self.mean_std_err     = {}

		for index in self.indicies:
			sic = self.seaice_data.copy()
			ind = self.index_data[index].copy()

			times = list(set(set(sic.time.values) & set(ind.time.values)))
			sic = sic.sel(time=times)
			if self.seaice_source == 'nsidc':
				sic = seaice_area_mean(sic,1)
			if self.seaice_source == 'ecmwf':
				sic = sic.mean(dim = ('longitude','latitude'))
			ind = ind.sel(time=times)

			sic = sic.sortby(sic.time)
			ind = ind.sortby(ind.time)

			m, b, r_value, p_value, std_err = scipy.stats.linregress(ind, sic)
			self.mean_regressions[index] = m
			self.mean_pvalue[index]      = p_value
			self.mean_rvalue[index]      = r_value
			self.mean_b[index]           = b
			self.mean_std_err[index]     = std_err

	def regress_spatial_sic_indicies(self):

		self.spatial_regressions = xr.Dataset()
		self.spatial_pvalue      = xr.Dataset()
		self.spatial_rvalue      = xr.Dataset()
		self.spatial_b           = xr.Dataset()
		self.spatial_std_err     = xr.Dataset()

		for index in self.indicies:
			sic = self.seaice_data.copy()
			ind = self.index_data[index].copy()

			times = list(set(set(sic.time.values) & set(ind.time.values)))
			sic = sic.sel(time=times)
			ind = ind.sel(time=times)

			sic = sic.sortby(sic.time)
			ind = ind.sortby(ind.time)

			sic = sic.transpose("time",...)

			m, b, r_value, p_value, std_err = xr.apply_ufunc(scipy.stats.linregress, ind, sic,
															  input_core_dims=[['time'], ['time']],
															  vectorize=True, # !Important!
															  dask='parallelized',
															  output_dtypes=[float]*5,
															  output_core_dims=[[]]*5
															  )

			regression = sic.mean(dim = 'time').copy()
			regression.values = m
			regression.name = index
			self.spatial_regressions[index] = regression

			pval = sic.mean(dim = 'time').copy()
			pval.values = p_value
			pval.name = index
			self.spatial_pvalue[index] = pval

			rvalue = sic.mean(dim = 'time').copy()
			rvalue.values = r_value
			rvalue.name = index
			self.spatial_rvalue[index] = rvalue

			bval = sic.mean(dim = 'time').copy()
			bval.values = b
			bval.name = index
			self.spatial_b[index] = bval

			stderr = sic.mean(dim = 'time').copy()
			stderr.values = std_err
			stderr.name = index
			self.spatial_std_err[index] = stderr

	def multiple_regression(self):

		sic = self.seaice_data.copy()
		indicies = [self.index_data[index].copy() for index in self.indicies]

		times = list(set.intersection(set(sic.time.values), *(set(indicies[i].time.values)for i in range(len(indicies)))))

		sic = sic.sel(time=times).sortby('time')
		if self.seaice_source == 'nsidc':
				sic = seaice_area_mean(sic,1)
		if self.seaice_source == 'ecmwf':
			sic = sic.mean(dim = ('longitude','latitude'))
		indicies = [ind.sel(time=times).sortby('time') for ind in indicies]

		p0 = [0]*5

		params, covariances = scipy.optimize.curve_fit(linear_model, indicies, sic, p0)
		
		multiple_regressions = {self.indicies[i]:params[i] for i in range(len(params)-1)}
		multiple_regressions['error'] = params[-1]
		self.multiple_covariances = xr.DataArray(dims = ['indicies_1', 'indicies_2'], coords = {'indicies_1':self.indicies + ['error'] ,'indicies_2':self.indicies + ['error']}, data = covariances)

		self.multiple_regressions = xr.Dataset(multiple_regressions)

	def multiple_spatial_regression(self):

		sic = self.seaice_data.copy()
		indicies = [self.index_data[index].copy() for index in self.indicies]

		times = list(set.intersection(set(sic.time.values), *(set(indicies[i].time.values)for i in range(len(indicies)))))

		sic = sic.sel(time=times).sortby('time')
		if self.seaice_source == 'ecmwf':
			sic = sic.stack(z=('latitude','longitude'))
			sic = sic.dropna(dim = 'z', how='all')
		indicies = [ind.sel(time=times).sortby('time') for ind in indicies]

		new_indicies = xr.Dataset({v.name:v for v in indicies}).to_array(dim='variable')
		p0 = [0]*5

		def apply_curvefit(linear_model, newvariables, seaice):
			"""Applies the linear fitting to the data and returns the parameters.
			
			Args:
				linear_model (TYPE): The plotting model to use
				newvariables (TYPE): The variables to go in the model
				seaice (TYPE): The dependant variable.
			
			Returns:
				TYPE: Description
			"""
			params, covariances = scipy.optimize.curve_fit(linear_model, newvariables.transpose(), seaice)
			a, b, c, d, e  = params
			return a, b, c, d, e

		params = xr.apply_ufunc(apply_curvefit, 
									 linear_model, new_indicies, sic,
									 input_core_dims=[[], ['time','variable'], ['time']],
									 vectorize=True, # !Important!
									 dask='parallelized',
									 output_dtypes=[float]*5,
									 output_core_dims=[[]]*5
									 )

		if self.seaice_source == 'ecmwf':
			sic = sic.unstack()
			params = [p.unstack() for p in params]
		multiple_regressions = {self.indicies[i]:params[i] for i in range(len(params)-1)}
		multiple_regressions['error'] = params[-1]

		self.params = xr.Dataset(multiple_regressions)

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

		# Single regressions
		xarray = xr.Dataset(self.mean_regressions)
		xarray.to_netcdf(self.outputfolder + 'single/regr_' + filename + '.nc')
		xarray = xr.Dataset(self.mean_pvalue)
		xarray.to_netcdf(self.outputfolder + 'single/pval_' + filename + '.nc')
		xarray = xr.Dataset(self.mean_rvalue)
		xarray.to_netcdf(self.outputfolder + 'single/rval_' + filename + '.nc')
		xarray = xr.Dataset(self.mean_b)
		xarray.to_netcdf(self.outputfolder + 'single/bval_' + filename + '.nc')
		xarray = xr.Dataset(self.mean_std_err)
		xarray.to_netcdf(self.outputfolder + 'single/std_err_' + filename + '.nc')

		# Single multiple regressions
		self.spatial_regressions.to_netcdf(self.outputfolder + 'spatial/regr_' + filename + '.nc')
		self.spatial_pvalue.to_netcdf(self.outputfolder + 'spatial/pval_' + filename + '.nc')
		self.spatial_rvalue.to_netcdf(self.outputfolder + 'spatial/rval_' + filename + '.nc')
		self.spatial_b.to_netcdf(self.outputfolder + 'spatial/bval_' + filename + '.nc')
		self.spatial_std_err.to_netcdf(self.outputfolder + 'spatial/std_err_' + filename + '.nc')

		# Spatial regressions
		self.multiple_regressions.to_netcdf(self.outputfolder + 'single_multiple/regr_' + filename + '.nc')
		xarray = xr.Dataset({'cov' : self.multiple_covariances})
		xarray.to_netcdf(self.outputfolder + 'single_multiple/cov_' + filename + '.nc')

		# Spatial multiple regressions
		self.params.to_netcdf(self.outputfolder + 'spatial_multiple/regr_' + filename + '.nc')