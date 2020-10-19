# """
# Code used for the week2 notebook for my MSc Research.

# Notes from the meeting at the start of the week:

# Individual data
# SAM_meiv2_IPO_DMI
# SAM_IPO_DMI_nina34_SOI

# Keep eye out for new data. (drop at the end)

# Test with -IPO as a bug test

# SAM impact on pressure and temperature

# Contribution to SIE anomalies for 2014 event.

# """

import modules.plotting2 as p2
import modules.dataprocessing as dp
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
from pyproj import Proj, transform
import scipy


import numpy as np
from numba import cuda


def load_variable(temporal_resolution,temporal_decomposition,detrend, variable):
	if variable == 't2m':
		file = 'data/2m-temperature-ERA5/download.nc'
		temp = xr.open_dataset(file).p0001
		temp = temp[:,temp.latitude<-55]

	elif variable == 'pressure':
		file = 'data/ECMWF/download.nc'
		temp = xr.open_dataset(file).sp
		temp = temp[:,temp.latitude<-55]

	if 'anomalous' in temporal_decomposition:
		climatology = temp.groupby("time.month").mean("time")
		temp = temp.groupby("time.month") - climatology

	if 'seasonal' in temporal_resolution:
		temp = temp.resample(time="QS-DEC").mean()
	elif 'annual' in temporal_resolution:
		temp = temp.resample(time="YS").mean()

	if 'detrended' in  detrend:
		temp = temp.sortby(temp.time)
		temp = temp.stack(z=('latitude','longitude'))
		temp = temp.dropna(dim = 'z', how='all')
		temp = dp.detrend_data(temp)
		temp = temp.unstack()

	return temp

def analyse_variable(temporal_resolution,temporal_decomposition,detrend, variable, subindicies):
	temperature = load_variable(temporal_resolution[0],temporal_decomposition[0],detrend[0], variable)
	indicies = p2.load_indicies(subindicies, temporal_resolution[0])
	indicies = p2.normalise_indicies(indicies)

	import xarray as xr
	temperature = xr.Dataset({variable:temperature})
	print(temperature)

	regression_results = p2.multiple_fast_regression(temperature,indicies)

	p2.plot_coefficients(regression_results)

	prediction = regression_results[variable]['prediction']
	p2.contribution_to_trends(temperature, prediction, subindicies, indicies, regression_results)

	p2.plot_contribution_timeseries(temperature ,prediction, indicies, regression_results)

	p2.individual_contribution_to_trends(regression_results, indicies, temperature)


def load_data(variable, projection, temporal_resolution,temporal_decomposition,detrend):
	"""
	Generalised function to load data of any variety from our input datasets and standardise it to the standard coordinates for quick analysis.

	Datasets:
	----------
		1) Index data           (Various sources)
		2) Seaice data          (NSIDC)
		3) Variable (ERA5) data (ECMWF)
	"""

	# Index Data
	if variable in ['IPO','nina34','nina12','DMI','SAM','meiv2','SOI','SAM']:
		# Do index loading stuff
		data = p2.load_indicies(variable, temporal_resolution)[variable]

		# temporal anomalies if needed
		if 'anomalous' in temporal_decomposition:
			climatology = data.groupby("time.month").mean("time")
			data = data.groupby("time.month") - climatology

		# temporal resolution
		if 'seasonal' in temporal_resolution:
			data = data.resample(time="QS-DEC").mean()
		elif 'annual' in temporal_resolution:
			data = data.resample(time="YS").mean()

		# normalise
		ind = data.copy()
		ind = (ind - ind.mean()) 
		ind =  ind / ind.std()
		data = ind

		# apply over area size
		if projection == 'SouthPolarStereo':
			X = data.data
			X = X.transpose()
			X = np.repeat(X[:, np.newaxis, np.newaxis], 316, axis=2)
			X = np.repeat(X, 332, axis = 1)
			dims = ['time','y','x']
			coords = [data.time,
					  10*np.arange(435000,-395000,-2500),
					  10*np.arange(-395000,395000,2500)]

			data = xr.DataArray(data   = X, 
								dims   = dims, 
								coords = coords)
		if projection == 'PlateCarree':
			longitude = np.arange(0,360,0.25)
			latitude = np.arange(-55.25,90.25,0.25)
			X = data.data
			X = X.transpose()
			X = np.repeat(X[:, np.newaxis, np.newaxis], len(latitude), axis=2)
			X = np.repeat(X, len(longitude), axis = 1)
			dims = ['time','longitude','latitude']
			coords = [data.time,
					  longitude,
					  latitude]

			data = xr.DataArray(data   = X, 
								dims   = dims, 
								coords = coords)

	# Seaice Data
	elif variable in ['seaice']: 
		# load seaice data
		data = p2.load_seaice([1], [temporal_resolution], [temporal_decomposition], [detrend])

		# if projection == 'PlateCarree':
		# 	y, x = [10*np.arange(435000,-395000,-2500),
		# 			  10*np.arange(-395000,395000,2500)]
		# 	x, y = np.meshgrid(x,y)
		# 	x = x.flatten()
		# 	y = y.flatten()
		# 	xy = np.array([[x[i],y[i]] for i in len(x)])
		# 	print(xy)
		# 	longitude = np.arange(0,360,0.25)
		# 	latitude = np.arange(-55.25,90.25,0.25)

		# else:
			# data = data


	# variable data
	elif variable in ['t2m','sp', 'u10', 'v10']:
		# Load variable data
		file = 'download.nc'
		data = xr.open_dataset(file)[variable]
		data = data.sel(expver=1)
		data = data[:,data.latitude<-40]

		if projection == 'PlateCarree':
			data = data[:,data.latitude<-55]
		else:
			Y, X = [10*np.arange(435000,-395000,-2500),
					10*np.arange(-395000,395000,2500)]
			x,y = np.meshgrid(X,Y)
			inProj = Proj(init='epsg:3031')
			outProj = Proj(init='epsg:4326')
			x,y = transform(inProj,outProj,x,y)
			x = x.flatten()
			y = y.flatten()
			x[x<0] = x[x<0]+360 
			# data = data.stack(z=('longitude','latitude'))
			x = xr.DataArray(x, dims='z')
			y = xr.DataArray(y, dims='z')
			data = data.interp(longitude=x, latitude=y, method = 'linear', kwargs={"fill_value": 0.0})

			interpolated = data.values.reshape([data.time.size,len(Y),len(X)])
			dims = ['time','y','x']

			data = xr.DataArray(data = interpolated, dims=dims,coords = [data.time,Y,X])
			

		# temporal anomalies if needed
		if 'anomalous' in temporal_decomposition:
			climatology = data.groupby("time.month").mean("time")
			data = data.groupby("time.month") - climatology

		# temporal resolution
		if 'seasonal' in temporal_resolution:
			data = data.resample(time="QS-DEC").mean()
		elif 'annual' in temporal_resolution:
			data = data.resample(time="YS").mean()




	else:
		sys.exit(f'ERROR: Cannot load variable: {variable}')

	return data




def regress(independant            = ['SAM'], 
			dependant              = 'seaice', 
			temporal_resolution    = 'annual',
		 	temporal_decomposition = 'anomalous',
		 	detrend                = 'raw'
		 	):
	"""
	Function to regress a set of datasets on a single variable over Antarctica.
	"""
	
	# Select spatial projection
	# if dependant == 'seaice':
	projection = 'SouthPolarStereo'
	# else:
	# 	projection = 'PlateCarree'

	# Let's start by loading in the dependant variable - this will be a datarray
	dependant_data = load_data(dependant, projection, temporal_resolution, temporal_decomposition, detrend)
	mode = [name for name in dependant_data][0]

	# Now we load the independant variables and save as a dataset
	#   dimensions = 2 spatial (determined by projection) 1 time
	independant_data = {}
	for variable in independant:
		independant_data[variable] = load_data(variable, projection, temporal_resolution, temporal_decomposition, detrend)
	independant_data = xr.Dataset(independant_data)
	independant_data = independant_data.stack(z=('x','y')).dropna(dim='z', how = 'all').dropna(dim='time').unstack().sortby('x').sortby('y')

	x = independant_data.x.sortby('x')
	y = independant_data.y.sortby('y')
	dependant_data = dependant_data.sel(x=x)
	dependant_data = dependant_data.sel(y=y)
	print(dependant_data)
	print(independant_data)


	regression_results = p2.multiple_fast_regression(dependant_data,independant_data)

	p2.plot_coefficients(regression_results)

	prediction = regression_results[mode]['prediction']
	p2.contribution_to_trends(dependant_data, prediction, independant, independant_data, regression_results)

	p2.plot_contribution_timeseries(dependant_data ,prediction, independant_data, regression_results)

	p2.individual_contribution_to_trends(regression_results, independant_data, dependant_data)