import modules.plotting2 as p2
import modules.dataprocessing as dp
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
from pyproj import Proj, transform
import scipy

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
from matplotlib import cm

import numpy as np
import pandas as pd
import modules.week2 as w2
from tqdm import tqdm
import itertools
import time

from scipy.linalg import lstsq

import matplotlib.ticker as ticker


def load_data(variables,projection, temporal_resolution,temporal_decomposition,detrend):
	"""
	Loads in all of the data
	"""

	seaice = 'seaice' in variables
	variables_to_load = [v for v in variables if v in ['u10', 'v10', 'si10', 't2m', 'sst', 'skt', 'ssr', 'sp', 'ssrd']]
	indicies_to_load  = [v for v in variables if v in ['IPO','nina34','nina12','DMI','SAM','meiv2','SOI','SAM']]
	geopotential_to_load = [v for v in variables if v in ['geopotential']]


	x = 10*np.arange(-395000,395000,2500)
	y = 10*np.arange(435000,-395000,-2500)
	time = pd.date_range(start='1979-01-01', end='2020-12-01', freq='MS')
	coords = {'x':x,'y':y,'time':time}
	dims = ['x','y', 'time']
	data = xr.Dataset(coords=coords)
	# temporal anomalies if needed
	if 'anomalous' in temporal_decomposition:
		climatology = data.groupby("time.month").mean("time")
		data = data.groupby("time.month") - climatology

	# temporal resolution
	if 'seasonal' in temporal_resolution:
		data = data.resample(time="QS-DEC").mean()
	elif 'annual' in temporal_resolution:
		data = data.resample(time="YS").mean()

	# Load indicies
	if len(indicies_to_load)>0:
		indicies_data  = p2.load_indicies(indicies_to_load, temporal_resolution)
		# temporal anomalies if needed
		if 'anomalous' in temporal_decomposition:
			climatology = indicies_data.groupby("time.month").mean("time")
			indicies_data = indicies_data.groupby("time.month") - climatology

		ind = indicies_data.copy()
		ind = (ind - ind.mean()) 
		ind =  ind / ind.std()
		indicies_data = ind

		# temporal resolution
		if 'seasonal' in temporal_resolution:
			indicies_data = indicies_data.resample(time="QS-DEC").mean()
		elif 'annual' in temporal_resolution:
			indicies_data = indicies_data.resample(time="YS").mean()

		indicies_data = indicies_data.interp(time=data.time)

		for index in indicies_to_load:
			X = indicies_data[index].data
			X = X.transpose()
			X = np.repeat(X[np.newaxis, np.newaxis,:], len(x), axis=0)
			X = np.repeat(X, len(y), axis = 1)
			dims = ['x','y','time']
			coords = [data.x,
					  data.y,
					  data.time]

			data[index] = xr.DataArray(data   = X, 
									   dims   = dims, 
									   coords = coords).transpose(*dims)
			data[index].attrs["units"] = "stddev"
			

	# Load vairables
	if len(variables_to_load)>0:
		# file = 'download.nc'
		variable_data = xr.Dataset()
		for variable in variables_to_load:
			variable_data[variable] = xr.open_dataset(f'download/{variable}_transformed.nc').__xarray_dataarray_variable__  

		attrs = {variable : variable_data[variable].attrs for variable in variables_to_load}
		# variable_data = variable_data.sel(expver=1)
		print(variable_data)
		variable_data = variable_data.sel(latitude=slice(-40,-90))

		if 'anomalous' in temporal_decomposition:
			climatology = variable_data.groupby("time.month").mean("time")
			variable_data = variable_data.groupby("time.month") - climatology


		# temporal resolution
		if 'seasonal' in temporal_resolution:
			variable_data = variable_data.resample(time="QS-DEC").mean()
		elif 'annual' in temporal_resolution:
			variable_data = variable_data.resample(time="YS").mean()

		variable_data = variable_data.interp(time=data.time)
		# Y, X = [10*np.arange(435000,-395000,-2500),
		# 		10*np.arange(-395000,395000,2500)]
		# x,y = np.meshgrid(X,Y)
		# inProj = Proj(init='epsg:3031')
		# outProj = Proj(init='epsg:4326')
		# x,y = transform(inProj,outProj,x,y)
		# x = x.flatten()
		# y = y.flatten()
		# x[x<0] = x[x<0]+360 
		# # data = data.stack(z=('longitude','latitude'))
		# x = xr.DataArray(x, dims='z')
		# y = xr.DataArray(y, dims='z')
		# variable_data = variable_data.interp(longitude=x, latitude=y, method = 'linear', kwargs={"fill_value": 0.0})
		# for variable in variables_to_load:
		# 	interpolated = variable_data[variable].values.reshape([data.time.size,len(Y),len(X)])
		# 	dims_ = ['time','y','x']

		# 	interpolated = xr.DataArray(data = interpolated, dims=dims_,coords = [data.time,Y,X])
		for variable in variables_to_load:
			print(variable)
			data[variable] = variable_data[variable].transpose(*dims)
			data[variable].attrs = attrs[variable]
	
	if len(geopotential_to_load)>0:
		file = 'geopotential.nc'
		variable_data = xr.open_dataset(file)[geopotential_to_load]

		attrs = {variable : variable_data[variable].attrs for variable in geopotential_to_load}
		variable_data = variable_data.sel(expver=1)
		variable_data = variable_data.sel(latitude=slice(-40,-90))
		if 'anomalous' in temporal_decomposition:
			climatology = variable_data.groupby("time.month").mean("time")
			variable_data = variable_data.groupby("time.month") - climatology

		# temporal resolution
		if 'seasonal' in temporal_resolution:
			variable_data = variable_data.resample(time="QS-DEC").mean()
		elif 'annual' in temporal_resolution:
			variable_data = variable_data.resample(time="YS").mean()

		variable_data = variable_data.interp(time=data.time)

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
		variable_data = variable_data.interp(longitude=x, latitude=y, method = 'linear', kwargs={"fill_value": 0.0})
		for variable in geopotential_to_load:
			interpolated = variable_data[variable].values.reshape([data.time.size,len(Y),len(X)])
			dims_ = ['time','y','x']

			interpolated = xr.DataArray(data = interpolated, dims=dims_,coords = [data.time,Y,X])

			data[variable] = interpolated.transpose(*dims)
			data[variable].attrs = attrs[variable]
			print(data[variable].attrs)

	# load Seaice
	if seaice:
		seaice_data = p2.load_seaice([1], [temporal_resolution], [temporal_decomposition], [detrend]).transpose(*dims)
		for seaicename in seaice_data:
			data['seaice'] = seaice_data[seaicename].interp(time=time)


	return data

from sklearn.linear_model import LinearRegression
def multiple_fast_regression(data, dependant, independant):
	"""
	does regression fast
	"""
	data = data[[dependant]+independant].copy()
	data = data.transpose('x','y','time')
	y = data[dependant]
	X = data[independant]
	times = y.dropna(dim='time').time.values
	data = data.sel(time=times)
	if type(dependant) == str:
		y = data[dependant].values
	else: y = data[dependant].to_array()

	if type(independant) == str:
		X = data[independant].values
	else: X = data[independant].to_array()

	if len(X.shape) ==4:
		newX = np.ones([X.shape[0]+1,*X.shape[1:]])
	else:
		newX = np.ones([2,*X.shape])
	newX[:-1,:] = X
	p = np.empty([*newX.shape[:-1]])

	print(f'Finding coefficients for {independant} against {dependant}')
	time.sleep(0.2)
	for i,j in tqdm(list(itertools.product(range(y.shape[0]), range(y.shape[1])))):
		if np.isnan(np.sum(newX[:,i,j,:])) or np.isnan(np.sum(y[i,j,:])):
			p[:,i,j] = 0
		else:
			p[:,i,j] = lstsq(newX[:,i,j,:].transpose(), y[i,j,:])[0]
		# reg = LinearRegression().fit(newX[:,i,j,:].transpose(), y[i,j,:])
		# p[:,i,j] = reg.coef_


	yhat = y.copy()
	print('Predicting SIC')
	time.sleep(0.2)
	yhat = np.einsum('nijt,nij->ijt',newX,p)
	dims = ['x','y','time']
	coords = [data[coord] for coord in dims]
	yhat = xr.DataArray(data=yhat, dims= dims, coords = coords)
	prediction_name = 'prediction_' + '_'.join(independant)
	data[prediction_name] = yhat
	for i in range (len (independant)):
		param = p[i]
		variable = independant[i]
		dims = ['x','y']
		coords = [data[coord] for coord in dims]
		data['regr_coef_'+variable] = xr.DataArray(data=param, dims= dims, coords = coords)
	data['regr_coef_error'] = xr.DataArray(data=p[-1], dims= dims, coords = coords)
	

	# individual predictions
	for i in range (len (independant)):
		param = p[i]
		variable = independant[i]
		yhat = np.einsum('ijt,ij->ijt',newX[i],p[i])
		dims = ['x','y','time']
		coords = [data[coord] for coord in dims]
		yhat = xr.DataArray(data=yhat, dims= dims, coords = coords)
		data['prediction_'+variable] = yhat

	return data


def plot_coefficients(regression_results, dependant,independant):

	variables = [f'regr_coef_{ind}' for ind in independant]
	N = len(variables)

	fig = plt.figure(figsize=(5*N,5))
	max_ = max([regression_results[indexname].max() for indexname in variables[:]])
	min_ = min([regression_results[indexname].min() for indexname in variables[:]])
	if max_>min_ and max_>0 and min_<0:
		divnorm = TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
	else:
		sys.exit(f'min = {min_.values}, max = {max_.values}, {variables}')
	for i in range(N):
		data = regression_results[variables[i]]
		data = data.where(np.abs(data) != 0.0)
		ax = fig.add_subplot(1,N,i+1, projection = ccrs.SouthPolarStereo())
		ax.contourf(data.x,data.y,data.values.transpose(), cmap = 'RdBu',norm = divnorm, transform = ccrs.SouthPolarStereo())
		ax.set_title(variables[i].replace('_', '\_'))
		ax.coastlines()
	fig.suptitle(f'Regression coefficients')
	fig.subplots_adjust(right=0.95)
	cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
	cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
	cbar.set_label('Regression Coefficients [$\\frac{\%}{\sigma}$]')

	plt.savefig(f'images/week4/coefficients_{dependant}_'+'_'.join(independant)+'.pdf')


def contribution_to_trends(regression_results, dependant,independant):


	variables = [v for v in regression_results if 'regr' in v]
	dependant_trend = regression_results[dependant].sortby('time').polyfit(dim='time', deg=1) * 1e9*60*60*24*365
    
	fig = plt.figure(figsize = (15,5))

	seaice_m = dependant_trend['polyfit_coefficients'].sel(degree=1)
	seaice_m = seaice_m.where(seaice_m !=0)
	max_ = seaice_m.max()
	min_ = seaice_m.min() 
	divnorm = TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
	levels = np.arange(min_,max_,0.001)
	ax = fig.add_subplot(131, projection = ccrs.SouthPolarStereo())
	# Plotting
	contor = ax.contourf(seaice_m.x, seaice_m.y, seaice_m.transpose(), cmap = 'RdBu', levels = levels, norm = divnorm, transform=ccrs.SouthPolarStereo())
	ax.coastlines()
	ax.set_axis_off()
	ax.set_title('Trend')
	cbar = plt.colorbar(contor, format=ticker.FuncFormatter(fmt_colorbar))
	cbar.set_label('Trend (\% yr$^{-1}$)')
	fig.suptitle(f'Seaice trends')

	data = regression_results['prediction_'+'_'.join(independant)].polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
	# print(data)
	data = data.where(abs(data) != 0.0)
	ax2 = fig.add_subplot(132, projection = ccrs.SouthPolarStereo())
	contor = ax2.contourf(data.x, data.y, data.values.transpose(), cmap = 'RdBu', levels = levels, norm = divnorm, transform=ccrs.SouthPolarStereo())
	ax2.coastlines()
	ax2.set_axis_off()
	cbar = plt.colorbar(contor, format=ticker.FuncFormatter(fmt_colorbar))
	cbar.set_label('Trend (\% yr$^{-1}$)')
	ax2.set_title('Predicted Trend')


	residual = seaice_m - data
	residual = residual.where(abs(residual) != 0.0)
	ax = fig.add_subplot(133, projection = ccrs.SouthPolarStereo())
	contor = ax.contourf(residual.x, residual.y, residual.values.transpose(), cmap = 'RdBu', levels = levels, norm = divnorm, transform=ccrs.SouthPolarStereo())
	ax.coastlines()
	ax.set_axis_off()
	cbar = plt.colorbar(contor, format=ticker.FuncFormatter(fmt_colorbar))
	cbar.set_label('Trend (\% yr$^{-1}$)')
	ax.set_title('Residual')
	plt.savefig(f'images/week4/Trend_Contribution_{dependant}_'+'_'.join(independant)+'.pdf')
	plt.show()

def plot_contribution_timeseries(regression_results, dependant,independant):

	if dependant == 'seaice':	
		area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
		seaice_timeseries = (regression_results[dependant]*area).sum(dim=('x','y'))
		predicted_timeseries = (regression_results['prediction_'+'_'.join(independant)]*area).sum(dim=('x','y'))
		predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

	else:
		seaice_timeseries = (regression_results[dependant]).mean(dim=('x','y'))
		predicted_timeseries = (regression_results['prediction_'+'_'.join(independant)]).mean(dim=('x','y'))
		predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)
	
	seaice_m = seaice_timeseries.polyfit(dim='time', deg=1)
	sic_m = seaice_m.polyfit_coefficients.sel(degree=1).values
	b = seaice_m.polyfit_coefficients.sel(degree=0).values
	linear_sic = sic_m * seaice_timeseries.time.values.astype(float) + b

	predicted_m = predicted_timeseries.polyfit(dim='time', deg=1)
	pred_m = predicted_m.polyfit_coefficients.sel(degree=1).values
	b = predicted_m.polyfit_coefficients.sel(degree=0).values
	linear_predict = pred_m * seaice_timeseries.time.values.astype(float) + b



	plt.style.use('stylesheets/timeseries.mplstyle')
	fig  = plt.figure()
	ln1 = plt.plot(regression_results.time, seaice_timeseries,color = '#1EA0AE', label = dependant)
	plt.plot(regression_results.time, linear_sic, color = '#1EA0AE')
	plt.axhline(0,alpha = 0.2)
	ln2 = plt.plot(predicted_timeseries.time, predicted_timeseries, color = '#BF160D', label='Prediction')
	plt.plot(seaice_timeseries.time, linear_predict, color = '#BF160D')

	lines = ln1 + ln2
	labels = [line.get_label() for line in lines]
	plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.15), ncol = 3, loc = 'upper right')
	fig.suptitle('Predicted SIE')
	plt.ylabel('SIE [$km^2$]')
	plt.savefig(f'images/week4/Timeseries_{dependant}_'+'_'.join(independant)+'.pdf')
	plt.show()


def plot_individual_spatial_contributions(regression_results, dependant,independant, proportional = False):
	variables = [f'prediction_{ind}' for ind in independant]

	N = len(variables)
	gradient = regression_results[variables].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365

	if proportional:
		gradient = gradient / (regression_results[dependant].polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365)

	# print(gradient, regression_results[dependant].polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365)
	fig = plt.figure(figsize=(5*N,5))
	# max_ = max([gradient[indexname].max() for indexname in gradient])
	# min_ = min([gradient[indexname].min() for indexname in gradient])
	max_ = 1
	min_ = -1
	if max_>min_ and max_>0 and min_<0:
		divnorm = TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
		levels = np.linspace(min_,max_,15)
	else:
		sys.exit(f'min = {min_.values}, max = {max_.values}')
	for i in range(N):
		data = gradient[variables[i]+'_polyfit_coefficients']
		data = data.where(np.abs(data) != 0.0)
		ax = fig.add_subplot(1,N,i+1, projection = ccrs.SouthPolarStereo())
		contor = ax.contourf(data.x,data.y,data.values.transpose(), cmap = 'RdBu',norm = divnorm, transform = ccrs.SouthPolarStereo(), levels = levels)
		ax.set_title(variables[i].replace('_', '\_'))
		ax.coastlines()
		plt.colorbar(contor, format=ticker.FuncFormatter(fmt_colorbar))
	fig.suptitle(f'Regression Contributions')
	# fig.subplots_adjust(right=0.95)
	# cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
	# cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
	# cbar.set_label('Regression Contributions')
	if proportional:
		plt.savefig(f'images/week4/individual_contributions_proportional_{dependant}_'+'_'.join(independant)+'.pdf')

	else:
		plt.savefig(f'images/week4/individual_contributions_{dependant}_'+'_'.join(independant)+'.pdf')

	plt.show()


def regress(independant            = ['SAM'], 
			dependant              = 'seaice', 
			temporal_resolution    = 'annual',
		 	temporal_decomposition = 'anomalous',
		 	detrend                = 'raw',
		 	do_plotting            = True,
		 	get_stats              = True,
		 	individual_stats       = True,
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
	data = load_data([dependant]+independant,projection, temporal_resolution,temporal_decomposition,detrend)

	regression_results = multiple_fast_regression(data, dependant, independant)
	
	if do_plotting:
		plotting(regression_results, dependant,independant)

	results = None
	if get_stats:
		results = _get_stats(regression_results, dependant)

	individual_results = None
	if individual_stats:
		individual_results = indicies_stats(data, regression_results,dependant, independant,projection, temporal_resolution,temporal_decomposition,detrend)



	return regression_results, results, individual_results

def plotting(regression_results, dependant,independant):

	v = [vi for vi in regression_results if any([ind in vi for ind in independant])] + [dependant]

	regression_results = regression_results[v].copy()
	plot_coefficients(regression_results, dependant,independant)
	# prediction = regression_results[mode]['prediction']
	contribution_to_trends(regression_results, dependant,independant)

	plot_contribution_timeseries(regression_results, dependant,independant)
	plot_individual_spatial_contributions(regression_results, dependant,independant)

def _get_stats(regression_results, dependant, independant):
	v = [vi for vi in regression_results if any([ind in vi for ind in independant])] + [dependant]

	regression_results = regression_results[v].copy()
	cols = ['spatial_correlation','temporal_correlation','Predicted_Trend', 'Actual_Trend']
	results = pd.Series(index=cols)
	
	# Spatial trend correlation
	prediction_trend = regression_results['prediction_'+'_'.join(independant)].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365
	dependant_trend  = regression_results[dependant].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365
	corr = xr.corr(dependant_trend.polyfit_coefficients, prediction_trend.polyfit_coefficients, dim = ('x','y')).values
	results['spatial_correlation'] = corr
	
	# Temporal Correlation
	if dependant == 'seaice':	
		area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
		dependant_timeseries = (regression_results[dependant]*area).sum(dim=('x','y'))
		predicted_timeseries = (regression_results['prediction_'+'_'.join(independant)]*area).sum(dim=('x','y'))
		predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

	else:
		dependant_timeseries = (regression_results[dependant]).mean(dim=('x','y'))
		predicted_timeseries = (regression_results['prediction_'+'_'.join(independant)]).mean(dim=('x','y'))
		predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

	corr = xr.corr(dependant_timeseries, predicted_timeseries).values
	results.loc['temporal_correlation'] = corr

	results['Actual_Trend'] =  dependant_timeseries.polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
	results['Predicted_Trend'] =  predicted_timeseries.polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
	return results

def indicies_stats(bigdata, regression_results,dependant, independant,projection, temporal_resolution,temporal_decomposition,detrend):
	results = pd.DataFrame()
	
	for ind in independant:
		ind = [ind]
		data = bigdata[[dependant,ind[0]]].copy()
		subregression_results = multiple_fast_regression(data, dependant, ind)

		# Spatial trend correlation
		prediction_trend = subregression_results['prediction_'+ind[0]].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365
		dependant_trend  = subregression_results[dependant].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365
		corr = xr.corr(dependant_trend.polyfit_coefficients, prediction_trend.polyfit_coefficients, dim = ('x','y')).values
		results.loc[ind[0],'individual_spatial_correlation'] = corr
		
		# Temporal Correlation
		if dependant == 'seaice':	
			area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
			dependant_timeseries = (subregression_results[dependant]*area).sum(dim=('x','y'))
			predicted_timeseries = (subregression_results['prediction_'+ind[0]]*area).sum(dim=('x','y'))
			predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

		else:
			dependant_timeseries = (subregression_results[dependant]).mean(dim=('x','y'))
			predicted_timeseries = (subregression_results['prediction_'+ind[0]]).mean(dim=('x','y'))
			predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

		corr = xr.corr(dependant_timeseries, predicted_timeseries).values
		results.loc[ind[0],'individual_temporal_correlation'] = corr

		results.loc[ind[0],'individual_Actual_Trend'] =  dependant_timeseries.polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients.data * 1e9*60*60*24*365
		results.loc[ind[0],'individual_Predicted_Trend'] =  predicted_timeseries.polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients.data * 1e9*60*60*24*365
	
		# Spatial trend correlation
		prediction_trend = regression_results['prediction_'+ind[0]].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365
		dependant_trend  = regression_results[dependant].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365
		corr = xr.corr(dependant_trend.polyfit_coefficients, prediction_trend.polyfit_coefficients, dim = ('x','y')).values
		results.loc[ind[0],'spatial_correlation'] = corr
		
		# Temporal Correlation
		if dependant == 'seaice':	
			area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
			dependant_timeseries = (regression_results[dependant]*area).sum(dim=('x','y'))
			predicted_timeseries = (regression_results['prediction_'+ind[0]]*area).sum(dim=('x','y'))
			predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

		else:
			dependant_timeseries = (regression_results[dependant]).mean(dim=('x','y'))
			predicted_timeseries = (regression_results['prediction_'+ind[0]]).mean(dim=('x','y'))
			predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

		corr = xr.corr(dependant_timeseries, predicted_timeseries).values
		results.loc[ind[0],'temporal_correlation'] = corr

		results.loc[ind[0],'Actual_Trend'] =  dependant_timeseries.polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
		results.loc[ind[0],'Predicted_Trend'] =  predicted_timeseries.polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365

	return results


def fmt(x, pos=2):
	if abs(x) <=999 and abs(x)> 10:
		return f'{x:.0f}'
	elif abs(x) <=10:
		return f'{x:.2f}'
	else:
		a, b = ('{:.'+str(pos)+'e}').format(x).split('e')
		b = int(b)
		return r'${} \times 10^{{{}}}$'.format(a, b)
def fmt_colorbar(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def plot_variable_trends(regression_results, dependant,independant):
	"""
	Plots the spatial distribution of each variable.
	"""
	variables = [f'{ind}' for ind in independant]

	N = len(variables)
	gradient = regression_results[variables].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365

	fig = plt.figure(figsize=(5*N,5))
	max_ = max([gradient[indexname].max() for indexname in gradient])
	min_ = min([gradient[indexname].min() for indexname in gradient])
	if max_>min_ and max_>0 and min_<0:
		divnorm = TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
		levels = np.linspace(min_,max_,15)
	else:
		sys.exit(f'min = {min_.values}, max = {max_.values}')
	for i in range(N):
		data = gradient[variables[i]+'_polyfit_coefficients']
		data = data.where(np.abs(data) != 0.0)
		ax = fig.add_subplot(1,N,i+1, projection = ccrs.SouthPolarStereo())
		contor = ax.contourf(data.x,data.y,data.values.transpose(), cmap = 'RdBu',norm = divnorm, transform = ccrs.SouthPolarStereo(), levels = levels)
		ax.set_title(variables[i].replace('_', '\_'))
		ax.coastlines()
		plt.colorbar(contor, format=ticker.FuncFormatter(fmt_colorbar))
	fig.suptitle(f'Trends of variables')

	plt.savefig(f'images/week4/individual_trends_{dependant}_'+'_'.join(independant)+'.pdf')

	plt.show()

def more_plotting(regression_results, dependant,independant):

	v = [vi for vi in regression_results if any([ind in vi for ind in independant])] + [dependant]

	regression_results = regression_results[v].copy()
	plot_coefficients(regression_results, dependant,independant)
	plot_variable_trends(regression_results, dependant,independant)
	plot_individual_spatial_contributions(regression_results, dependant,independant, proportional = True)


