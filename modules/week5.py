import numpy as np
import datetime
import xarray as xr
from pyproj import Proj, transform
import xarray as xr
from tqdm import tqdm
import itertools
import scipy
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
from matplotlib import cm
import matplotlib.ticker as ticker
import pandas as pd
from modules import w4
import time

from scipy.linalg import lstsq
import sys

# For printing headings
from modules.misc import print_heading
from modules import misc


def process_seaice(files):
	print_heading(f'Loading seaice data from {len(files):.0f} files')
	data      = []
	dates     = []
	errorlist = []
	sic_files = files
	n         = 1
	for file in sic_files:
		date  = file.split('_')[-4]
		try:
			data += [readfile(file)[::n,::n]]
		except ValueError:
			print(file)
			data += [data[-1]]
			errorlist += [(date,file)]

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

	data = seaice
	data = data.sortby('time')

	return data

def readfile(file):
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

def process_variables():
	Y, X = [10*np.arange(435000,-395000,-2500),
			10*np.arange(-395000,395000,2500)]
	x,y = np.meshgrid(X,Y)
	inProj = Proj(init='epsg:3031')
	outProj = Proj(init='epsg:4326')
	x,y = transform(inProj,outProj,x,y)
	x = x.flatten()
	y = y.flatten()
	x[x<0] = x[x<0]+360 
	x = xr.DataArray(x, dims='z')
	y = xr.DataArray(y, dims='z')

	files = glob('data/ERA5/*')
	files = [f for f in files if 'geopotential' not in f]
	print(files)
	for file in files:
		data = xr.open_dataset(file)
		data = data.sel(latitude=slice(-40,-90)).sel(expver=1)
		# data = data.resample(time="YS").mean()
		variable = [v for v in data][0]
		if 'level' in data.dims:
			dims_ = ['time','y','x','level']
			newdata = xr.DataArray(dims=dims_,coords = [data.time,Y,X, data.level])
		else:
			dims_ = ['time','y','x']
			newdata = xr.DataArray(dims=dims_,coords = [data.time,Y,X])
		for time in tqdm(data.time.values):
			subdata = data[variable].sel(time=time)
			variable_data = subdata.interp(longitude=x, latitude=y, method = 'linear', kwargs={"fill_value": 0.0})
			if 'level' in data.dims:
				newdata.loc[newdata.time==time] = variable_data.values.reshape([1,len(Y),len(X),3])
			else:
				newdata.loc[newdata.time==time] = variable_data.values.reshape([1,len(Y),len(X)])
		newdata.name = variable
		newdata.attrs = data[variable].attrs
		if 'level' in newdata.dims:
			for level in newdata.level:
				subdata = newdata.sel(level=level).copy()
				subdata.to_netcdf(f'processed_data/{variable}_{level.values}.nc')
		else:
			newdata.to_netcdf(f'processed_data/{variable}.nc')
		print(f'{variable} processed')


def sort_axes(data):
	for dim in data.dims:
		data = data.sortby(dim)
	return data

def find_anomalies(data):
	"""Finds the monthly anomaly in any data"""
	climatology = data.groupby("time.month").mean("time")
	data = data.groupby("time.month") - climatology
	return data

def detrend(data):
	"""Removes trend in data"""
	return xr.apply_ufunc(scipy.signal.detrend, data,
						  input_core_dims=[['time']],
						  vectorize=True, # !Important!
						  dask='parallelized',
						  output_core_dims=[['time']],
						  )
def normalise_indepenant(data, dependant=None):
	"""normalizes data"""
	independent = [v for v in data if v != dependant]
	subdata = data[independent]
	mean_vec = subdata.mean(dim='time')
	std_vec = subdata.std(dim='time')
	subdata =  (subdata-mean_vec)/std_vec
	subdata = subdata.fillna(0)
	for ind in independent:
		data[ind] = subdata[ind]
	return data

def yearly_average(data):
	return data.resample(time="YS").mean()

def seasonal_average(data):
	offset = pd.offsets.QuarterBegin(startingMonth=12)
	return data.resample(time=offset).mean()

def clean_axis(data, dim='time'):
	dims_to_stack = [d for d in data.dims if d !=dim]
	return data.stack(z = dims_to_stack).dropna(dim=dim).unstack()



def plot_coefficients(regression_results, dependant,independant,folder='week5'):

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
		ax.set_title(variables[i])
		ax.coastlines()
	fig.suptitle(f'Regression coefficients')
	fig.subplots_adjust(right=0.95)
	cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
	cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
	cbar.set_label('Regression Coefficients [$\\frac{\%}{\sigma}$]')

	# plt.savefig(f'images/{folder}/coefficients_{dependant}_'+'_'.join(independant)+'.pdf', dpi=500)
	misc.savefigures(folder=f'images/{folder}',
                     filename=f'coefficients_{dependant}_'+'_'.join(independant))


def contribution_to_trends(regression_results, dependant,independant,folder='week5'):


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
	# plt.savefig(f'images/{folder}/Trend_Contribution_{dependant}_'+'_'.join(independant)+'.pdf', dpi=500)
	misc.savefigures(folder=f'images/{folder}',
                     filename=f'Trend_Contribution_{dependant}_'+'_'.join(independant))
	plt.show()

def plot_contribution_timeseries(regression_results, dependant,independant,folder='week5'):

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
	# plt.savefig(f'images/{folder}/Timeseries_{dependant}_'+'_'.join(independant)+'.pdf', dpi=500)
	misc.savefigures(folder=f'images/{folder}',
                     filename=f'Timeseries_{dependant}_'+'_'.join(independant))
	plt.show()


def plot_individual_spatial_contributions(regression_results, dependant,independant, proportional = False,folder='week5'):
	variables = [f'prediction_{ind}' for ind in independant]

	N = len(variables)
	gradient = regression_results[variables].polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365

	if proportional:
		gradient = gradient / (regression_results[dependant].polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365)

	# print(gradient, regression_results[dependant].polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365)
	fig = plt.figure(figsize=(5*N,5))
	max_ = max([gradient[indexname].max() for indexname in gradient])
	min_ = min([gradient[indexname].min() for indexname in gradient])
	# max_ = 1
	# min_ = -1
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
		ax.set_title(variables[i])
		ax.coastlines()
		plt.colorbar(contor, format=ticker.FuncFormatter(fmt_colorbar))
	fig.suptitle(f'Regression Contributions')
	# fig.subplots_adjust(right=0.95)
	# cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
	# cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
	# cbar.set_label('Regression Contributions')
	if proportional:
		# plt.savefig(f'images/{folder}/individual_contributions_proportional_{dependant}_'+'_'.join(independant)+'.pdf', dpi=500)
		misc.savefigures(folder=f'images/{folder}',
                    	 filename=f'individual_contributions_proportional_{dependant}_'+'_'.join(independant))

	else:
		# plt.savefig(f'images/{folder}/individual_contributions_{dependant}_'+'_'.join(independant)+'.pdf', dpi=500)
		misc.savefigures(folder=f'images/{folder}',
                    	 filename=f'individual_contributions_{dependant}_'+'_'.join(independant))

	plt.show()

def plotting(regression_results, dependant,independant,folder='week5'):

	v = [vi for vi in regression_results if any([ind in vi for ind in independant])] + [dependant]

	regression_results = regression_results[v].copy()
	plot_coefficients(regression_results, dependant,independant, folder=folder)
	# prediction = regression_results[mode]['prediction']
	contribution_to_trends(regression_results, dependant,independant, folder=folder)

	plot_contribution_timeseries(regression_results, dependant,independant, folder=folder)
	plot_individual_spatial_contributions(regression_results, dependant,independant, folder=folder)

def _get_stats(regression_results, dependant, independant,folder='week5'):
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

def plot_variable_trends(regression_results, dependant,independant, folder='week5'):
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
		ax.set_title(variables[i])
		ax.coastlines()
		plt.colorbar(contor, format=ticker.FuncFormatter(fmt_colorbar))
	fig.suptitle(f'Trends of variables')

	# plt.savefig(f'images/{folder}/individual_trends_{dependant}_'+'_'.join(independant)+'.pdf', dpi=500)
	misc.savefigures(folder=f'images/{folder}',
                	 filename=f'individual_trends_{dependant}_'+'_'.join(independant))

	plt.show()

def more_plotting(regression_results, dependant,independant, folder='week5'):

	v = [vi for vi in regression_results if any([ind in vi for ind in independant])] + [dependant]

	regression_results = regression_results[v].copy()
	plot_coefficients(regression_results, dependant,independant, folder=folder)
	plot_variable_trends(regression_results, dependant,independant, folder=folder)
	plot_individual_spatial_contributions(regression_results, dependant,independant, proportional = False, folder=folder)




def main(dependant, independant):
	files = glob('processed_data/*')
	files = [f for f in files if '_' not in f.split('\\')[1]]
	ds = xr.open_mfdataset(files)

	# Preprocess the data
	ds = (ds
		  .pipe(find_anomalies)
		  .pipe(yearly_average)
		  .pipe(normalise_indepenant, dependant=dependant)
		 ).compute()

	ds = ds.sel(time=slice('1979-01-01','2019-12-31'))
	print(ds)

	v = [v for v in ds]
	correlation_matrix = pd.DataFrame(index=v,columns=v, dtype=np.float64)
	for v1,v2 in tqdm(list(itertools.product(v,v))):
		vec1 = ds[v1].mean(dim=('x','y'))
		vec2 = ds[v2].mean(dim=('x','y'))
		correlation_matrix.loc[v1,v2]=xr.corr(vec1,vec2).values
		
	def significant_bold(val, sig_level=0.9):
		bold = 'bold' if val > sig_level or val < -sig_level else ''
		return 'font-weight: %s' % bold
	print(correlation_matrix.style.applymap(significant_bold,sig_level=0.9))


	plt.pcolormesh(v,v,correlation_matrix.transpose())
	plt.colorbar()
	plt.show()

	regression_results = w4.multiple_fast_regression(ds, dependant, independant)
	plotting(regression_results, dependant, independant)
	more_plotting(regression_results, dependant, independant)


from sklearn.linear_model import LinearRegression
def multiple_fast_regression(data, dependant, independant):
	"""
	does regression fast
	"""
	data = data[[dependant]+independant].copy()
	data = data.transpose('x','y','time')
	y = data[dependant]
	X = data[independant]
	if y.name == 'sic':
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

from modules import p2

def load_indicies(indicies_to_load):
	indicies_data  = p2.load_indicies(indicies_to_load, 'monthly')
	
	# temporal anomalies if needed
	y, x = [10*np.arange(435000,-395000,-2500),
			10*np.arange(-395000,395000,2500)]

	data = xr.Dataset()

	for index in indicies_to_load:
		X = indicies_data[index].data
		X = X.transpose()
		X = np.repeat(X[np.newaxis, np.newaxis,:], len(x), axis=0)
		X = np.repeat(X, len(y), axis = 1)
		dims = ['x','y','time']
		coords = [x,
				  y,
				  indicies_data.time]

		data[index] = xr.DataArray(data   = X, 
								   dims   = dims, 
								   coords = coords).transpose(*dims)
		data[index].attrs["units"] = "stddev"

	return data