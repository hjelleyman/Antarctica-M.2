from modules import combine_ice as ci
from modules import misc

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import itertools
from plyer import notification
import glob

from pyproj import Proj, transform

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def load_landmask():
	files = glob.glob('data/landice2/*.nc')
	ds = xr.open_mfdataset(files)
	ds = ds.land_mask
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
	newdata = xr.DataArray(dims=('y','x'),coords = [Y,X])
	variable_data = ds.interp(lon=x, lat=y, method = 'linear', kwargs={"fill_value": 0.0})	
	newdata.loc[:,:] = variable_data.values.reshape([len(Y),len(X)])
	return newdata.compute()

def load_ice_data():
	"""Loads in ice data, projected to the polar stereographic grid."""
	SIC = ci.load_seaice()
	LIC = ci.load_landice()
	LIC = LIC.sel(lat=slice(-90,-55))
	LIC = ci.latlon_to_polarstereo(LIC)
	LIC.name = 'lic'
	return SIC, LIC

def load_temp_data():
	SST = xr.open_dataset('processed_data/sst.nc').sst
	SKT = xr.open_dataset('processed_data/skt.nc').skt
	T2M = xr.open_dataset('processed_data/t2m.nc').t2m

	temperature = xr.Dataset()
	temperature['sst'] = SST
	temperature['t2m'] = T2M
	temperature['skt'] = SKT

	return temperature


def plot_2_timeseries(SIC, temperature, landmask):

	# Normalizing variables by area of grid cells This currently doesn't work
	area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
	SIC *= area
	temperature['sst'] = temperature['sst'] * area / area.sum()
	temperature['skt'] = temperature['skt'] * area / area.sum()
	temperature['t2m'] = temperature['t2m'] * area / area.sum()


	# Getting Trend of different variables

	predicted = xr.Dataset()
	for variable in temperature:
		data = temperature[variable]
		if variable == 'lic':
			gradient = data.sum(dim=('lat','lon')).polyfit(dim='time', deg=1)
		else:
			gradient = data.sum(dim=('x','y')).polyfit(dim='time', deg=1)

		gradient_m = gradient.sel(degree=1)
		gradient_b = gradient.sel(degree=0)
		predicted[variable] = \
		gradient_m['polyfit_coefficients'].values * temperature.time.values.astype(float) + gradient_b['polyfit_coefficients'].values

	gradient = SIC.sum(dim=('x','y')).polyfit(dim='time', deg=1)
	gradient_m = gradient.sel(degree=1)
	gradient_b = gradient.sel(degree=0)
	predicted['sic'] = \
		gradient_m['polyfit_coefficients'].values * SIC.time.values.astype(float) \
		+ gradient_b['polyfit_coefficients'].values


	# Creating figure and setting layout of axes
	plt.style.use('stylesheets/timeseries.mplstyle')
	fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,4), sharex='col')
	fig.tight_layout()

	# ax1.axhline(0,color='k', alpha=0.5)
	# ax2.axhline(0,color='k', alpha=0.5)

	# ax1 is a plot of SIE over Antarctica
	ax1.set_title('Total Sea Ice Extent over Antarctica')
	ax1.set_ylabel('SIE [$km^2$]')
	p_sic = ax1.plot(SIC.time, SIC.sum(dim=('x','y')), label = 'sie')
	ax1.plot(SIC.time, predicted.sic, color = p_sic[0].get_color())

	# ax2 is a plot of Temperature over the sea ice
	ax2.set_title('Mean Temperature over Sea Ice')
	ax2.set_ylabel('T [$^\circ$C]')
	ax2.plot([],[])
	p_t2m = ax2.plot(temperature.time, temperature.t2m.sum(dim=('x','y')), label = 't2m')
	p_sst = ax2.plot(temperature.time, temperature.sst.sum(dim=('x','y')), label = 'sst')
	p_skt = ax2.plot(temperature.time, temperature.skt.sum(dim=('x','y')), label = 'skt')
	ax2.plot(temperature.time, predicted.t2m, color = p_t2m[0].get_color())
	ax2.plot(temperature.time, predicted.sst, color = p_sst[0].get_color())
	ax2.plot(temperature.time, predicted.skt, color = p_skt[0].get_color())


	# legend
	lines = p_sic + p_sst + p_skt + p_t2m
	labels = [line.get_label() for line in lines]
	plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.15), ncol = 4, loc = 'upper right')

	misc.savefigures(folder='images/week8',filename='seaice_temperature_timeseries')



def plot_6_timeseries(SIC, LIC, temperature, landmask):

	# Normalizing variables by area of grid cells This currently doesn't work
	area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
	SIC *= area
	LIC *= area
	temperature['sst'] = temperature['sst'] * area / area.sum()
	temperature['skt'] = temperature['skt'] * area / area.sum()
	temperature['t2m'] = temperature['t2m'] * area / area.sum()
	
	SIC = SIC.sel(time=slice('2002-01-01','2019-12-31'))
	LIC = LIC.sel(time=slice('2002-01-01','2019-12-31'))
	temperature = temperature.sel(time=slice('2002-01-01','2019-12-31'))

	# fig = plt.figure()
	# ax = plt.axes(projection =ccrs.SouthPolarStereo())
	# ax.contourf(landmask.x,landmask.y,landmask)
	# plt.show()

	# Getting Trend of different variables
	predicted = xr.Dataset()
	for variable in temperature:
		data = temperature[variable]
		if variable == 'lic':
			gradient = data.sum(dim=('lat','lon')).polyfit(dim='time', deg=1)
		else:
			gradient = data.sum(dim=('x','y')).polyfit(dim='time', deg=1)

		gradient_m = gradient.sel(degree=1)
		gradient_b = gradient.sel(degree=0)
		predicted[variable] = \
		gradient_m['polyfit_coefficients'].values * temperature.time.values.astype(float) + gradient_b['polyfit_coefficients'].values

	gradient = SIC.sum(dim=('x','y')).polyfit(dim='time', deg=1)
	gradient_m = gradient.sel(degree=1)
	gradient_b = gradient.sel(degree=0)
	predicted['sic'] = \
		gradient_m['polyfit_coefficients'].values * SIC.time.values.astype(float) \
		+ gradient_b['polyfit_coefficients'].values

	gradient = LIC.sum(dim=('x','y')).polyfit(dim='time', deg=1)
	gradient_m = gradient.sel(degree=1)
	gradient_b = gradient.sel(degree=0)
	predicted['lic'] = \
		gradient_m['polyfit_coefficients'].values * LIC.time.values.astype(float) \
		+ gradient_b['polyfit_coefficients'].values

	# Creating figure and setting layout of axes
	plt.style.use('stylesheets/timeseries.mplstyle')
	fig, axes = plt.subplots(3,2, figsize=(5,4), sharex='col')
	fig.tight_layout()


	cols = ['Ice','Temperature']
	rows = ['Land', 'Sea', 'Antarctica']

	for ax, col in zip(axes[0], cols):
	    ax.set_title(col)

	for ax, row in zip(axes[:,0], rows):
	    ax.set_ylabel(row, size = 'large')

	for i,j in itertools.product(*[range(n) for n in axes.shape]):
		ax = axes[i,j]
		ax.axhline(0,color='k', alpha=0.5)

		 # Land variables
		if i==0:
			data = temperature.where(landmask)
			for variable in data:
				subdata = data[variable]
				gradient = subdata.sum(dim=('x','y')).polyfit(dim='time', deg=1)
				gradient_m = gradient.sel(degree=1)
				gradient_b = gradient.sel(degree=0)
				predicted[variable] = gradient_m['polyfit_coefficients'].values * temperature.time.values.astype(float) + gradient_b['polyfit_coefficients'].values
			if j == 0:
				ax.plot(LIC.time, LIC.sum(dim=('x','y')))
			else:
				ax.plot([],[])
				p_t2m = ax.plot(data.time, data.t2m.sum(dim=('x','y')), label = 't2m')
				p_sst = ax.plot([],[])
				p_skt = ax.plot(data.time, data.skt.sum(dim=('x','y')), label = 'skt')
				
				ax.plot(data.time, predicted.t2m, color = p_t2m[0].get_color())
				# ax.plot(data.time, predicted.sst, color = p_sst[0].get_color())
				ax.plot(data.time, predicted.skt, color = p_skt[0].get_color())

		 # Sea variables
		elif i==1:
			data = temperature.where(~landmask).where(SIC!=0)
			for variable in data:
				subdata = data[variable]
				gradient = subdata.sum(dim=('x','y')).polyfit(dim='time', deg=1)
				gradient_m = gradient.sel(degree=1)
				gradient_b = gradient.sel(degree=0)
				predicted[variable] = gradient_m['polyfit_coefficients'].values * temperature.time.values.astype(float) + gradient_b['polyfit_coefficients'].values
			if j == 0:
				ax.plot(SIC.time, SIC.sum(dim=('x','y')))
			else:
				ax.plot([],[])
				p_t2m = ax.plot(data.time, data.t2m.sum(dim=('x','y')), label = 't2m')
				p_sst = ax.plot(data.time, data.sst.sum(dim=('x','y')), label = 'sst')
				p_skt = ax.plot(data.time, data.skt.sum(dim=('x','y')), label = 'skt')

				ax.plot(data.time, predicted.t2m, color = p_t2m[0].get_color())
				ax.plot(data.time, predicted.sst, color = p_sst[0].get_color())
				ax.plot(data.time, predicted.skt, color = p_skt[0].get_color())

		elif i==2:
			data = temperature.copy()
			if j==0:
				pass
			else:
				ax.plot([],[])
				p_t2m = ax.plot(data.time, data.t2m.sum(dim=('x','y')), label = 't2m')
				p_sst = ax.plot(data.time, data.sst.sum(dim=('x','y')), label = 'sst')
				p_skt = ax.plot(data.time, data.skt.sum(dim=('x','y')), label = 'skt')
				ax.plot(data.time, predicted.t2m, color = p_t2m[0].get_color())
				ax.plot(data.time, predicted.sst, color = p_sst[0].get_color())
				ax.plot(data.time, predicted.skt, color = p_skt[0].get_color())
	lines = p_sst + p_skt + p_t2m
	labels = [line.get_label() for line in lines]
	plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.2), ncol = 4, loc = 'upper right')

	misc.savefigures(folder='images/week8',filename='six_timeseries')


def plot_2_trends(SIC, LIC, temperature, landmask):
	# Creating figure and setting layout of axes
	plt.style.use('stylesheets/timeseries.mplstyle')
	fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,10), subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
	fig.tight_layout()
	SIC = SIC.sel(time=slice('2002-01-01','2019-12-31'))
	temperature = temperature.sel(time=slice('2002-01-01','2019-12-31'))

	# Getting Trend of different variables
	
	# Creating figure and setting layout of axes


	# Sea variables
	trend = SIC.where(~landmask).where(SIC!=0).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
	divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
	plot = ax1.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
	cbar = plt.colorbar(plot,ax=ax1)
	cbar.ax.set_ylabel('[\% yr$^{-1}$]')
	ax1.set_title('Trends in SIE over Antarctica')
	ax1.coastlines()

	trend = temperature.t2m.where(~landmask).where(SIC!=0).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
	divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
	plot = ax2.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
	cbar = plt.colorbar(plot,ax=ax2)
	cbar.ax.set_ylabel('[$^\circ$C yr$^{-1}$]')
	ax2.set_title('Trends in 2MT over Antarctica')
	ax2.coastlines()
	misc.savefigures(folder='images/week8',filename='seaice_temperature_trends')


def plot_6_trends(SIC, LIC, temperature, landmask):

	SIC = SIC.sel(time=slice('2002-01-01','2019-12-31'))
	LIC = LIC.sel(time=slice('2002-01-01','2019-12-31'))
	temperature = temperature.sel(time=slice('2002-01-01','2019-12-31'))

	# fig = plt.figure()
	# ax = plt.axes(projection =ccrs.SouthPolarStereo())
	# ax.contourf(landmask.x,landmask.y,landmask)
	# plt.show()

	# Getting Trend of different variables
	
	# Creating figure and setting layout of axes
	plt.style.use('stylesheets/timeseries.mplstyle')
	fig, axes = plt.subplots(3,2, figsize=(5,5), subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
	fig.tight_layout()


	cols = ['Ice','Temperature']
	rows = ['Land', 'Sea', 'Antarctica']

	for ax, col in zip(axes[0], cols):
	    ax.set_title(col)

	for ax, row in zip(axes[:,0], rows):
	    ax.set_ylabel(row, size = 'large')

	for i,j in itertools.product(*[range(n) for n in axes.shape]):
		ax = axes[i,j]
		ax.axis('off')

		 # Land variables
		if i==0:
			data = temperature.where(landmask)
			if j == 0:
				trend = LIC.where(landmask).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
				divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
				plot = ax.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
				cbar = plt.colorbar(plot,ax=ax)
				cbar.ax.set_ylabel('[m yr$^{-1}$]')
			else:
				trend = temperature.t2m.where(landmask).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
				divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
				plot = ax.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
				cbar = plt.colorbar(plot,ax=ax)
				cbar.ax.set_ylabel('[$^\circ$C yr$^{-1}$]')
			ax.coastlines()

		 # Sea variables
		if i==1:
			if j == 0:
				trend = SIC.where(~landmask).where(SIC!=0).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
				divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
				plot = ax.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
				cbar = plt.colorbar(plot,ax=ax)
				cbar.ax.set_ylabel('[\% yr$^{-1}$]')
				
			else:
				trend = temperature.t2m.where(~landmask).where(SIC!=0).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
				divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
				plot = ax.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
				cbar = plt.colorbar(plot,ax=ax)
				cbar.ax.set_ylabel('[$^\circ$C yr$^{-1}$]')
			ax.coastlines()

	fig.tight_layout()
	misc.savefigures(folder='images/week8',filename='six_trends')
	# 	elif i==2:
	# 		data = temperature.copy()
	# 		if j==0:
	# 			pass
	# 		else:
	# 			ax.plot([],[])
	# 			p_t2m = ax.plot(data.time, data.t2m.mean(dim=('x','y')), label = 't2m')
	# 			p_sst = ax.plot(data.time, data.sst.mean(dim=('x','y')), label = 'sst')
	# 			p_skt = ax.plot(data.time, data.skt.mean(dim=('x','y')), label = 'skt')
	# lines = p_sst + p_skt + p_t2m
	# labels = [line.get_label() for line in lines]
	# plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.2), ncol = 4, loc = 'upper right')


def plot_2_scatter(SIC, LIC, temperature, landmask):

	# Normalizing variables by area of grid cells This currently doesn't work
	area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
	SIC *= area
	LIC *= area
	temperature['sst'] = temperature['sst'] * area / area.sum()
	temperature['skt'] = temperature['skt'] * area / area.sum()
	temperature['t2m'] = temperature['t2m'] * area / area.sum()
	
	SIC = SIC.sel(time=slice('2002-01-01','2019-12-31'))
	LIC = LIC.sel(time=slice('2002-01-01','2019-12-31'))
	temperature = temperature.sel(time=slice('2002-01-01','2019-12-31'))

	# Getting Trend of different variables
	predicted = xr.Dataset()

	# Creating figure and setting layout of axes
	plt.style.use('stylesheets/timeseries.mplstyle')
	fig, axes = plt.subplots(2,1, figsize=(5,10))
	fig.tight_layout()


	rows = ['Land', 'Sea']

	for ax, row in zip(axes, rows):
	    ax.set_ylabel(row, size = 'large')

	j = 0
	for i in range(2):
		ax = axes[i]
		ax.axhline(0,color='k', alpha=0.5)
		ax.axvline(0,color='k', alpha=0.5)

		 # Land variables
		if i==0:
			data = temperature.where(landmask)
			if j == 0:
				ax.scatter(data.t2m.values.flatten(), LIC.values.flatten())

		 # Sea variables
		elif i==1:
			data = temperature.where(~landmask).where(SIC!=0)
			if j == 0:
				ax.scatter(data.t2m.values.flatten(), SIC.values.flatten())

	# lines = p_sst + p_skt + p_t2m
	# labels = [line.get_label() for line in lines]
	# plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.2), ncol = 4, loc = 'upper right')

	# misc.savefigures(folder='images/week8',filename='six_timeseries')
