from modules import combine_ice as ci

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import itertools

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


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


def plot_2_timeseries(SIC, temperature):

	# Normalizing variables by area of grid cells This currently doesn't work
	# area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
	# normalized_area = area / area.sum() * area.size
	# print(area.size)
	# longterm_data['sic'] = longterm_data.sic * area
	# longterm_data['sst'] = longterm_data['sst'] * normalized_area
	# longterm_data['skt'] = longterm_data['skt'] * normalized_area
	# longterm_data['t2m'] = longterm_data['t2m'] * normalized_area

	# selecting only areas with seaice
	# longterm_data = longterm_data.where(longterm_data.sic != np.nan)


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
	predicted['sst'] = predicted['sst'] / (len(temperature.x)*len(temperature.y))
	predicted['skt'] = predicted['skt'] / (len(temperature.x)*len(temperature.y))
	predicted['t2m'] = predicted['t2m'] / (len(temperature.x)*len(temperature.y))

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

	ax1.axhline(0,color='k', alpha=0.5)
	ax2.axhline(0,color='k', alpha=0.5)

	# ax1 is a plot of SIE over Antarctica
	ax1.set_title('Total Sea Ice Extent over Antarctica')
	ax1.set_ylabel('SIE [$km^2$]')
	p_sic = ax1.plot(SIC.time, SIC.sum(dim=('x','y')), label = 'sie')
	ax1.plot(SIC.time, predicted.sic, color = p_sic[0].get_color())

	# ax2 is a plot of Temperature over the sea ice
	ax2.set_title('Mean Temperature over Sea Ice')
	ax2.set_ylabel('T [$^\circ$C]')
	ax2.plot([],[])
	p_t2m = ax2.plot(temperature.time, temperature.t2m.mean(dim=('x','y')), label = 't2m')
	p_sst = ax2.plot(temperature.time, temperature.sst.mean(dim=('x','y')), label = 'sst')
	p_skt = ax2.plot(temperature.time, temperature.skt.mean(dim=('x','y')), label = 'skt')
	ax2.plot(temperature.time, predicted.t2m, color = p_t2m[0].get_color())
	ax2.plot(temperature.time, predicted.sst, color = p_sst[0].get_color())
	ax2.plot(temperature.time, predicted.skt, color = p_skt[0].get_color())


	# legend
	lines = p_sic + p_sst + p_skt + p_t2m
	labels = [line.get_label() for line in lines]
	plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.15), ncol = 4, loc = 'upper right')


def plot_6_timeseries(SIC, LIC, temperature):

	# Normalizing variables by area of grid cells This currently doesn't work
	# area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
	# normalized_area = area * (area.size / area.sum().sum())
	# print(area.sum().sum(), area.size, area, normalized_area, normalized_area.min(), normalized_area.max())
	# SIC = SIC * normalized_area
	# LIC = LIC * normalized_area
	# temperature['sst'] = temperature['sst'] * normalized_area
	# temperature['skt'] = temperature['skt'] * normalized_area
	# temperature['t2m'] = temperature['t2m'] * normalized_area

	# selecting only areas with seaice
	# longterm_data = longterm_data.where(longterm_data.sic != np.nan)
	SIC = SIC.sel(time=slice('2002-01-01','2019-12-31'))
	LIC = LIC.sel(time=slice('2002-01-01','2019-12-31'))
	temperature = temperature.sel(time=slice('2002-01-01','2019-12-31'))

	landmask = LIC.sum(dim='time') !=0
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
	predicted['sst'] = predicted['sst'] / (len(temperature.x)*len(temperature.y))
	predicted['skt'] = predicted['skt'] / (len(temperature.x)*len(temperature.y))
	predicted['t2m'] = predicted['t2m'] / (len(temperature.x)*len(temperature.y))

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
			if j == 0:
				ax.plot(LIC.time, LIC.sum(dim=('x','y')))
			else:
				ax.plot([],[])
				p_t2m = ax.plot(data.time, data.t2m.mean(dim=('x','y')), label = 't2m')
				p_sst = ax.plot([],[])
				p_skt = ax.plot(data.time, data.skt.mean(dim=('x','y')), label = 'skt')

		 # Sea variables
		elif i==1:
			data = temperature.where(~landmask).where(SIC!=0)
			if j == 0:
				ax.plot(SIC.time, SIC.sum(dim=('x','y')))
			else:
				ax.plot([],[])
				p_t2m = ax.plot(data.time, data.t2m.mean(dim=('x','y')), label = 't2m')
				p_sst = ax.plot(data.time, data.sst.mean(dim=('x','y')), label = 'sst')
				p_skt = ax.plot(data.time, data.skt.mean(dim=('x','y')), label = 'skt')

		elif i==2:
			data = temperature.copy()
			if j==0:
				pass
			else:
				ax.plot([],[])
				p_t2m = ax.plot(data.time, data.t2m.mean(dim=('x','y')), label = 't2m')
				p_sst = ax.plot(data.time, data.sst.mean(dim=('x','y')), label = 'sst')
				p_skt = ax.plot(data.time, data.skt.mean(dim=('x','y')), label = 'skt')
	lines = p_sst + p_skt + p_t2m
	labels = [line.get_label() for line in lines]
	plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.2), ncol = 4, loc = 'upper right')


def plot_2_trends(SIC, temperature):

	# Normalizing variables by area of grid cells This currently doesn't work
	# area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
	# normalized_area = area / area.sum() * area.size
	# print(area.size)
	# longterm_data['sic'] = longterm_data.sic * area
	# longterm_data['sst'] = longterm_data['sst'] * normalized_area
	# longterm_data['skt'] = longterm_data['skt'] * normalized_area
	# longterm_data['t2m'] = longterm_data['t2m'] * normalized_area

	# selecting only areas with seaice
	# longterm_data = longterm_data.where(longterm_data.sic != np.nan)


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
	predicted['sst'] = predicted['sst'] / (len(temperature.x)*len(temperature.y))
	predicted['skt'] = predicted['skt'] / (len(temperature.x)*len(temperature.y))
	predicted['t2m'] = predicted['t2m'] / (len(temperature.x)*len(temperature.y))

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

	ax1.axhline(0,color='k', alpha=0.5)
	ax2.axhline(0,color='k', alpha=0.5)

	# ax1 is a plot of SIE over Antarctica
	ax1.set_title('Total Sea Ice Extent over Antarctica')
	ax1.set_ylabel('SIE [$km^2$]')
	p_sic = ax1.plot(SIC.time, SIC.sum(dim=('x','y')), label = 'sie')
	ax1.plot(SIC.time, predicted.sic, color = p_sic[0].get_color())

	# ax2 is a plot of Temperature over the sea ice
	ax2.set_title('Mean Temperature over Sea Ice')
	ax2.set_ylabel('T [$^\circ$C]')
	ax2.plot([],[])
	p_t2m = ax2.plot(temperature.time, temperature.t2m.mean(dim=('x','y')), label = 't2m')
	p_sst = ax2.plot(temperature.time, temperature.sst.mean(dim=('x','y')), label = 'sst')
	p_skt = ax2.plot(temperature.time, temperature.skt.mean(dim=('x','y')), label = 'skt')
	ax2.plot(temperature.time, predicted.t2m, color = p_t2m[0].get_color())
	ax2.plot(temperature.time, predicted.sst, color = p_sst[0].get_color())
	ax2.plot(temperature.time, predicted.skt, color = p_skt[0].get_color())


	# legend
	lines = p_sic + p_sst + p_skt + p_t2m
	labels = [line.get_label() for line in lines]
	plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.15), ncol = 4, loc = 'upper right')


def plot_6_trends(SIC, LIC, temperature):

	# Normalizing variables by area of grid cells This currently doesn't work
	# area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
	# normalized_area = area * (area.size / area.sum().sum())
	# print(area.sum().sum(), area.size, area, normalized_area, normalized_area.min(), normalized_area.max())
	# SIC = SIC * normalized_area
	# LIC = LIC * normalized_area
	# temperature['sst'] = temperature['sst'] * normalized_area
	# temperature['skt'] = temperature['skt'] * normalized_area
	# temperature['t2m'] = temperature['t2m'] * normalized_area

	# selecting only areas with seaice
	# longterm_data = longterm_data.where(longterm_data.sic != np.nan)
	SIC = SIC.sel(time=slice('2002-01-01','2019-12-31'))
	LIC = LIC.sel(time=slice('2002-01-01','2019-12-31'))
	temperature = temperature.sel(time=slice('2002-01-01','2019-12-31'))

	landmask = LIC.sum(dim='time') !=0
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
			else:
				trend = temperature.t2m.where(landmask).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
				divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
				plot = ax.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
			ax.coastlines()

		 # Sea variables
		if i==1:
			if j == 0:
				trend = SIC.where(~landmask).where(SIC!=0).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
				divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
				plot = ax.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
				
			else:
				trend = temperature.t2m.where(~landmask).where(SIC!=0).polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
				divnorm = TwoSlopeNorm(vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
				plot = ax.contourf(trend.x, trend.y,trend, transform=ccrs.SouthPolarStereo(),cmap = 'RdBu', norm=divnorm)
			ax.coastlines()

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