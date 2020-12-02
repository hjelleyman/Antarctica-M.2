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

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm


def load_landmask():
    files = glob.glob('data/landice2/*.nc')
    ds = xr.open_mfdataset(files)
    ds = ds.land_mask
    Y, X = [10*np.arange(435000, -395000, -2500),
            10*np.arange(-395000, 395000, 2500)]
    x, y = np.meshgrid(X, Y)
    inProj = Proj(init='epsg:3031')
    outProj = Proj(init='epsg:4326')
    x, y = transform(inProj, outProj, x, y)
    x = x.flatten()
    y = y.flatten()
    x[x < 0] = x[x < 0]+360
    x = xr.DataArray(x, dims='z')
    y = xr.DataArray(y, dims='z')
    newdata = xr.DataArray(dims=('y', 'x'), coords=[Y, X])
    variable_data = ds.interp(
        lon=x, lat=y, method='linear', kwargs={"fill_value": 0.0})
    newdata.loc[:, :] = variable_data.values.reshape([len(Y), len(X)])
    return newdata.compute()


def load_ice_data():
    """Loads in ice data, projected to the polar stereographic grid."""
    SIC = ci.load_seaice() * 100/250
    LIC = ci.load_landice()
    LIC = LIC.sel(lat=slice(-90, -55))
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
            gradient = data.sum(dim=('lat', 'lon')).polyfit(dim='time', deg=1)
        else:
            gradient = data.sum(dim=('x', 'y')).polyfit(dim='time', deg=1)

        gradient_m = gradient.sel(degree=1)
        gradient_b = gradient.sel(degree=0)
        predicted[variable] = \
            gradient_m['polyfit_coefficients'].values * temperature.time.values.astype(
                float) + gradient_b['polyfit_coefficients'].values

    gradient = SIC.sum(dim=('x', 'y')).polyfit(dim='time', deg=1)
    gradient_m = gradient.sel(degree=1)
    gradient_b = gradient.sel(degree=0)
    predicted['sic'] = \
        gradient_m['polyfit_coefficients'].values * SIC.time.values.astype(float) \
        + gradient_b['polyfit_coefficients'].values

    # Creating figure and setting layout of axes
    plt.style.use('stylesheets/timeseries.mplstyle')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex='col')
    fig.tight_layout()

    # ax1.axhline(0,color='k', alpha=0.5)
    # ax2.axhline(0,color='k', alpha=0.5)

    # ax1 is a plot of SIE over Antarctica
    ax1.set_title('Total Sea Ice Extent over Antarctica')
    ax1.set_ylabel('SIE [$km^2$]')
    p_sic = ax1.plot(SIC.time, SIC.sum(dim=('x', 'y')), label='sie')
    ax1.plot(SIC.time, predicted.sic, color=p_sic[0].get_color())

    # ax2 is a plot of Temperature over the sea ice
    ax2.set_title('Mean Temperature over Sea Ice')
    ax2.set_ylabel('T [$^\circ$C]')
    ax2.plot([], [])
    p_t2m = ax2.plot(temperature.time, temperature.t2m.sum(
        dim=('x', 'y')), label='t2m')
    p_sst = ax2.plot(temperature.time, temperature.sst.sum(
        dim=('x', 'y')), label='sst')
    p_skt = ax2.plot(temperature.time, temperature.skt.sum(
        dim=('x', 'y')), label='skt')
    ax2.plot(temperature.time, predicted.t2m, color=p_t2m[0].get_color())
    ax2.plot(temperature.time, predicted.sst, color=p_sst[0].get_color())
    ax2.plot(temperature.time, predicted.skt, color=p_skt[0].get_color())

    # legend
    lines = p_sic + p_sst + p_skt + p_t2m
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, bbox_to_anchor=(
        0.99, -0.15), ncol=4, loc='upper right')

    misc.savefigures(folder='images/week8',
                     filename='seaice_temperature_timeseries')


def plot_6_timeseries(SIC, LIC, temperature, landmask):

    # Normalizing variables by area of grid cells This currently doesn't work
    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    SIC *= area
    LIC *= area
    temperature['sst'] = temperature['sst'] * area / area.sum()
    temperature['skt'] = temperature['skt'] * area / area.sum()
    temperature['t2m'] = temperature['t2m'] * area / area.sum()

    SIC = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    LIC = LIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature = temperature.sel(time=slice('2002-01-01', '2019-12-31'))

    # fig = plt.figure()
    # ax = plt.axes(projection =ccrs.SouthPolarStereo())
    # ax.contourf(landmask.x,landmask.y,landmask)
    # plt.show()

    # Getting Trend of different variables
    predicted = xr.Dataset()
    for variable in temperature:
        data = temperature[variable]
        if variable == 'lic':
            gradient = data.sum(dim=('lat', 'lon')).polyfit(dim='time', deg=1)
        else:
            gradient = data.sum(dim=('x', 'y')).polyfit(dim='time', deg=1)

        gradient_m = gradient.sel(degree=1)
        gradient_b = gradient.sel(degree=0)
        predicted[variable] = \
            gradient_m['polyfit_coefficients'].values * temperature.time.values.astype(
                float) + gradient_b['polyfit_coefficients'].values

    gradient = SIC.sum(dim=('x', 'y')).polyfit(dim='time', deg=1)
    gradient_m = gradient.sel(degree=1)
    gradient_b = gradient.sel(degree=0)
    predicted['sic'] = \
        gradient_m['polyfit_coefficients'].values * SIC.time.values.astype(float) \
        + gradient_b['polyfit_coefficients'].values

    gradient = LIC.sum(dim=('x', 'y')).polyfit(dim='time', deg=1)
    gradient_m = gradient.sel(degree=1)
    gradient_b = gradient.sel(degree=0)
    predicted['lic'] = \
        gradient_m['polyfit_coefficients'].values * LIC.time.values.astype(float) \
        + gradient_b['polyfit_coefficients'].values

    # Creating figure and setting layout of axes
    plt.style.use('stylesheets/timeseries.mplstyle')
    fig, axes = plt.subplots(2, 2, figsize=(5, 3), sharex='col')
    fig.tight_layout()

    cols = ['Ice', 'Temperature']
    rows = ['Land', 'Sea', 'Antarctica']

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, size='large')

    for i, j in itertools.product(*[range(n) for n in axes.shape]):
        ax = axes[i, j]
        ax.axhline(0, color='k', alpha=0.5)
        ax.xaxis.set_major_formatter('%Y')

        # Land variables
        if i == 0:
            data = temperature.where(landmask)
            for variable in data:
                subdata = data[variable]
                gradient = subdata.sum(
                    dim=('x', 'y')).polyfit(dim='time', deg=1)
                gradient_m = gradient.sel(degree=1)
                gradient_b = gradient.sel(degree=0)
                predicted[variable] = gradient_m['polyfit_coefficients'].values * \
                    temperature.time.values.astype(
                        float) + gradient_b['polyfit_coefficients'].values
            if j == 0:
                gradient = LIC.sum(
                    dim=('x', 'y')).polyfit(dim='time', deg=1)
                gradient_m = gradient.sel(degree=1)
                gradient_b = gradient.sel(degree=0)
                LIC_predicted = gradient_m['polyfit_coefficients'].values * \
                    temperature.time.values.astype(
                        float) + gradient_b['polyfit_coefficients'].values

                p = ax.plot(LIC.time, LIC.sum(dim=('x', 'y')))
                ax.plot(LIC.time, LIC_predicted, color=p[0].get_color())
            else:
                ax.plot([], [])
                p_t2m = ax.plot(data.time, data.t2m.sum(
                    dim=('x', 'y')), label='t2m')
                p_sst = ax.plot([], [])
                p_skt = ax.plot(data.time, data.skt.sum(
                    dim=('x', 'y')), label='skt')

                ax.plot(data.time, predicted.t2m, color=p_t2m[0].get_color())
                # ax.plot(data.time, predicted.sst, color = p_sst[0].get_color())
                ax.plot(data.time, predicted.skt, color=p_skt[0].get_color())

         # Sea variables
        elif i == 1:
            data = temperature.where(~landmask).where(SIC != 0)
            for variable in data:
                subdata = data[variable]
                gradient = subdata.sum(
                    dim=('x', 'y')).polyfit(dim='time', deg=1)
                gradient_m = gradient.sel(degree=1)
                gradient_b = gradient.sel(degree=0)
                predicted[variable] = gradient_m['polyfit_coefficients'].values * \
                    temperature.time.values.astype(
                        float) + gradient_b['polyfit_coefficients'].values
            if j == 0:
                gradient = SIC.sortby('time').sum(
                    dim=('x', 'y')).polyfit(dim='time', deg=1)
                gradient_m = gradient.sel(degree=1)
                gradient_b = gradient.sel(degree=0)
                SIC_predicted = gradient_m['polyfit_coefficients'].values * \
                    temperature.time.values.astype(
                        float) + gradient_b['polyfit_coefficients'].values

                p = ax.plot(SIC.time, SIC.sum(dim=('x', 'y')))
                ax.plot(SIC.time, SIC_predicted, color=p[0].get_color())

            else:
                ax.plot([], [])
                p_t2m = ax.plot(data.time, data.t2m.sum(
                    dim=('x', 'y')), label='t2m')
                p_sst = ax.plot(data.time, data.sst.sum(
                    dim=('x', 'y')), label='sst')
                p_skt = ax.plot(data.time, data.skt.sum(
                    dim=('x', 'y')), label='skt')

                ax.plot(data.time, predicted.t2m, color=p_t2m[0].get_color())
                ax.plot(data.time, predicted.sst, color=p_sst[0].get_color())
                ax.plot(data.time, predicted.skt, color=p_skt[0].get_color())

        elif i == 2:
            data = temperature.copy()
            if j == 0:
                pass
            else:
                ax.plot([], [])
                p_t2m = ax.plot(data.time, data.t2m.sum(
                    dim=('x', 'y')), label='t2m')
                p_sst = ax.plot(data.time, data.sst.sum(
                    dim=('x', 'y')), label='sst')
                p_skt = ax.plot(data.time, data.skt.sum(
                    dim=('x', 'y')), label='skt')
                ax.plot(data.time, predicted.t2m, color=p_t2m[0].get_color())
                ax.plot(data.time, predicted.sst, color=p_sst[0].get_color())
                ax.plot(data.time, predicted.skt, color=p_skt[0].get_color())
    lines = p_sst + p_skt + p_t2m
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, bbox_to_anchor=(
        0.99, -0.2), ncol=4, loc='upper right')

    misc.savefigures(folder='images/week8', filename='six_timeseries')


def plot_2_trends(SIC, LIC, temperature, landmask):
    # Creating figure and setting layout of axes
    plt.style.use('stylesheets/timeseries.mplstyle')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10),
                                   subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
    fig.tight_layout()
    SIC = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature = temperature.sel(time=slice('2002-01-01', '2019-12-31'))

    # Getting Trend of different variables

    # Creating figure and setting layout of axes

    # Sea variables
    trend = SIC.where(~landmask).where(SIC != 0).polyfit(
        dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
    divnorm = TwoSlopeNorm(vmin=trend.min().min(),
                           vcenter=0, vmax=trend.max().max())
    plot = ax1.contourf(trend.x, trend.y, trend,
                        transform=ccrs.SouthPolarStereo(), cmap='RdBu', norm=divnorm)
    cbar = plt.colorbar(plot, ax=ax1)
    cbar.ax.set_ylabel('[\% yr$^{-1}$]')
    ax1.set_title('Trends in SIE over Antarctica')
    ax1.coastlines()

    trend = temperature.t2m.where(~landmask).where(SIC != 0).polyfit(
        dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
    divnorm = TwoSlopeNorm(vmin=trend.min().min(),
                           vcenter=0, vmax=trend.max().max())
    plot = ax2.contourf(trend.x, trend.y, trend,
                        transform=ccrs.SouthPolarStereo(), cmap='RdBu', norm=divnorm)
    cbar = plt.colorbar(plot, ax=ax2)
    cbar.ax.set_ylabel('[$^\circ$C yr$^{-1}$]')
    ax2.set_title('Trends in 2MT over Antarctica')
    ax2.coastlines()
    misc.savefigures(folder='images/week8',
                     filename='seaice_temperature_trends')


def plot_6_trends(SIC, LIC, temperature, landmask):

    SIC = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    LIC = LIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature = temperature.sel(time=slice('2002-01-01', '2019-12-31'))

    # fig = plt.figure()
    # ax = plt.axes(projection =ccrs.SouthPolarStereo())
    # ax.contourf(landmask.x,landmask.y,landmask)
    # plt.show()

    # Getting Trend of different variables

    # Creating figure and setting layout of axes
    plt.style.use('stylesheets/timeseries.mplstyle')
    fig, axes = plt.subplots(3, 2, figsize=(
        5, 5), subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
    fig.tight_layout()

    cols = ['Ice', 'Temperature']
    rows = ['Land', 'Sea', 'Antarctica']

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, size='large')

    for i, j in itertools.product(*[range(n) for n in axes.shape]):
        ax = axes[i, j]
        ax.axis('off')

        # Land variables
        if i == 0:
            data = temperature.where(landmask)
            if j == 0:
                trend = LIC.where(landmask).polyfit(dim='time', deg=1).sel(
                    degree=1).polyfit_coefficients * 1e9*60*60*24*365
                divnorm = TwoSlopeNorm(
                    vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
                plot = ax.contourf(trend.x, trend.y, trend, transform=ccrs.SouthPolarStereo(
                ), cmap='RdBu', norm=divnorm)
                cbar = plt.colorbar(plot, ax=ax)
                cbar.ax.set_ylabel('[m yr$^{-1}$]')
            else:
                trend = temperature.t2m.where(landmask).polyfit(dim='time', deg=1).sel(
                    degree=1).polyfit_coefficients * 1e9*60*60*24*365
                divnorm = TwoSlopeNorm(
                    vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
                plot = ax.contourf(trend.x, trend.y, trend, transform=ccrs.SouthPolarStereo(
                ), cmap='RdBu', norm=divnorm)
                cbar = plt.colorbar(plot, ax=ax)
                cbar.ax.set_ylabel('[$^\circ$C yr$^{-1}$]')
            ax.coastlines()

         # Sea variables
        if i == 1:
            if j == 0:
                trend = SIC.where(~landmask).where(SIC != 0).polyfit(
                    dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
                divnorm = TwoSlopeNorm(
                    vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
                plot = ax.contourf(trend.x, trend.y, trend, transform=ccrs.SouthPolarStereo(
                ), cmap='RdBu', norm=divnorm)
                cbar = plt.colorbar(plot, ax=ax)
                cbar.ax.set_ylabel('[\% yr$^{-1}$]')

            else:
                trend = temperature.t2m.where(~landmask).where(SIC != 0).polyfit(
                    dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
                divnorm = TwoSlopeNorm(
                    vmin=trend.min().min(), vcenter=0, vmax=trend.max().max())
                plot = ax.contourf(trend.x, trend.y, trend, transform=ccrs.SouthPolarStereo(
                ), cmap='RdBu', norm=divnorm)
                cbar = plt.colorbar(plot, ax=ax)
                cbar.ax.set_ylabel('[$^\circ$C yr$^{-1}$]')
            ax.coastlines()

    fig.tight_layout()
    misc.savefigures(folder='images/week8', filename='six_trends')
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


def plot_2_scatter(SIC, LIC, temperature, landmask, filename='distribution_of_temperature_ice'):

    # Normalizing variables by area of grid cells This currently doesn't work
    # area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    # SIC *= area
    # LIC *= area
    # temperature['sst'] = temperature['sst'] * area / area.sum()
    # temperature['skt'] = temperature['skt'] * area / area.sum()
    # temperature['t2m'] = temperature['t2m'] * area / area.sum()

    SIC = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    LIC = LIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature = temperature.sel(time=slice('2002-01-01', '2019-12-31'))

    # Creating figure and setting layout of axes
    plt.style.use('stylesheets/timeseries.mplstyle')
    fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex='col')

    plt.suptitle('Joint Distribution of Temperature and Ice Extents')

    j = 0
    for i in range(2):
        ax = axes[i]
        # ax.axhline(0,color='k', alpha=0.3)
        # ax.axvline(0,color='k', alpha=0.3)

        # Land variables
        if i == 0:
            data = temperature.where(landmask)
            if j == 0:
                x = data.skt.values.flatten()
                y = LIC.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts, norm=LogNorm())
                ax.set_xlabel(r'Skin Temperature [K]')
                ax.set_ylabel(
                    r'Land Ice Equivilent Liquid Water Thickness [m]')

         # Sea variables
        elif i == 1:
            data = temperature.where(~landmask).where(SIC != 0)
            if j == 0:
                x = data.skt.values.flatten()
                y = SIC.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts, norm=LogNorm())
                ax.set_xlabel(r'Skin Temperature [K]')
                ax.set_ylabel(r'Sea Ice Concentration [\% area]')
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    misc.savefigures(folder='images/week8', filename=filename)

    # lines = p_sst + p_skt + p_t2m
    # labels = [line.get_label() for line in lines]
    # plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.2), ncol = 4, loc = 'upper right')

    # misc.savefigures(folder='images/week8',filename='six_timeseries')


from modules import week5 as w5

def plot_4_scatter(SIC, LIC, temperature, landmask, filename='distribution_of_temperature_ice_both_raw_and_anomalous'):

    SIC = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    LIC = LIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature = temperature.sel(time=slice('2002-01-01', '2019-12-31'))

    SIC_anomalous = SIC.pipe(w5.find_anomalies)
    LIC_anomalous = LIC.pipe(w5.find_anomalies)
    temperature_anomalous = temperature.pipe(w5.find_anomalies)
    
    # Creating figure and setting layout of axes
    plt.style.use('stylesheets/timeseries.mplstyle')
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharex='col', sharey='row')

    plt.suptitle('Joint Distribution of Temperature and Ice Extents')

    for i,j in itertools.product(range(2),range(2)):
        ax = axes[i,j]
        # ax.axhline(0,color='k', alpha=0.3)
        # ax.axvline(0,color='k', alpha=0.3)

        # Land variables
        if i == 0:
            if j == 0:
                data = temperature.where(landmask)
                x = data.skt.values.flatten()
                y = LIC.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts, norm=LogNorm(), vmin = 0.1, vmax = 1e4)
                # ax.set_xlabel(r'Skin Temperature [K]')
                ax.set_ylabel(r'LIELWT [m]')
                ax.set_title('Raw data plots')

            if j == 1:
                data = temperature_anomalous.where(landmask)
                x = data.skt.values.flatten()
                y = LIC_anomalous.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts,
                                 norm=LogNorm(), vmin=1, vmax=1e4)
                # ax.set_xlabel(r'Skin Temperature [K]')
                # ax.set_ylabel(
                    # r'Land Ice Equivilent Liquid Water Thickness [m]')
                ax.set_title('Anomalous data plots')

        # Sea variables
        elif i == 1:
            if j == 0:
                data = temperature.where(~landmask).where(SIC != 0)
                x = data.skt.values.flatten()
                y = SIC.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts,
                                 norm=LogNorm(), vmin=1, vmax=1e4)
                ax.set_xlabel(r'Skin Temperature [K]')
                ax.set_ylabel(r'SIC [\% area]')

            if j == 1:
                data = temperature_anomalous.where(~landmask).where(SIC != 0)
                x = data.skt.values.flatten()
                y = SIC_anomalous.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts,
                                 norm=LogNorm(), vmin=1, vmax=1e4)
                ax.set_xlabel(r'Skin Temperature [K]')
                # ax.set_ylabel(r'Sea Ice Concentration [\% area]')
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.99, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.88)
    # cbar.set_label('Regression Coefficients [$\\frac{\%}{\sigma}$]')
    fig.tight_layout()
    misc.savefigures(folder='images/week8', filename=filename)

    # lines = p_sst + p_skt + p_t2m
    # labels = [line.get_label() for line in lines]
    # plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.2), ncol = 4, loc = 'upper right')

    # misc.savefigures(folder='images/week8',filename='six_timeseries')


def scatter_just_seaice(SIC, temperature, landmask): 

    SIC = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    LIC = LIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature = temperature.sel(time=slice('2002-01-01', '2019-12-31'))

    SIC_anomalous = SIC.pipe(w5.find_anomalies)
    LIC_anomalous = LIC.pipe(w5.find_anomalies)
    temperature_anomalous = temperature.pipe(w5.find_anomalies)

    # Creating figure and setting layout of axes
    plt.style.use('stylesheets/timeseries.mplstyle')
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharex='col', sharey='row')

    plt.suptitle('Joint Distribution of Temperature and Ice Extents')

    for i, j in itertools.product(range(2), range(2)):
        ax = axes[i, j]
        # ax.axhline(0,color='k', alpha=0.3)
        # ax.axvline(0,color='k', alpha=0.3)

        # Land variables
        if i == 0:
            if j == 0:
                data = temperature.where(landmask)
                x = data.skt.values.flatten()
                y = LIC.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts,
                                 norm=LogNorm(), vmin=0.1, vmax=1e4)
                # ax.set_xlabel(r'Skin Temperature [K]')
                ax.set_ylabel(r'LIELWT [m]')
                ax.set_title('Raw data plots')

            if j == 1:
                data = temperature_anomalous.where(landmask)
                x = data.skt.values.flatten()
                y = LIC_anomalous.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts,
                                 norm=LogNorm(), vmin=1, vmax=1e4)
                # ax.set_xlabel(r'Skin Temperature [K]')
                # ax.set_ylabel(
                # r'Land Ice Equivilent Liquid Water Thickness [m]')
                ax.set_title('Anomalous data plots')

        # Sea variables
        elif i == 1:
            if j == 0:
                data = temperature.where(~landmask).where(SIC != 0)
                x = data.skt.values.flatten()
                y = SIC.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts,
                                 norm=LogNorm(), vmin=1, vmax=1e4)
                ax.set_xlabel(r'Skin Temperature [K]')
                ax.set_ylabel(r'SIC [\% area]')

            if j == 1:
                data = temperature_anomalous.where(~landmask).where(SIC != 0)
                x = data.skt.values.flatten()
                y = SIC_anomalous.values.flatten()
                mask = np.isfinite(x) * np.isfinite(y)
                X = x[mask]
                Y = y[mask]
                counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
                xedges = (xedges[1:] + xedges[:-1]) / 2
                yedges = (yedges[1:] + yedges[:-1]) / 2
                im = ax.contourf(xedges, yedges, counts,
                                 norm=LogNorm(), vmin=1, vmax=1e4)
                ax.set_xlabel(r'Skin Temperature [K]')
                # ax.set_ylabel(r'Sea Ice Concentration [\% area]')
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.99, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.88)
    # cbar.set_label('Regression Coefficients [$\\frac{\%}{\sigma}$]')
    fig.tight_layout()
    misc.savefigures(folder='images/week8', filename=filename)


def mean_distribution(SIC, LIC, temperature, landmask):
    """[summary]

    Args:
        variable (xarray dataarray): variable for which we want the mean spatial distribution.
    """

    SIC_mean = SIC.mean(dim='time').copy()
    SIC_mean = SIC_mean.where(SIC_mean <= 100)

    LIC_mean = LIC.mean(dim='time').copy()
    LIC_mean = LIC_mean.where(landmask)

    temperature_mean = temperature.mean(dim='time').copy().skt
    temperature_mean = temperature_mean.where(temperature_mean > 100) - 273.15



    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(1,1,1, projection=ccrs.SouthPolarStereo())
    plot = ax.contourf(SIC_mean.x, SIC_mean.y, SIC_mean, crs=ccrs.SouthPolarStereo(), vmin = 0, vmax = 100, levels = 10)
    ax.coastlines()
    ax.set_title('Mean Sea Ice Concentration')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Mean Sea Ice Concentration [%]')
    misc.savefigures(folder='images/week8', filename='mean_sic_distribution')
    plt.show()

    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(1,1,1, projection=ccrs.SouthPolarStereo())
    plot = ax.contourf(LIC_mean.x, LIC_mean.y, LIC_mean, levels = 20, crs=ccrs.SouthPolarStereo())
    ax.coastlines()
    ax.set_title('Mean Land Ice LWET')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Mean Land Ice LWET [m]')
    misc.savefigures(folder='images/week8', filename='mean_lic_distribution')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin = temperature_mean.min().min(), vcenter = 0, vmax=temperature_mean.max().max())
    plot = ax.contourf(temperature_mean.x, temperature_mean.y, temperature_mean,
                       crs=ccrs.SouthPolarStereo(), cmap = 'RdBu_r', norm = divnorm, levels = 16)
    ax.coastlines()
    ax.set_title('Mean Skin Temperature')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Mean Skin Temperature [$^\circ$C]')
    misc.savefigures(folder='images/week8', filename='mean_skt_distribution')
    plt.show()


def trend_distribution(SIC, LIC, temperature, landmask):
    """[summary]

    Args:
        variable (xarray dataarray): variable for which we want the mean spatial distribution.
    """

    SIC_mean = SIC.polyfit(dim='time', deg=1).copy().sel(
        degree=1).polyfit_coefficients * 1e9*60*60*24*365
    SIC_mean = SIC_mean.where(SIC.mean(dim='time') <= 100).where(~landmask)

    LIC_mean = LIC.polyfit(dim='time', deg=1).copy().sel(
        degree=1).polyfit_coefficients * 1e9*60*60*24*365
    LIC_mean = LIC_mean.where(landmask)

    temperature_mean = temperature.skt.polyfit(dim='time', deg=1).copy().sel(
        degree=1).polyfit_coefficients * 1e9*60*60*24*365
    # temperature_mean = temperature_mean.where(temperature_mean > 100)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=SIC_mean.min(
    ).min(), vcenter=0, vmax=SIC_mean.max().max())
    plot = ax.contourf(SIC_mean.x, SIC_mean.y, SIC_mean,
                       crs=ccrs.SouthPolarStereo(), cmap='PuOr', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Trend in Sea Ice Concentration')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Trend in Sea Ice Concentration [\% yr$^{-1}$]')
    misc.savefigures(folder='images/week8', filename='trend_sic_distribution')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=LIC_mean.min(
    ).min(), vcenter=0, vmax=LIC_mean.max().max())
    plot = ax.contourf(LIC_mean.x, LIC_mean.y, LIC_mean,
                       crs=ccrs.SouthPolarStereo(), cmap='PuOr', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Trend in Land Ice LWET')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Trend in Land Ice LWET [m yr$^{-1}$]')
    misc.savefigures(folder='images/week8', filename='trend_lic_distribution')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=temperature_mean.min(
    ).min(), vcenter=0, vmax=temperature_mean.max().max())
    plot = ax.contourf(temperature_mean.x, temperature_mean.y, temperature_mean,
                       crs=ccrs.SouthPolarStereo(), cmap='RdBu_r', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Trend in Skin Temperature')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Trend in Skin Temperature [$^\circ$C yr$^{-1}$]')
    misc.savefigures(folder='images/week8', filename='trend_skt_distribution')
    plt.show()
    

    
def correlation_plots(SIC, LIC, temperature, landmask):
    """Plots all the correlation plots. and outputs key statistics.

    Args:
        SIC ([type]): [description]
        LIC ([type]): [description]
        temperature ([type]): [description]
        landmask ([type]): [description]
    """

    # Generate timeseries with different lengths
    SIC_short = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    LIC_short = LIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature_short = temperature.sel(time=slice('2002-01-01', '2019-12-31')).skt

    SIC_long = SIC.sel(time=slice('1979-01-01', '2019-12-31'))
    temperature_long = temperature.sel(time=slice('1979-01-01', '2019-12-31')).skt



    # Plot spatial correlation of variables
    corr_SIC_temp_long  = xr.corr(SIC_long,  temperature_long, dim='time')
    corr_SIC_temp_short = xr.corr(SIC_short, temperature_short, dim='time')
    corr_LIC_temp_short = xr.corr(LIC_short, temperature_short, dim='time')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.contourf(corr_SIC_temp_short.x, corr_SIC_temp_short.y, corr_SIC_temp_short,
                       crs=ccrs.SouthPolarStereo(), cmap='PuOr', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Correlations between SIC and SKT')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Correlations')
    misc.savefigures(folder='images/week8',
                     filename='corr_sic_skt_shortterm_spatial')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.contourf(corr_SIC_temp_long.x, corr_SIC_temp_long.y, corr_SIC_temp_long,
                       crs=ccrs.SouthPolarStereo(), cmap='PuOr', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Correlations between SIC and SKT')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Correlations')
    misc.savefigures(folder='images/week8',
                     filename='corr_sic_skt_longterm_spatial')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.contourf(corr_LIC_temp_short.x, corr_LIC_temp_short.y, corr_LIC_temp_short,
                       crs=ccrs.SouthPolarStereo(), cmap='PuOr', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Correlations between LIC and SKT')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Correlations')
    misc.savefigures(folder='images/week8',
                     filename='corr_lic_skt_shortterm_spatial')
    plt.show()

    # Plot temporal correlation of variables

    corr_SIC_temp_long = xr.corr(SIC_long,  temperature_long, dim=('x','y'))
    corr_SIC_temp_short = xr.corr(SIC_short, temperature_short, dim=('x','y'))
    corr_LIC_temp_short = xr.corr(LIC_short, temperature_short, dim=('x','y'))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.plot(corr_SIC_temp_long.time, corr_SIC_temp_long,)
    ax.set_title('Correlations between SIC and SKT')
    ax.set_ylim([-1,1])
    ax.axhline(0,color='k',alpha=0.5)
    misc.savefigures(folder='images/week8',
                     filename='corr_sic_skt_longterm_temporal')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.plot(corr_SIC_temp_short.time, corr_SIC_temp_short,)
    ax.set_title('Correlations between SIC and SKT')
    ax.set_ylim([-1,1])
    ax.axhline(0,color='k',alpha=0.5)
    misc.savefigures(folder='images/week8',
                     filename='corr_sic_skt_shortterm_temporal')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.plot(corr_LIC_temp_short.time, corr_LIC_temp_short,)
    ax.set_title('Correlations between LIC and SKT')
    ax.set_ylim([-1, 1])
    ax.axhline(0, color='k', alpha=0.5)
    misc.savefigures(folder='images/week8',
                     filename='corr_lic_skt_shortterm_temporal')
    plt.show()

    # Generate timeseries with different lengths
    SIC_short = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    LIC_short = LIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature_short = temperature.sel(
        time=slice('2002-01-01', '2019-12-31')).skt

    SIC_long = SIC.sel(time=slice('1979-01-01', '2019-12-31'))
    temperature_long = temperature.sel(
        time=slice('1979-01-01', '2019-12-31')).skt

    # Plot spatial correlation of variables
    SIC_long_anmomalous = SIC_long.pipe(w5.find_anomalies)
    SIC_short_anmomalous = SIC_short.pipe(w5.find_anomalies)
    LIC_short_anmomalous = LIC_short.pipe(w5.find_anomalies)
    temperature_short_anmomalous = temperature_short.pipe(w5.find_anomalies)
    temperature_long_anmomalous = temperature_long.pipe(w5.find_anomalies)

    
    corr_SIC_temp_long = xr.corr(SIC_long_anmomalous,  temperature_long_anmomalous, dim='time')
    corr_SIC_temp_short = xr.corr(SIC_short_anmomalous, temperature_short_anmomalous, dim='time')
    corr_LIC_temp_short = xr.corr(LIC_short_anmomalous, temperature_short_anmomalous, dim='time')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.contourf(corr_SIC_temp_short.x, corr_SIC_temp_short.y, corr_SIC_temp_short,
                       crs=ccrs.SouthPolarStereo(), cmap='PuOr', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Correlations between SIC and SKT')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Correlations')
    misc.savefigures(folder='images/week8',
                     filename='corr_sic_skt_shortterm_spatial_anmomalous')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.contourf(corr_SIC_temp_long.x, corr_SIC_temp_long.y, corr_SIC_temp_long,
                       crs=ccrs.SouthPolarStereo(), cmap='PuOr', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Correlations between SIC and SKT')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Correlations')
    misc.savefigures(folder='images/week8',
                     filename='corr_sic_skt_longterm_spatial_anmomalous')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.contourf(corr_LIC_temp_short.x, corr_LIC_temp_short.y, corr_LIC_temp_short,
                       crs=ccrs.SouthPolarStereo(), cmap='PuOr', norm=divnorm, levels=16)
    ax.coastlines()
    ax.set_title('Correlations between LIC and SKT')
    cbar = plt.colorbar(plot)
    cbar.set_label(r'Correlations')
    misc.savefigures(folder='images/week8',
                     filename='corr_lic_skt_shortterm_spatial_anmomalous')
    plt.show()

    # Plot temporal correlation of variables

    corr_SIC_temp_long = xr.corr(SIC_long_anmomalous,  temperature_long_anmomalous, dim=('x', 'y'))
    corr_SIC_temp_short = xr.corr(SIC_short_anmomalous, temperature_short_anmomalous, dim=('x', 'y'))
    corr_LIC_temp_short = xr.corr(LIC_short_anmomalous, temperature_short_anmomalous, dim=('x', 'y'))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.plot(corr_SIC_temp_long.time, corr_SIC_temp_long,)
    ax.set_title('Correlations between SIC and SKT')
    ax.set_ylim([-1, 1])
    ax.axhline(0, color='k', alpha=0.5)
    misc.savefigures(folder='images/week8',
                     filename='corr_sic_skt_longterm_temporal_anmomalous')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.plot(corr_SIC_temp_short.time, corr_SIC_temp_short,)
    ax.set_title('Correlations between SIC and SKT')
    ax.set_ylim([-1, 1])
    ax.axhline(0, color='k', alpha=0.5)
    misc.savefigures(folder='images/week8',
                     filename='corr_sic_skt_shortterm_temporal_anmomalous')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = ax.plot(corr_LIC_temp_short.time, corr_LIC_temp_short,)
    ax.set_title('Correlations between LIC and SKT')
    ax.set_ylim([-1, 1])
    ax.axhline(0, color='k', alpha=0.5)
    misc.savefigures(folder='images/week8',
                     filename='corr_lic_skt_shortterm_temporal_anmomalous')
    plt.show()

def regression_plots(SIC, LIC, temperature, landmask):

    # Generate timeseries with different lengths
    SIC_short = SIC.sel(time=slice('2002-01-01', '2019-12-31'))
    LIC_short = LIC.sel(time=slice('2002-01-01', '2019-12-31'))
    temperature_short = temperature.sel(time=slice('2002-01-01', '2019-12-31')).skt
    SIC_long = SIC.sel(time=slice('1979-01-01', '2019-12-31'))
    temperature_long = temperature.sel(time=slice('1979-01-01', '2019-12-31')).skt

    # Put everything into a dataset
    ds = xr.Dataset()
    ds['SIC_short'] = SIC_short
    ds['LIC_short'] = LIC_short
    ds['temperature_short'] = temperature_short
    ds['SIC_long'] = SIC_long
    ds['temperature_long'] = temperature_long

    # Calculate regressions
    sic_temp_long = w5.multiple_fast_regression(ds, 'SIC_long', ['temperature_long'])
    sic_temp_short = w5.multiple_fast_regression(ds, 'SIC_short', ['temperature_short'])
    lic_temp_short = w5.multiple_fast_regression(ds, 'LIC_short', ['temperature_short'])

    # Calculate trends spatially
    gradient = xr.Dataset()
    gradient['sic_temp_long']  = sic_temp_long.prediction_temperature_long.polyfit(dim='time', deg=1).copy().sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
    gradient['sic_temp_short'] = sic_temp_short.prediction_temperature_short.polyfit(dim='time', deg=1).copy().sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
    gradient['lic_temp_short'] = lic_temp_short.prediction_temperature_short.polyfit(dim='time', deg=1).copy().sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
    
    # Plot spatial trends
    for variable in gradient:
        data = gradient[variable]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        divnorm = TwoSlopeNorm(vmin=data.min().min(), vcenter=0, vmax=data.max().max())
        plot = ax.contourf(data.x, data.y, data.transpose(),
                        crs=ccrs.SouthPolarStereo(), cmap='RdBu_r', norm=divnorm, levels=16)
        ax.coastlines()
        ax.set_title(variable)
        cbar = plt.colorbar(plot)
        cbar.set_label(r'Trend in Predicted SKT [$^\circ$C yr$^{-1}$]')
        misc.savefigures(folder='images/week8',
                        filename=f'trend_predicted_{variable}')
        plt.show()

    return gradient