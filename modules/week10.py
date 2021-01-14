from tqdm import tqdm
import numba
from scipy import stats
from modules import week5 as w5
from modules import week8 as w8
from modules import combine_ice as ci
from modules import misc

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import itertools
from plyer import notification
import glob
import time

from pyproj import Proj, transform

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm

from sklearn.linear_model import LinearRegression


def process_data(dataarray, variable):
    misc.print_heading(f'Processing {variable}')
    time.sleep(0.1)
    Y, X = [10*np.arange(435000, -395000, -2500),
            10*np.arange(-395000, 395000, 2500)]
    x, y = np.meshgrid(X, Y)
    inProj = Proj(init='epsg:3031')
    outProj = Proj(init='epsg:4326')
    x, y = transform(inProj, outProj, x, y)
    x = x.flatten()
    y = y.flatten()
    if 'cc' not in variable:
        x[x < 0] = x[x < 0]+360
    x = xr.DataArray(x, dims='z')
    y = xr.DataArray(y, dims='z')

    data = dataarray.sel(latitude=slice(-40, -90))
    dims_ = ['time', 'y', 'x']
    newdata = xr.DataArray(dims=dims_, coords=[data.time, Y, X])
    for t in tqdm(data.time.values):
        subdata = data.sel(time=t)
        variable_data = subdata.interp(
            longitude=x, latitude=y, method='linear', kwargs={"fill_value": 0.0})
        newdata.loc[newdata.time == t] = variable_data.values.reshape([
            1, len(Y), len(X)])
    newdata.name = variable
    newdata.attrs = data.attrs
    newdata.to_netcdf(f'processed_data/{variable}.nc')
    print(f'{variable} processed')


def plot_scatter(data, independent, dependant, landmask=True, filename='distribution_of_temperature_ice'):

    # Normalizing variables by area of grid cells This currently doesn't work
    # area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    # SIC *= area
    # LIC *= area
    # temperature['sst'] = temperature['sst'] * area / area.sum()
    # temperature['skt'] = temperature['skt'] * area / area.sum()
    # temperature['t2m'] = temperature['t2m'] * area / area.sum()
    # SIC = data.SIC

    data = data.where(data.landmask==landmask)

    y = data[dependant].values.copy().flatten()
    x = data[independent].values.copy().flatten()


    # Creating figure and setting layout of axes
    plt.style.use('stylesheets/timeseries.mplstyle')
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    plt.suptitle(f'Joint Distribution of {independent} and {dependant}')

    # ax.axhline(0,color='k', alpha=0.3)
    # ax.axvline(0,color='k', alpha=0.3)

    mask = np.isfinite(x) * np.isfinite(y)
    X = x[mask]
    Y = y[mask]
    counts, xedges, yedges = np.histogram2d(X, Y, bins=100)
    xedges = (xedges[1:] + xedges[:-1]) / 2
    yedges = (yedges[1:] + yedges[:-1]) / 2
    im = ax.contourf(xedges, yedges, counts, norm=LogNorm())
    if data[independent].attrs != {}:
        ax.set_xlabel(
            f'{data[independent].attrs["long_name"]} [{data[independent].attrs["units"]}]')
    if data[dependant].attrs != {}:
        ax.set_ylabel(
            f'{data[dependant].attrs["long_name"]} [{data[dependant].attrs["units"]}]')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    misc.savefigures(folder='images/2021w2', filename=filename)
