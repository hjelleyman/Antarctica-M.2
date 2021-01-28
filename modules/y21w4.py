import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm
import cartopy.crs as ccrs
from modules import misc
import time
import numpy as np
from tqdm import tqdm
from pyproj import Proj, transform


def load_new_landice():
    directory = 'data/landice3'
    da = xr.open_mfdataset(directory+'/*.nc')

    LIC = da.lwe_thickness
    misc.print_heading(f'Processing LIC')
    time.sleep(0.1)
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

    data = LIC.sel(lat=slice(-90, -40))
    dims_ = ['time', 'y', 'x']
    newdata = xr.DataArray(dims=dims_, coords=[data.time, Y, X])
    for t in tqdm(data.time.values):
        subdata = data.sel(time=t)
        variable_data = subdata.interp(
            lon=x, lat=y, method='linear', kwargs={"fill_value": 0.0})
        newdata.loc[newdata.time == t] = variable_data.values.reshape([
            1, len(Y), len(X)])
    newdata.name = 'LIC'
    newdata.attrs = data.attrs

    return newdata.compute()
