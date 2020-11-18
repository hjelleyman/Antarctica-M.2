"""
For this project we have multiple ice datasets. This script contains the helper functions for the notebook which is used to combine them to a new dataset:
	
	Antarctic Ice Concentration (AIC)

"""
import numpy as np
import xarray as xr
import glob
from tqdm import tqdm
from pyproj import Proj, transform

def load_seaice():
	SIC = xr.open_dataset('processed_data/seaice.nc').sic
	return SIC


def load_landice():
	files = glob.glob('data/landice2/*.nc')
	ds = xr.open_mfdataset(files)
	ds = ds.where(ds.land_mask==1).lwe_thickness
	return ds.compute()


def latlon_to_polarstereo(da):
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
	newdata = xr.DataArray(dims=('time','y','x'),coords = [da.time,Y,X])

	for time in tqdm(da.time.values):
		subdata = da.sel(time=time)
		variable_data = subdata.interp(lon=x, lat=y, method = 'linear', kwargs={"fill_value": 0.0})
		newdata.loc[newdata.time==time] = variable_data.values.reshape([1,len(Y),len(X)])
	return newdata