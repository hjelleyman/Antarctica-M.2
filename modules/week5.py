import numpy as np
import datetime
import xarray as xr
from pyproj import Proj, transform
import xarray as xr
from tqdm import tqdm
import itertools
import scipy
from glob import glob

# For printing headings
from modules.misc import print_heading


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
	print(files)
	for file in files:
		data = xr.open_dataset(file)
		data = data.sel(latitude=slice(-40,-90)).sel(expver=1)
		data = data.resample(time="YS").mean()
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