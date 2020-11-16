# Script to run streamlit visualisations of station temperature data.

import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt      # Making plots and figures
import matplotlib.dates  as mdates   # Formatting time labels on figures
import cartopy.crs 	     as ccrs     # Geographic projection for plotting
import matplotlib as mpl
from pyproj import Proj, transform
from modules import *


# Title for app
st.title('Weather Station Temperature data over Antarctica')

# Load Data
def load_data():
	return w7.load_air_temp_data()

data_load_state = st.text('Loading data...')
T2M, STT, locations = w7.load_air_temp_data()
data_load_state.text('Loading data...done!')

# Checkboxes
show_data = st.sidebar.checkbox('Display Raw Data', value=False, key=None)
plot_station_data = st.sidebar.checkbox('Plot Station Data', value=True, key=None)


if show_data:
	df = STT.to_dataframe()
	# df = df.stack()
	st.header('Raw Station Data')
	st.write(df)



if plot_station_data:

	st.sidebar.header('Stations to Plot')
	station = st.sidebar.selectbox('Which Station Should we Plot?', (['Multiple Stations']+[station for station in STT.variables if station !='time']))


	mpl.style.use('stylesheets/contour.mplstyle')
	st.header(f'Location of Stations')
	fig_locations, ax_locations  = plt.subplots(figsize=(5,5),subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
	ax_locations.coastlines()
	locations_scatter = ax_locations.scatter(x = [locations[station]['longitude'] for station in STT.variables if station !='time'], y = [locations[station]['latitude'] for station in STT.variables if station !='time'], transform=ccrs.PlateCarree())
	for i, location in enumerate([station for station in STT.variables if station !='time']):
		x,y = locations_scatter.get_offsets()[i]
		inProj = Proj(init='epsg:3031')
		outProj = Proj(init='epsg:4326')
		x,y = transform(outProj, inProj,x,y)
		ax_locations.annotate(xy=(x, y),
							  text=location,
							  fontsize=5
							  )


	if station in STT.variables:
		selected = ax_locations.scatter(x = locations[station]['longitude'], y = locations[station]['latitude'], transform=ccrs.PlateCarree())
	x = 10*np.arange(-395000,395000,2500)
	y = 10*np.arange(435000,-395000,-2500)
	ax_locations.scatter(*np.meshgrid(x,y),alpha=0)
	st.pyplot(fig=fig_locations)

	mpl.style.use('stylesheets/timeseries.mplstyle')

	if station != 'Multiple Stations':
		st.header(f'Timeseries of average temperature at {station}')

		data = STT[station]
		fig, ax  = plt.subplots()
		ax.plot(data)
		ax.set_ylabel('Temperature [$^\\circ$C]')
		ax.set_xlabel('Time [Units still to fix]')
		st.pyplot(fig=fig)

		locations_scatter = ax_locations.scatter(x = locations[station]['longitude'], y = locations[station]['latitude'], transform=ccrs.PlateCarree(), color='red')
		# st.pyplot(fig=fig_locations)


	else: 
		data = STT
		fig, ax  = plt.subplots()
		# ax.plot(data)

		st.pyplot(fig=fig)