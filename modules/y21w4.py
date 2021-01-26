# # Homemade models
# from modules import week8 as w8
# from modules import week5 as w5
# from modules import week9 as w9
# from modules import week10 as w10
# from modules import misc

# # Calculations and data stuff
# import numpy as np
# import xarray as xr
# from pyproj import Proj, transform

# # Better loops
# import itertools

# # plotting
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm


# def process_data():
#     # Cloud Cover
#     clouds = xr.open_dataset('data/ERA5/clouds.nc')
#     cc = clouds.cc.sel(level=700).sel(expver=1)
#     w10.process_data(cc, variable = 'cc_700')
#     cc = clouds.cc.sel(level=500).sel(expver=1)
#     w10.process_data(cc, variable = 'cc_500')
#     cc = clouds.cc.sel(level=200).sel(expver=1)
#     w10.process_data(cc, variable = 'cc_200')

#     # Ozone
#     ozone = xr.open_dataset('data/ERA5/ozone_mass_mixing_ratio.nc')
#     o3 = ozone.o3.sel(level=700).sel(expver=1)
#     w10.process_data(o3, variable = 'o3_700')
#     o3 = ozone.o3.sel(level=500).sel(expver=1)
#     w10.process_data(o3, variable = 'o3_500')
#     o3 = ozone.o3.sel(level=200).sel(expver=1)
#     w10.process_data(o3, variable = 'o3_200')

#     ozone = xr.open_dataset('data/ERA5/ozone_mass_mixing_ratio_50_100.nc')
#     o3 = ozone.o3.sel(level=50).sel(expver=1)
#     w10.process_data(o3, variable = 'o3_50')
#     o3 = ozone.o3.sel(level=100).sel(expver=1)
#     w10.process_data(o3, variable = 'o3_100')

#     Wind Velocities
#     winds = xr.open_dataset('data/ERA5/windspeed.nc')
#     u = winds.u.sel(level=700).sel(expver=1)
#     w10.process_data(u, variable = 'u_700')
#     u = winds.u.sel(level=500).sel(expver=1)
#     w10.process_data(u, variable = 'u_500')
#     u = winds.u.sel(level=200).sel(expver=1)
#     w10.process_data(u, variable = 'u_200')
#     v = winds.v.sel(level=700).sel(expver=1)
#     w10.process_data(v, variable = 'v_700')
#     v = winds.v.sel(level=500).sel(expver=1)
#     w10.process_data(v, variable = 'v_500')
#     v = winds.v.sel(level=200).sel(expver=1)
#     w10.process_data(v, variable = 'v_200')

#     # Radiation
#     surface_net_solar_radiation = xr.open_dataset('data/ERA5/surface_net_solar_radiation.nc')
#     surface_solar_radiation_downwards = xr.open_dataset('data/ERA5/surface_solar_radiation_downwards.nc')
#     ssr = surface_net_solar_radiation.ssr.sel(expver=1)
#     ssrd = surface_solar_radiation_downwards.ssrd.sel(expver=1)
#     ssru = ssr-ssrd
#     w10.process_data(ssr, variable = 'ssr')
#     w10.process_data(ssr, variable = 'ssrd')
#     w10.process_data(ssr, variable = 'ssru')

#     # precipitation
#     precipitation = xr.open_dataarray('data/ERA5/precipitation.nc')
#     # precipitation
#     w10.process_data(precipitation, variable = 'tp')

#     # geopotential
#     geopotential = xr.open_dataset('data/ERA5/geopotential2.nc')
#     z = geopotential.z.sel(level=700).sel(expver=1)
#     w10.process_data(z, variable = 'z_700')
#     z = geopotential.z.sel(level=500).sel(expver=1)
#     w10.process_data(z, variable = 'z_500')
#     z = geopotential.z.sel(level=200).sel(expver=1)
#     w10.process_data(z, variable = 'z_200')

# def load_data():

#     # Loading data from file
#     SIC, LIC = w8.load_ice_data()
#     SIC = SIC.sel(time=slice('1979-01-01', '2019-12-31')
#                 ).compute().pipe(w5.yearly_average)
#     LIC = LIC.sel(time=slice('1979-01-01', '2019-12-31')
#                 ).compute().pipe(w5.yearly_average)
#     temperature = w8.load_temp_data().sel(time=slice('1979-01-01', '2019-12-31')
#                                         ).compute().pipe(w5.yearly_average)

#     # Landmask
#     landmask = w8.load_landmask()
#     landmask = landmask >= .5

#     # New variables
#     cc_700 = xr.open_dataarray('processed_data/cc_700.nc').pipe(w5.yearly_average)
#     cc_500 = xr.open_dataarray('processed_data/cc_500.nc').pipe(w5.yearly_average)
#     cc_200 = xr.open_dataarray('processed_data/cc_200.nc').pipe(w5.yearly_average)
#     o3_700 = xr.open_dataarray('processed_data/o3_700.nc').pipe(w5.yearly_average)
#     o3_500 = xr.open_dataarray('processed_data/o3_500.nc').pipe(w5.yearly_average)
#     o3_200 = xr.open_dataarray('processed_data/o3_200.nc').pipe(w5.yearly_average)
#     o3_100 = xr.open_dataarray('processed_data/o3_100.nc').pipe(w5.yearly_average)
#     o3_50 = xr.open_dataarray('processed_data/o3_50.nc').pipe(w5.yearly_average)
#     u_700 = xr.open_dataarray('processed_data/u_700.nc').pipe(w5.yearly_average)
#     u_500 = xr.open_dataarray('processed_data/u_500.nc').pipe(w5.yearly_average)
#     u_200 = xr.open_dataarray('processed_data/u_200.nc').pipe(w5.yearly_average)
#     v_700 = xr.open_dataarray('processed_data/v_700.nc').pipe(w5.yearly_average)
#     v_500 = xr.open_dataarray('processed_data/v_500.nc').pipe(w5.yearly_average)
#     v_200 = xr.open_dataarray('processed_data/v_200.nc').pipe(w5.yearly_average)
#     ssrd = xr.open_dataarray('processed_data/ssrd.nc').pipe(w5.yearly_average)
#     ssru = xr.open_dataarray('processed_data/ssru.nc').pipe(w5.yearly_average)
#     ssr = xr.open_dataarray('processed_data/ssr.nc').pipe(w5.yearly_average)
#     tp = xr.open_dataarray('processed_data/tp.nc').pipe(w5.yearly_average)
#     z_700 = xr.open_dataarray('processed_data/z_700.nc').pipe(w5.yearly_average)
#     z_500 = xr.open_dataarray('processed_data/z_500.nc').pipe(w5.yearly_average)
#     z_200 = xr.open_dataarray('processed_data/z_200.nc').pipe(w5.yearly_average)

#     data = xr.Dataset()
#     data['SIC'] = SIC
#     data['LIC'] = LIC
#     data['landmask'] = landmask
#     data['cc_700'] = cc_700
#     data['cc_500'] = cc_500
#     data['cc_200'] = cc_200
#     data['o3_700'] = o3_700
#     data['o3_500'] = o3_500
#     data['o3_200'] = o3_200
#     data['u_700'] = u_700
#     data['u_500'] = u_500
#     data['u_200'] = u_200
#     data['v_700'] = v_700
#     data['v_500'] = v_500
#     data['v_200'] = v_200
#     data['o3_100'] = o3_100
#     data['o3_50'] = o3_50
#     data['ssrd'] = ssrd
#     data['ssru'] = ssru
#     data['ssr'] = ssr
#     data['skt'] = temperature.skt
#     data['tp'] = tp
#     data['z_700'] = z_700
#     data['z_500'] = z_500
#     data['z_200'] = z_200

#     data['LIC'] = data.LIC.where(landmask)

#     attrs = {v: data[v].attrs for v in data}

#     for variable, attributes in attrs.items():
#         data[variable].attrs = attributes

#     misc.print_heading("Data Loaded")

#     return data


# def plot_mean_spatial_distribution(data, name):
#     data_mean = data.mean(dim='time')
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
#     vmax = max(-data_mean.min(), data_mean.max())
#     vmin = -vmax
#     divnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
#     contour = plt.contourf(data_mean.x, data_mean.y, data_mean.values,
#                            cmap='RdBu_r', norm=divnorm, transform=ccrs.SouthPolarStereo())
#     ax.coastlines()
#     ax.set_title(f'Mean {name} from 1979 to 2019')
#     fig.subplots_adjust(right=0.95)
#     cbar_ax = fig.add_axes([0.93, 0.2, 0.05, 0.6])
#     cbar = fig.colorbar(contour, cax=cbar_ax, shrink=0.88)
#     if data.attrs != {}:
#         cbar.ax.set_ylabel(
#             f'{data.attrs["long_name"]} [{data.attrs["units"]}]')
#     misc.savefigures(folder=f'images/2021w4/mean_spatial_plots',
#                      filename=f'{data.name}')
#     plt.show()


# namedict = {'SIC': 'SIC',
#             'LIC': 'LIC',
#             'landmask': 'landmask',
#             'cc_700': 'Cloud cover at 700 hPa',
#             'cc_500': 'Cloud cover at 500 hPa',
#             'cc_200': 'Cloud cover at 200 hPa',
#             'o3_700': 'O3 Mixing Ratio at 700 hPa',
#             'o3_500': 'O3 Mixing Ratio at 500 hPa',
#             'o3_200': 'O3 Mixing Ratio at 200 hPa',
#             'o3_100': 'O3 Mixing Ratio at 100 hPa',
#             'o3_50': 'O3 Mixing Ratio at 50 hPa',
#             'u_700': 'U wind component at 700 hPa',
#             'u_500': 'U wind component at 500 hPa',
#             'u_200': 'U wind component at 200 hPa',
#             'v_700': 'V wind compnent at 700 hPa',
#             'v_500': 'V wind compnent at 500 hPa',
#             'v_200': 'V wind compnent at 200 hPa',
#             'ssrd': 'Surface Solar Radiation Downwards',
#             'ssru': 'Surface Solar Radiation Upwards',
#             'skt': 'Skin Temperature',
#             'tp': 'Total Precipitation',
#             'z_700': 'Geopotential at 700 hPa',
#             'z_500': 'Geopotential at 500 hPa',
#             'z_200': 'Geopotential at 200 hPa', }




# def main():