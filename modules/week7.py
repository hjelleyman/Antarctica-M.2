"""Code for the week 7. This is used for the week 7 notebook.
"""

from pyproj import Transformer, CRS
import xarray as xr       # Storing data in n-dimensional datasets
import numpy as np       # Numerical Opperations
import matplotlib.pyplot as plt      # Making plots and figures
import matplotlib.dates as mdates   # Formatting time labels on figures
import cartopy.crs as ccrs     # Geographic projection for plotting
import itertools                     # Make loops more efficient.
import glob                          # Find all the files in a directory.
from pyproj import Proj, transform
import pandas as pd

# 	Interactive Plots
import plotly.express as px
import plotly.graph_objects as go


from modules import misc


# -------------- SST WITH ICE --------------

# Checking SST has data when ice is covering

# Load the two datasets
def load_sst_sic():
    SST = xr.open_dataset('processed_data/sst.nc').sst
    SIC = xr.open_dataset('processed_data/seaice.nc').sic
    SKT = xr.open_dataset('processed_data/skt.nc').skt
    T2M = xr.open_dataset('processed_data/t2m.nc').t2m

    data = xr.Dataset()
    data['sst'] = SST
    data['sic'] = SIC
    data['skt'] = SKT
    data['t2m'] = T2M

    return data

# select locations where there is seaice.
# pick some at random and make timeseries plots


def select_over_ice_and_plot(data, n):
    """Randomly selects n points in the dataset which sometimes have some sea ice and sometimes doesn't.
    Makes a plot of the different temperatures we are looking at alongside a plot of Sea Ice.
    We can use this to make judgements as to how sea ice is responding to temperature changes over time.
    """
    i = 1
    data = data.sel(time=slice('1990-01-01', '1992-12-31'))
    plotted_x = []
    plotted_y = []
    X = data.x.values
    Y = data.y.values
    X, Y = np.meshgrid(X, Y)
    X = X.flatten()
    Y = Y.flatten()
    np.random.shuffle(X)
    np.random.shuffle(Y)
    for x, y in zip(X.flatten(), Y.flatten()):
        subdata = data.sel(x=x, y=y)
        # subdata = subdata.where(subdata>0)
        if subdata.sic.min() == 0 and subdata.sic.max() <= 250 and subdata.sic.max() > 2.5:
            # Adding coordinates so we can make scatter plot
            plotted_x += [x]
            plotted_y += [y]

            # Making subplots
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)

            # Formatting the xaxis ticks
            ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ln1 = ax.plot(subdata.time, subdata.sst-273.15, label='SST')
            ln2 = ax.plot(subdata.time, subdata.skt-273.15, label='SKT')
            ln3 = ax.plot(subdata.time, subdata.t2m-273.15, label='T2M')
            ax.set_ylabel('Temperature [$^\circ$C]')
            ax2.plot([], [])
            ax2.plot([], [])  # These are to cycle the color options so SIC
            ax2.plot([], [])  # is a different temperature to the temperatures
            ln4 = ax2.plot(subdata.time, subdata.sic/2.5, label='SIC')
            ax2.set_ylabel('SIC [%]')

            # If there is ice I will highlight the area so we can look at what temperature is doing.
            ax.fill_between(subdata.time, 0, 1, where=subdata.sic > 0,
                            color='black', alpha=0.2, transform=ax.get_xaxis_transform(), step='pre')
            ax2.fill_between(subdata.time, 0, 1, where=subdata.sic > 0,
                             color='black', alpha=0.2, transform=ax2.get_xaxis_transform(), step='pre')

            lines = ln1 + ln2 + ln3 + ln4
            labels = [line.get_label() for line in lines]
            fig.suptitle(f'Location number {i}')
            plt.legend(lines, labels, bbox_to_anchor=(
                0.99, -0.15), ncol=5, loc='upper right')
            misc.savefigures(folder='images/week7',
                             filename=f'temperature_ice_timeseries_{i}')
            plt.show()
            plt.close()
            i += 1
        if i == n+1:
            break
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.scatter(plotted_x, plotted_y, transform=ccrs.SouthPolarStereo())
    for i in range(n):
        ax.annotate(i+1, (plotted_x[i], plotted_y[i]))
    ax.set_xlim([min(data.x), max(data.x)])
    ax.set_ylim([min(data.y), max(data.y)])
    ax.coastlines()
    misc.savefigures(folder='images/week7',
                     filename='temperature_ice_scatter')
    plt.show()

# -------------- SST VERIFICATION --------------

# Acquire ARGO data  and Ensure data is on the same grids


def download_argo_data():
    return None

# Load in the two datasets


def load_sst_data():
    ERA5 = None
    ARGO = None
    return ERA5, ARGO

# Plot mean time series over our relevant areas


def plot_mean_sst_timeseries(ERA5, ARGO):
    return None

# Plot the spatial distribution of trned in data.


def plot_spatial_sst_trends(ERA5, ARGO):
    return None

# Calculate statistics for quality of ocean data.


def generate_sst_statistics(ERA5, ARGO):
    stats = None
    return stats

# -------------- AIR TEMPERATURE VERIFICATION --------------

# Download Station Data


def download_station_data():
    files = glob.glob('data/stations/*.nc')
    for file in files:
        print(xr.open_dataset(file, decode_times=False))
    # STT = xr.open_mfdataset(files, parallel=True, compat='override', decode_times=False)
    return None

# Load in datasets


def load_air_temp_data():
    T2M = None
    STT = None

    files = glob.glob('data/stations/*.nc')
    STT = xr.Dataset()
    locations = {}
    for file in files:
        data = xr.open_dataset(file, decode_times=False)
        units, reference_date = data.time.attrs['units'].split('since')
        data['time'] = pd.date_range(
            start=reference_date, periods=data.sizes['time'], freq='MS')
        STT[data.description] = data.tavg
        locations[data.description] = {}
        locations[data.description]['longitude'] = np.float(
            data.longitude.strip().split(' ')[0])
        locations[data.description]['latitude'] = np.float(
            data.latitude .strip().split(' ')[0])
    return T2M, STT, locations


# Plot timeseries.


def plot_station_timeseries(T2M, STT, locations):
    df = STT.to_dataframe()
    df = df.stack()
    df = df.reset_index()
    df.columns = ['Time', 'Station', 'Temperature']
    fig = px.line(df, x="Time", y="Temperature", color='Station',
                  title='Station Temperature Data Across Antarctica')
    fig.show()

    df.to_csv('images/week7/station_data.csv')

    fig_locations, ax_locations = plt.subplots(
        figsize=(10, 10), subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
    ax_locations.coastlines()
    locations_scatter = ax_locations.scatter(x=[locations[station]['longitude'] for station in STT.variables if station != 'time'], y=[
                                             locations[station]['latitude'] for station in STT.variables if station != 'time'], transform=ccrs.PlateCarree())
    x = 10*np.arange(-395000, 395000, 2500)
    y = 10*np.arange(435000, -395000, -2500)
    ax_locations.scatter(*np.meshgrid(x, y), alpha=0)
    plt.show()
    return


def station_scatter(STT, locations):
    # Loading data
    temperature = xr.open_dataarray(
        'data/ERA5/skin_temperature.nc').sel(latitude=slice(-40, -90)).sel(expver=1).sel(time=slice('1979-01-01', '2019-12-31'))
    STT = STT.sel(time=slice('1979-01-01', '2019-12-31'))

    # Interpolating
    x = xr.DataArray([locations[station]['longitude']
                      for station in STT.variables if station != 'time'], dims="z")
    x['z'] = [station for station in STT.variables if station != 'time']
    y = xr.DataArray([locations[station]['latitude']
                      for station in STT.variables if station != 'time'], dims="z")
    ERA_data = temperature.interp(
        longitude=x, latitude=y, method='linear', kwargs={"fill_value": np.nan})

    time_union = pd.date_range(
        start='1979-01-01', periods=STT.sizes['time'], freq='MS')

    # Plotting
    x = []
    y = []
    for station in STT.variables:
        if station != 'time':
            plt.scatter(STT[station].sel(time=time_union)+273.15,
                        ERA_data.sel(z=station).sel(time=time_union), label = station)
            x += list(STT[station].sel(time=time_union).values+273.15)
            y += list(ERA_data.sel(z=station).sel(time=time_union).values)
    min_ = np.min([np.nanmin(y), np.nanmin(x)])
    max_ = np.max([np.nanmax(y), np.nanmax(x)])

    x = np.array(x)
    y = np.array(y)

# Fitting data to line
    mask = np.isfinite(x) * np.isfinite(y)
    x = x[mask]
    y = y[mask]
    p, cov = np.polyfit(x,y,1,cov=True)
    yfit = np.linspace(min_, max_)*p[0] + p[1]
    V = np.ones([50,2])
    V[:, 1] = np.linspace(min_, max_)
    var = (V.dot(cov.dot(V.transpose())))

    plt.title('Reanalysis skin temperature plotted against station data')
    plt.xlabel('Station measured temperature [K]')
    plt.ylabel('Reanalysis calculated temperature [K]')

    plt.text(min_, max_*0.95, f'Gradient {p[0]:.3f}$\pm${np.sqrt(cov[0,0]):.3f}')
    plt.plot(np.linspace(min_, max_), np.linspace(min_, max_), color='black', label='Diagonal')
    plt.plot(np.linspace(min_, max_), yfit, color='black', linewidth=3, label='Best fit')
    # plt.fill_between(np.linspace(min_, max_), yfit+var, yfit-var)
    plt.legend(bbox_to_anchor=(0.99, -0.15), ncol=3, loc='upper right')
    
    misc.savefigures(folder='images/week7',
                     filename='temperature_station_verification_scatter')
    plt.show()
        

# Calculate statistics for station temperature.


def generate_station_statistics(T2M, STT):
    stats = None
    return stats
