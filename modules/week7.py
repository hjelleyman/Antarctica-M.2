"""Code for the week 7. This is used for the week 7 notebook.
"""


# -------------- SST WITH ICE --------------

# Checking SST has data when ice is covering

# Load the two datasets
def load_sst_sic():
	SST = None
	SIC = None
	return SST, SIC

# select locations where there is seaice.
# pick some at random and make timeseries plots
def select_over_ice_and_plot(SST, SIC):
	return None

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
def plot_mean_sst_timeseries(ERA5,ARGO):
	return None

# Plot the spatial distribution of trned in data.
def plot_spatial_sst_trends(ERA5,ARGO):
	return None

# Calculate statistics for quality of ocean data.
def generate_sst_statistics(ERA5,ARGO):
	stats = None
	return stats

# -------------- AIR TEMPERATURE VERIFICATION --------------

# Download Station Data
def download_station_data():
	return None

# Load in datasets
def load_air_temp_data():
	T2M = None
	STT = None
	return T2M, STT

# Plot timeseries.
def plot_station_timeseries(T2M, STT):
	return None

# Calculate statistics for station temperature.
def generate_station_statistics(T2M, STT):
	stats = None
	return stats