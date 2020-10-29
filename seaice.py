import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import glob
import tqdm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import cm
import itertools
import time
import streamlit as st

    
    
files = glob.glob('processed_data/*')
variables  = ['t2m.nc', 'skt.nc', 'sst.nc']
files = [f for f in files if '_' not in f.split('\\')[1] and  f.split('\\')[1] in variables]
ds = xr.open_mfdataset(files, parallel=True, compat='override')

ds = ds.compute()
print(ds)