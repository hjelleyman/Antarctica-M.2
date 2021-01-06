from tqdm import tqdm
import numba
from scipy import stats
from modules import week5 as w5
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

from sklearn.linear_model import LinearRegression


