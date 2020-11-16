# Data Structures
import numpy as np   # nd arrays.
import pandas as pd  # DataFrames.
import xarray as xr  # nd labeled data.

# File management
import glob

# Used to make looping over variables better.
import tqdm      # progress bars.
import itertools # better loops.


# Plotting things
# 	Static Plots
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import cm
# 	Coordinate projections and transformations.
import cartopy.crs as ccrs
# 	Interactive Plots
import plotly.express as px
import plotly.graph_objects as go


# Import my own functions for use in notebook.
# 	The modules folder contains a number of scripts written for
# 	different parts of this project.
from modules import *
