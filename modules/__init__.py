"""modules used for my MSc project.
"""
# For labeled notebooks.
from modules import dataprocessing as dp    # for Processing data
from modules import correlations   as corr  # Correlations
from modules import regressions    as regr  # Regressions
from modules import composites     as comp  # Composites [unused]
from modules import plotting       as plot  # First attempts at plotting
from modules import misc           as misc  # Miscelaneous helper functions
from modules import plotting2      as p2    # 2nd set of plotting functions

# The project got more complicated so I turned to breaking the work up into weekly chunks
from modules import week2          as w2
from modules import week3          as w3
from modules import week4          as w4
from modules import week5          as w5
from modules import week7          as w7

# When new land ice data was downloaded I made this script to process it.
from modules import combine_ice    as ci