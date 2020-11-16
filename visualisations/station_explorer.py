# Script to run streamlit visualisations of station temperature data.

import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
from modules import *

# # Title for app
# st.title('Weather station temperature data over Antarctica')

# # Load Data
# data_load_state = st.text('Loading data...')
# T2M, STT, locations = w7.load_air_temp_data()
# data_load_state.text('Loading data...done!')


# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(locations)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)