#!/usr/bin/env python
# coding: utf-8

# # Standardize data.
# 
# Before we begin we want to standardize the different datasets so they can be processed quickly.

# In[1]:


from modules import *
import glob

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Processing the seaice dataset
files = glob.glob('data/NSIDC/*')
seaice_data = w5.process_seaice(files)
seaice_data.name = 'sic'
seaice_data.to_netcdf('processed_data/seaice.nc')
seaice_data


# In[3]:


# processing the era5 dataset
w5.process_variables()


# In[ ]:




