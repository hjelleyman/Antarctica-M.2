#!/usr/bin/env python
# coding: utf-8

# # Regression analysis
# 
# 

# In[16]:


import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
from matplotlib import pyplot as plt
import glob
import scipy
from modules import *

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Preprocessing
# 
# We want to apply the following to the data:
# 
#     1) Find the anomalies
#     2) Get the annual average
#     3) Normalise the indepenant variables

# In[17]:


files = glob.glob('processed_data/*')
files = [f for f in files if '_' not in f.split('\\')[1]]
ds = xr.open_mfdataset(files, parallel=True)


# In[18]:


ds


# In[19]:


ds = ds.sel(time=slice('1979-01-01','2019-12-31'))
ds = (ds
      .pipe(w5.find_anomalies)
      .pipe(w5.yearly_average)
      .pipe(w5.normalise_indepenant, dependant='sic')
     ).compute()


# In[20]:


ds


# # Correlations

# In[21]:


v = [v for v in ds]
correlation_matrix = pd.DataFrame(index=v,columns=v, dtype=np.float64)
for v1,v2 in tqdm(list(itertools.product(v,v))):
    vec1 = ds[v1].mean(dim=('x','y'))
    vec2 = ds[v2].mean(dim=('x','y'))
    correlation_matrix.loc[v1,v2]=xr.corr(vec1,vec2).values
    
def significant_bold(val, sig_level=0.9):
    bold = 'bold' if val > sig_level or val < -sig_level else ''
    return 'font-weight: %s' % bold
correlation_matrix.style.applymap(significant_bold,sig_level=0.9)


# In[22]:


fig = plt.figure(figsize=(5,5))
plt.pcolormesh(correlation_matrix)
plt.colorbar()
plt.xticks(np.arange(0,len(v))+0.5,v)
plt.yticks(np.arange(0,len(v))+0.5,v)
plt.savefig('images/week5/correlations.pdf')
plt.show()


# # Regressions

# In[23]:


x_surface = ['si10','sp', 'ssr', 'sst','t2m','u10','v10']

regression_results = w4.multiple_fast_regression(ds, 'sic', x_surface)


# In[24]:


regression_results


# In[25]:


w5.plotting(regression_results, 'sic', x_surface)


# In[26]:


w5.more_plotting(regression_results, 'sic', x_surface)


# In[27]:


stats = w4._get_stats(regression_results, 'sic' ,x_surface)
stats.name = 'Quality of Regression'
stats.index = [col.replace('_', ' ').capitalize() for col in stats.index]
stats.to_latex('..\Environmental-Impact-On-Sea-Ice\\index_stats_0.tex',  escape=False, column_format='l'*(len(stats.index)+1))


# In[28]:


results = w4.indicies_stats(ds, regression_results, 'sic', x_surface,'SouthPolarStereo', 'annual','anomalous','raw')
results = results.sort_values('individual_Predicted_Trend', ascending=False)
results.columns = [col.replace('_', ' ').capitalize() for col in results.columns]
results.transpose().to_latex('..\Environmental-Impact-On-Sea-Ice\\index_stats_1.tex',  escape=False, column_format='l'*(len(results.index)+1))
results.transpose()


# In[29]:


dependant = 't2m'
independant = ['IPO','nina34','nina12','DMI','SAM','meiv2','SOI','SAM']
files = glob.glob('processed_data/*')
files = [f for f in files if '_' not in f.split('\\')[1]]
ds = xr.open_mfdataset(files)

indicies_to_load = ['IPO','nina34','nina12','DMI','SAM','meiv2','SOI','SAM']
indicies = w5.load_indicies(indicies_to_load)
for ind in indicies:
    ds[ind] = indicies[ind]

ds = ds[independant + [dependant]]

# Preprocess the data
ds = (ds
      .pipe(w5.find_anomalies)
      .pipe(w5.yearly_average)
      .pipe(w5.normalise_indepenant, dependant=dependant)
     ).compute()

ds = ds.sel(time=slice('1979-01-01','2019-12-31'))
print(ds)

regression_results = w4.multiple_fast_regression(ds, dependant, independant)
w5.plotting(regression_results, dependant, independant)
w5.more_plotting(regression_results, dependant, independant)


# In[30]:


dependant = 'sic'
independant = ['t2m','sst','ssr']
files = glob.glob('processed_data/*')
files = [f for f in files if '_' not in f.split('\\')[1]]
ds = xr.open_mfdataset(files)

ds = ds[independant + [dependant]]

# Preprocess the data
ds = (ds
      .pipe(w5.find_anomalies)
      .pipe(w5.yearly_average)
      .pipe(w5.normalise_indepenant, dependant=dependant)
     ).compute()

ds = ds.sel(time=slice('1979-01-01','2019-12-31'))
print(ds)

regression_results = w4.multiple_fast_regression(ds, dependant, independant)
w5.plotting(regression_results, dependant, independant)
w5.more_plotting(regression_results, dependant, independant)


# In[31]:


dependant = 'sst'
independant = ['IPO','nina34','nina12','DMI','SAM','meiv2','SOI','SAM']
files = glob.glob('processed_data/*')
files = [f for f in files if '_' not in f.split('\\')[1]]
ds = xr.open_mfdataset(files)

indicies_to_load = ['IPO','nina34','nina12','DMI','SAM','meiv2','SOI','SAM']
indicies = w5.load_indicies(indicies_to_load)
for ind in indicies:
    ds[ind] = indicies[ind]

ds = ds[independant + [dependant]]

# Preprocess the data
ds = (ds
      .pipe(w5.find_anomalies)
      .pipe(w5.yearly_average)
      .pipe(w5.normalise_indepenant, dependant=dependant)
     ).compute()

ds = ds.sel(time=slice('1979-01-01','2019-12-31'))
print(ds)

regression_results = w4.multiple_fast_regression(ds, dependant, independant)
w5.plotting(regression_results, dependant, independant)
w5.more_plotting(regression_results, dependant, independant)


# In[ ]:


dependant = 'ssr'
independant = ['IPO','nina34','nina12','DMI','SAM','meiv2','SOI','SAM']
files = glob.glob('processed_data/*')
files = [f for f in files if '_' not in f.split('\\')[1]]
ds = xr.open_mfdataset(files)

indicies_to_load = ['IPO','nina34','nina12','DMI','SAM','meiv2','SOI','SAM']
indicies = w5.load_indicies(indicies_to_load)
for ind in indicies:
    ds[ind] = indicies[ind]

ds = ds[independant + [dependant]]

# Preprocess the data
ds = (ds
      .pipe(w5.find_anomalies)
      .pipe(w5.yearly_average)
      .pipe(w5.normalise_indepenant, dependant=dependant)
     ).compute()

ds = ds.sel(time=slice('1979-01-01','2019-12-31'))
print(ds)

regression_results = w4.multiple_fast_regression(ds, dependant, independant)
print(regression_results)
w5.plotting(regression_results, dependant, independant)
w5.more_plotting(regression_results, dependant, independant)


# # Indicies

# In[5]:


indicies_to_load = ['SAM', 'IPO']
# indicies = p2.load_indicies(indicies_to_load, 'monthly')


# In[8]:


w5.load_indicies(indicies_to_load)


# In[ ]:




