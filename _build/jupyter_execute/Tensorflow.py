#!/usr/bin/env python
# coding: utf-8

# # Regression analysis
# 
# 

# In[1]:


import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
from matplotlib import pyplot as plt
import glob
import scipy
from modules import *
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Preprocessing
# 
# We want to apply the following to the data:
# 
#     1) Find the anomalies
#     2) Get the annual average
#     3) Normalise the indepenant variables

# In[2]:


# files = glob.glob('processed_data/*')
# files = [f for f in files if '_' not in f.split('\\')[1]]
# ds = xr.open_mfdataset(files, chunks = {'x':100,'y':100})


# In[3]:


# ds


# In[4]:


# ds = ds.sel(time=slice('1979-01-01','2019-12-31'))
# ds = (ds
#       .pipe(w5.find_anomalies)
#       .pipe(w5.yearly_average)
#       .pipe(w5.normalise_indepenant, dependant='sic')
#      )

# ds.to_netcdf('processed_data/temp.nc')


# In[46]:


ds = xr.open_dataset('processed_data/temp.nc')
y  =ds['sic'].copy()
ds = ds.drop('sic')
ds


# # Correlations

# In[47]:


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


# In[48]:


fig = plt.figure(figsize=(5,5))
plt.pcolormesh(correlation_matrix)
plt.colorbar()
plt.xticks(np.arange(0,len(v))+0.5,v)
plt.yticks(np.arange(0,len(v))+0.5,v)
plt.savefig('images/week5/correlations.pdf')
plt.show()


# # running a NN model

# In[49]:


data = ds.to_array()
data


# In[50]:


data = data.stack(X=('variable','x','y'))


# In[51]:


df = data.to_pandas()
df.head()


# In[57]:


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)


# In[59]:


y = y.stack(z=('x','y')).to_pandas()
y


# In[60]:


y_train = y.loc[train_dataset.index]
y_test = y.loc[test_dataset.index]


# In[68]:


simple_model = tf.keras.Sequential([
    layers.Dense(units=1000),
    layers.Dense(units=1000),
    layers.Dense(units=y_test.shape[1])
])
simple_model.predict(test_dataset[:10])
simple_model.summary()


# In[69]:


simple_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[85]:


history = simple_model.fit(
    train_dataset,
    y_train,
    epochs=1000,
    # suppress logging
#     verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


# In[ ]:


plt.scatter(simple_model.predict(test_dataset),y_test.values)


# In[79]:


y_test.iloc[0]

