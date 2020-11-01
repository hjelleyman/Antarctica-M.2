#!/usr/bin/env python
# coding: utf-8

# In[34]:


from glob import glob
from tika import parser # pip install tika
import itertools


# In[35]:


files = glob('papers/*')
content = {}


# In[36]:


for file in files:
    content[file] = parser.from_file(file)['content']


# In[37]:


import pandas as pd


# In[39]:


for file in files:
    words = [c.split(' ') for c in content[file].split('\n') if c != '']
    words = list(itertools.chain.from_iterable(words))
    content[file] = words


# In[41]:


N = max([len(content[file]) for file in files])


# In[42]:




