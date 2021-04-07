#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('NYC_2019/AB_NYC_2019.csv')
df.head()


# In[3]:


df_norm1 = df.copy()
df_norm1.head()


# In[4]:


df_norm1 = df.drop(columns=["host_name"])
df_norm1.head()


# In[5]:


df_norm1["room_type"].unique()


# In[6]:


def norm_room_type(letter: str) -> int:
    if letter == "Private room":
        return 1
    elif letter == "Entire home/apt":
        return 2
    else:
        return 3 


# In[7]:


df_norm1["room_type"] = df_norm1["room_type"].apply(norm_room_type)
df_norm1.head()


# In[8]:


df_norm1["room_type"].unique()


# In[9]:


# выбросы
print(df_norm["availability_365"].max())
print(df_norm["availability_365"].min())


# In[ ]:


print(df_norm["availability_365"][df_norm.availability_365 == 0].count())
print(df_norm["availability_365"][df_norm.availability_365 == 365].count())


# In[ ]:


print(df_norm["reviews_per_month"].max())
print(df_norm["reviews_per_month"].min())


# In[10]:


print(df_norm["reviews_per_month"][df_norm.reviews_per_month <= 1].count())
print(df_norm["reviews_per_month"][df_norm.reviews_per_month > 1].count())
print(df_norm["reviews_per_month"][df_norm.reviews_per_month == 0].count())


# In[11]:


print(df_norm["calculated_host_listings_count"].max())
print(df_norm["calculated_host_listings_count"].min())


# In[12]:


calculated_host_listings_count_stat = {"mean": df_norm["calculated_host_listings_count"].mean(),
            "median": df_norm["calculated_host_listings_count"].median(),
            "mode": df_norm["calculated_host_listings_count"].mode().to_list(),
            "interquartile_range": df_norm["calculated_host_listings_count"].quantile(0.75) - df_norm["calculated_host_listings_count"].quantile(0.25),
            }
calculated_host_listings_count_stat


# In[13]:


print(df_norm["calculated_host_listings_count"].quantile(0.05))
print(df_norm["calculated_host_listings_count"].quantile(0.95))


# In[14]:


print(df_norm["calculated_host_listings_count"][df_norm.calculated_host_listings_count < 1.0].count())
print(df_norm["calculated_host_listings_count"][df_norm.calculated_host_listings_count > 15.0].count())


# In[15]:


# Удалим выброс. Учитывая что в датасете около 50000 строк
df_norm1 = df_norm1[df_norm1.calculated_host_listings_count <= 15.0]
df_norm1


# In[16]:


from operator import itemgetter

region_dt = [(name, df["neighbourhood"].to_list().count(name)) 
                  for name in df["neighbourhood"].unique() ]
region_dt = sorted(region_dt, key=itemgetter(1))
region_dt


# In[17]:


df_norm1["neighbourhood_group"].unique()
df2 = df_norm1.copy()
df2


# In[18]:


import numpy as np


# In[19]:


def nan_to_median(cell, col) -> int:
  if np.isnan(cell):
    return 0
  else:
    return cell
    


# In[20]:


df_norm1["reviews_per_month"] = df_norm1["reviews_per_month"].apply(nan_to_median, col=df_norm1["reviews_per_month"].to_list())
df_norm1.head()
vy = df_norm1.copy()
vy


# In[21]:


from astropy.time import Time
import datetime
def nan_to_date(cell, col):
    if cell != cell:
        return '2001-01-01'
    else:
        return cell
df_norm1["last_review"] = df_norm1["last_review"].apply(nan_to_date, col=vy["last_review"].to_list())
df_norm1["last_review"] = df_norm1["last_review"].astype('datetime64[D]')
df_norm1["last_review"]


# In[22]:


df_norm1

