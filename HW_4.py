#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('NYC_2019/AB_NYC_2019.csv')
df.head()


# In[2]:


df_norm1 = df.copy()
df_norm1.head()


# In[3]:


df_norm1 = df.drop(columns=["host_name"])
df_norm1.head()


# In[164]:


df_norm1["room_type"].unique()


# In[175]:


df_norm1["last_review"].unique()


# In[5]:


def norm_room_type(letter: str) -> int:
    if letter == "Private room":
        return 1
    elif letter == "Entire home/apt":
        return 2
    else:
        return 3 


# In[6]:


df_norm1["room_type"] = df_norm1["room_type"].apply(norm_room_type)
df_norm1.head()


# In[7]:


df_norm1["room_type"].unique()


# In[208]:


df_norm1["last_review"].unique()


# In[8]:


# Удалим выброс. Учитывая что в датасете около 50000 строк
df_norm1 = df_norm1[df_norm1.calculated_host_listings_count <= 15.0]
df_norm1


# In[58]:


from operator import itemgetter

region_dt = [(name, df["neighbourhood"].to_list().count(name)) 
                  for name in df["neighbourhood"].unique() ]
region_dt = sorted(region_dt, key=itemgetter(1))
region_dt


# In[10]:


df_norm1["neighbourhood_group"].unique()
df2 = df_norm1.copy()
df2


# In[11]:


import numpy as np


# In[12]:


def nan_to_median(cell, col) -> int:
  if np.isnan(cell):
    return 0
  else:
    return cell
    


# In[13]:


df_norm1["reviews_per_month"] = df_norm1["reviews_per_month"].apply(nan_to_median, col=df_norm1["reviews_per_month"].to_list())
df_norm1.head()
vy = df_norm1.copy()
vy


# In[14]:


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


# In[15]:


df_norm1


# In[117]:


# Дз 4 
from operator import itemgetter
import matplotlib.pyplot as plt
# самые часто встречаемые районы 
region_dt = [(name, df_norm1["neighbourhood"].to_list().count(name)) 
                  for name in df_norm1["neighbourhood"].unique() 
                  if df_norm1["neighbourhood"].to_list().count(name) > 1200]
region_dt = sorted(region_dt, key=itemgetter(1))
regions = []
counts = []
for region, count in region_dt :
  regions.append(region)
  counts.append(count)
plt.barh(regions, counts)


# In[64]:


region_dt = pd.DataFrame.from_records(region_dt, 
                                         columns=["Region", "Total"])
region_dt


# In[17]:


# тип жилья по всем районам
room_type_dt = {room_type: df_norm1["room_type"].to_list().count(room_type) for room_type in set(df_norm1["room_type"])}
print(room_type_dt) 


# In[18]:


cases1 = []
room_1 = []
for room_type, cases_num in room_type_dt.items():
  room_1.append(room_type)
  cases1.append(cases_num)

room_info_1 = pd.DataFrame({"": cases1},
                       index=['Private room','Entire home/apt', 'Shared room'])

cases2 = []
room_2 = []
for room_type, cases_num in room_type_dt.items():
  if room_type != 3:
    room_2.append(room_type)
    cases2.append(cases_num)

room_info_2 = pd.DataFrame({"":cases2}, index=['Private room','Entire home/apt'])

fig, axes = plt.subplots(nrows=1, ncols=2)
room_info_1.plot.pie(ax=axes[0], y="",
                 colors=["mediumspringgreen", "orange" ,"deepskyblue"],
                 autopct="%.1f%%",
                 legend = False,
                 fontsize=13,
                 figsize=(12, 12));
room_info_2.plot.pie(ax=axes[1], y="",
                 colors=["mediumspringgreen", "deepskyblue",],
                 autopct="%.1f%%",
                 legend = False,
                 fontsize=13,
                 figsize=(12, 12));


# In[19]:


# проверим остается ли такое распределение в найденных раннее районах
data_region = [(name,
                   df_norm1["neighbourhood"].to_list().count(name),
                   df_norm1.query("room_type == 2")["neighbourhood"].to_list().count(name),
                   df_norm1.query("room_type == 1")["neighbourhood"].to_list().count(name),
                   df_norm1.query("room_type == 3")["neighbourhood"].to_list().count(name)) 
                  for name in df_norm1["neighbourhood"].unique() 
                  if df_norm1["neighbourhood"].to_list().count(name) > 1200]

data_region = sorted(data_region, key=itemgetter(1))
region_df = pd.DataFrame.from_records(data_region, 
                                         columns=["Region", "Total", "Entire home/apt", "Private room","Shared room"])
region_df


# In[20]:


df_draw = pd.DataFrame({"Entire home/apt": region_df["Entire home/apt"].to_list(),
                        "Private room": region_df["Private room"].to_list(),
                        "Shared room": region_df["Shared room"].to_list()},
                       index=region_df["Region"].to_list())
df_draw.plot.barh(color=["deepskyblue", "mediumspringgreen", "orange"],
                  stacked=True,
                  figsize=(12, 5));


# In[21]:


data_region_1 = [(name,
                   df_norm1["neighbourhood_group"].to_list().count(name),
                   df_norm1.query("room_type == 2")["neighbourhood_group"].to_list().count(name),
                   df_norm1.query("room_type == 1")["neighbourhood_group"].to_list().count(name),
                   df_norm1.query("room_type == 3")["neighbourhood_group"].to_list().count(name)) 
                  for name in df_norm1["neighbourhood_group"].unique() 
                  if df_norm1["neighbourhood_group"].to_list().count(name) > 0]

data_region_1 = sorted(data_region_1, key=itemgetter(1))
region_df_1 = pd.DataFrame.from_records(data_region_1, 
                                         columns=["Region", "Total", "Entire home/apt", "Private room","Shared room"])
region_df_1


# In[22]:


df_draw_1 = pd.DataFrame({"Entire home/apt": region_df_1["Entire home/apt"].to_list(),
                        "Private room": region_df_1["Private room"].to_list(),
                        "Shared room": region_df_1["Shared room"].to_list()},
                       index=region_df_1["Region"].to_list())
df_draw_1.plot.barh(color=["deepskyblue", "mediumspringgreen", "orange"],
                  stacked=True,
                  figsize=(12, 5));


# In[131]:


manheten_df = df_norm1.loc[df_norm1["neighbourhood_group"] == "Manhattan"]
manheten_df_pr = manheten_df.loc[manheten_df["room_type"] == 1]
manheten_df_app = manheten_df.loc[manheten_df["room_type"] == 2]
manheten_df_sr = manheten_df.loc[manheten_df["room_type"] == 3]


# In[132]:


brooklyn_df = df_norm1.loc[df_norm1["neighbourhood_group"] == "Brooklyn"]
brooklyn_df_pr = brooklyn_df.loc[brooklyn_df["room_type"] == 1]
brooklyn_df_app = brooklyn_df.loc[brooklyn_df["room_type"] == 2]
brooklyn_df_sr = brooklyn_df.loc[brooklyn_df["room_type"] == 3]


# In[133]:


queens_df = df_norm1.loc[df_norm1["neighbourhood_group"] == "Queens"]
queens_df_pr = queens_df.loc[queens_df["room_type"] == 1]
queens_df_app = queens_df.loc[queens_df["room_type"] == 2]
queens_df_sr = queens_df.loc[queens_df["room_type"] == 3]


# In[134]:


bronx_df = df_norm1.loc[df_norm1["neighbourhood_group"] == "Bronx"]
bronx_df_pr = bronx_df.loc[bronx_df["room_type"] == 1]
bronx_df_app = bronx_df.loc[bronx_df["room_type"] == 2]
bronx_df_sr = bronx_df.loc[bronx_df["room_type"] == 3]


# In[135]:


staten_Island_df = df_norm1.loc[df_norm1["neighbourhood_group"] == "Staten Island"]
staten_Island_pr = staten_Island_df.loc[staten_Island_df["room_type"] == 1]
staten_Island_app = staten_Island_df.loc[staten_Island_df["room_type"] == 2]
staten_Island_sr = staten_Island_df.loc[staten_Island_df["room_type"] == 3]


# In[136]:


price_boro = pd.DataFrame({"Manhattan": [manheten_df_app["price"].mean(),manheten_df_pr["price"].mean(), manheten_df_sr["price"].mean()], 
                            "Brooklyn": [brooklyn_df_app["price"].mean(), brooklyn_df_pr["price"].mean(), brooklyn_df_sr["price"].mean()],
                            "Queens": [queens_df_app["price"].mean(), queens_df_pr["price"].mean(), queens_df_sr["price"].mean()],
                            "Bronx": [bronx_df_app["price"].mean(), bronx_df_pr["price"].mean(), bronx_df_sr["price"].mean()],
                            "Staten Island":[staten_Island_app["price"].mean(), staten_Island_pr["price"].mean(), staten_Island_sr["price"].mean()]
                          },
                             index=["Entire home/apt", "Private room", "Shared room"])
price_boro.plot.bar(rot=0);


# In[29]:


# зависимость кол-во отзывов от Боро
manheten_df_1 = df_norm1.loc[df_norm1["neighbourhood_group"] == "Manhattan"]
manheten_df_2 = manheten_df_1.loc[manheten_df_1["number_of_reviews"] != 0]
print(manheten_df_2["number_of_reviews"].mean())


# In[30]:


brooklyn_df_1 = df_norm1.loc[df_norm1["neighbourhood_group"] == "Brooklyn"]
brooklyn_df_2 = brooklyn_df_1.loc[brooklyn_df_1["number_of_reviews"] != 0]
print(brooklyn_df_2["number_of_reviews"].mean())

queens_df_1 = df_norm1.loc[df_norm1["neighbourhood_group"] == "Queens"]
queens_df_2 = queens_df_1.loc[queens_df_1["number_of_reviews"] != 0]
print(queens_df_2["number_of_reviews"].mean())

bronx_df_1 = df_norm1.loc[df_norm1["neighbourhood_group"] == "Bronx"]
bronx_df_2 = bronx_df_1.loc[bronx_df_1["number_of_reviews"] != 0]
print(bronx_df_2["number_of_reviews"].mean())

staten_Island_df_1 = df_norm1.loc[df_norm1["neighbourhood_group"] == "Staten Island"]
staten_Island_2 = staten_Island_df_1.loc[staten_Island_df_1["number_of_reviews"] != 0]
print(staten_Island_2["number_of_reviews"].mean())


# In[31]:


fig, ax = plt.subplots()
plt.barh(["Manhattan", "Brooklyn", "Queens", "Bronx","Staten Island"], [manheten_df_2["number_of_reviews"].mean(), brooklyn_df_2["number_of_reviews"].mean(), queens_df_2["number_of_reviews"].mean(), bronx_df_2["number_of_reviews"].mean(),staten_Island_2["number_of_reviews"].mean()])
ax.set_xlabel('Среднее количество отзывов')
ax.set_ylabel('Боро')


# In[32]:


# Самые популярные районы по количеству объявлений в каждом боро
manheten= df_norm1.loc[df_norm1["neighbourhood_group"] == "Manhattan"]
cout = [1,1,1,1,1]
name_region = [0,0,0,0,0]
for name in manheten["neighbourhood"].unique():
    if manheten["neighbourhood"].to_list().count(name) > cout[0]:
      cout[0] =  manheten["neighbourhood"].to_list().count(name)
      name_region[0] = name + " (Manhattan)"
print(cout[0], name_region[0])


# In[33]:


brooklyn = df_norm1.loc[df_norm1["neighbourhood_group"] == "Brooklyn"]
for name in brooklyn["neighbourhood"].unique():
    if brooklyn["neighbourhood"].to_list().count(name) > cout[1]:
      cout[1] =  brooklyn["neighbourhood"].to_list().count(name)
      name_region[1] = name + " (Brooklyn)"
print(cout[1], name_region[1])

queens = df_norm1.loc[df_norm1["neighbourhood_group"] == "Queens"]
for name in queens["neighbourhood"].unique():
    if queens["neighbourhood"].to_list().count(name) > cout[2]:
      cout[2] =  queens["neighbourhood"].to_list().count(name)
      name_region[2] = name + " (Queens)"
print(cout[2], name_region[2])

bronx = df_norm1.loc[df_norm1["neighbourhood_group"] == "Bronx"]
for name in bronx["neighbourhood"].unique():
    if bronx["neighbourhood"].to_list().count(name) > cout[3]:
      cout[3] =  bronx["neighbourhood"].to_list().count(name)
      name_region[3] = name + " (Bronx)" 
print(cout[3], name_region[3])

Staten_Island = df_norm1.loc[df_norm1["neighbourhood_group"] == "Staten Island"]
for name in Staten_Island["neighbourhood"].unique():
    if Staten_Island["neighbourhood"].to_list().count(name) > cout[4]:
      cout[4] =  Staten_Island["neighbourhood"].to_list().count(name)
      name_region[4] = name + " (Staten Island)"
print(cout[4], name_region[4])


# In[34]:


import numpy as np
color_rectangle = np.random.rand(7, 3)    # цвет
reg_df = pd.DataFrame({"Регион": name_region, "Количество объявлений":  cout})
reg_df.plot.barh(x="Регион",figsize=(12, 5), fontsize=12, color = color_rectangle)


# In[35]:


# наглядно узнаем доступность жилья
avalibity_data = {availability_365: df_norm1["availability_365"].to_list().count(availability_365) for availability_365 in set(df_norm1["availability_365"])}
print(avalibity_data)
del avalibity_data[0]
cases1 = []
regions1 = []
for region, cases_num in avalibity_data.items():
    regions1.append(region)
    cases1.append(cases_num)


# In[36]:


fig, ax = plt.subplots()
ax.plot(regions1, cases1)
ax.set_xlabel('Дней')
ax.set_ylabel('Количество объявлений')
fig.set_figwidth(12)
fig.set_figheight(5)


# In[37]:


# количество минимальных ночей
minimum_nights_data = {minimum_night: df_norm1["minimum_nights"].to_list().count(minimum_night) for minimum_night in set(df_norm1["minimum_nights"])}
print(minimum_nights_data)
cases1 = []
regions1 = []
for region, cases_num in minimum_nights_data.items():
    if region < 51:
        regions1.append(region)
        cases1.append(cases_num)


# In[38]:


fig, ax = plt.subplots()
ax.plot(regions1, cases1)
ax.set_xlabel('Минимальное количество ночей')
ax.set_ylabel('Количество объявлений')
fig.set_figwidth(10)
fig.set_figheight(8)


# In[39]:


couty = {calculated_host_listings_count: df_norm1["calculated_host_listings_count"].to_list().count(calculated_host_listings_count) for calculated_host_listings_count in set(df_norm1["calculated_host_listings_count"])}
print(couty)
cases1 = []
regions1 = []
for region, cases_num in couty.items():
        regions1.append(region)
        cases1.append(cases_num/region)


# In[40]:


fig, ax = plt.subplots()
ax.plot( regions1, cases1)
ax.set_ylabel('Количество арендодателей')
ax.set_xlabel('Количество объявлений у одного арендодателя')
fig.set_figwidth(16)
fig.set_figheight(6)


# In[41]:


# Гипотезы из 3-го дз.
# № 1 больше 80 процентов арендодателей у которых больше 1 объявления сдают квартиры в одном боро. 
dict_host_id = {} # Словарь, где ключ - id арендодателя, значение - список из боро в которых он сдает(по каждому объявлению)
for i, row in df_norm1.iterrows(): 
    dict_host_id[row['host_id']]= dict_host_id.get(row['host_id'], []) + [row['neighbourhood_group']]
print(dict_host_id)


# In[42]:


count_yes, count_no = 0, 0
print(len(dict_host_id))
for key, item in dict_host_id.items():
    if len(item) != 1:
        if len(set(item)) == 1:
            count_yes += 1
        else:
            count_no += 1
print(count_yes, count_no)


# In[43]:


pip install plotly


# In[44]:


import plotly
import plotly.graph_objs as go
fig = go.Figure(data=[go.Pie(labels=['в одном боро', 'в разных боро'], values=[count_yes, count_no], hole=.5)])
fig.show()


# In[46]:


# № 2 Больше 90 процентов арендодателей, у которых несколько объявлений, не имеют закрытых объявлений.   
dict_host_close = {} # Словарь, где ключ - id арендодателя, значение - список доступность каждого объявления 
for i, row in df_norm1.iterrows(): 
    dict_host_close[row['host_id']]= dict_host_close.get(row['host_id'], []) + [row['availability_365']]
print(dict_host_close)


# In[47]:


count_yes_2, count_no_2 = 0, 0
for key, item in dict_host_close.items():
    if len(item) != 1:
        if 0 in set(item):
            count_no_2 += 1
        else:
            count_yes_2 += 1
print(count_yes_2, count_no_2)


# In[48]:


fig = go.Figure(data=[go.Pie(labels=['Нет закрытых', 'Есть закрытые'], values=[count_yes_2, count_no_2])])
fig.show()


# In[ ]:


# № 3 Если убрать закрытые объявления, то распределение по популярности районов остается тем же. 
# Распределение по боро тоже остается - т.е. Манхэттен, Бруклин, Квинс, Бронкс, Статен Айленд. 


# In[115]:


avalibility_df = df_norm1.loc[df_norm1["availability_365"] != 0] # убираем закрытые объявления 
region_dt_1 = [(name, avalibility_df["neighbourhood"].to_list().count(name)) 
                  for name in avalibility_df["neighbourhood"].unique() if avalibility_df["neighbourhood"].to_list().count(name) > 800] 
region_dt_1 = sorted(region_dt_1, key=itemgetter(1))
regions_1 = []
counts_1 = []
for region_1, count_1 in region_dt_1 :
  regions_1.append(region_1)
  counts_1.append(count_1)


# In[118]:


fig = go.Figure(data=[
    go.Bar(name='Без закрытых объявлений', x=regions_1, y=counts_1, opacity=0.5),
    go.Bar(name='С закрытыми', x=regions, y=counts, opacity=0.5)
])
# Change the bar mode
fig.update_layout(barmode = 'overlay',
                 margin=dict(l=0, r=0, t=30, b=0),
                 xaxis_title="Район",
                yaxis_title="Количество объявлений",)
fig.show()


# In[120]:


boro_dt_1 = [(name, avalibility_df["neighbourhood_group"].to_list().count(name)) 
                  for name in avalibility_df["neighbourhood_group"].unique()] 
boro_dt_1
boro_dt_1 = sorted(boro_dt_1, key=itemgetter(1))
boro_dt = pd.DataFrame.from_records(boro_dt_1, 
                                         columns=["Region", "Total"])
boro_dt


# In[126]:


fig = go.Figure(data=[
    go.Bar(name='Без закрытых объявлений', x=region_df_1['Region'].to_list(), y=region_df_1['Total'].to_list(), opacity=0.5),
    go.Bar(name='С закрытыми', x=boro_dt['Region'].to_list(), y=boro_dt['Total'].to_list(), opacity=0.5)
])
# Change the bar mode
fig.update_layout(barmode = 'group',
                 margin=dict(l=0, r=0, t=30, b=0),
                 xaxis_title="Боро",
                yaxis_title="Количество объявлений",)
fig.show()


# In[130]:


# № 4 Закрытые объявления влияют на распределение по средней цене в районах и в боро
manheten_df_1 = avalibility_df.loc[avalibility_df["neighbourhood_group"] == "Manhattan"]
manheten_df_pr_1 = manheten_df.loc[manheten_df["room_type"] == 1]
manheten_df_app_1 = manheten_df.loc[manheten_df["room_type"] == 2]
manheten_df_sr_1 = manheten_df.loc[manheten_df["room_type"] == 3]

brooklyn_df_1 = avalibility_df.loc[avalibility_df["neighbourhood_group"] == "Brooklyn"]
brooklyn_df_pr_1 = brooklyn_df.loc[brooklyn_df["room_type"] == 1]
brooklyn_df_app_1 = brooklyn_df.loc[brooklyn_df["room_type"] == 2]
brooklyn_df_sr_1 = brooklyn_df.loc[brooklyn_df["room_type"] == 3]

queens_df_1 = avalibility_df.loc[avalibility_df["neighbourhood_group"] == "Queens"]
queens_df_pr_1 = queens_df.loc[queens_df["room_type"] == 1]
queens_df_app_1 = queens_df.loc[queens_df["room_type"] == 2]
queens_df_sr_1 = queens_df.loc[queens_df["room_type"] == 3]

bronx_df_1 = avalibility_df.loc[avalibility_df["neighbourhood_group"] == "Bronx"]
bronx_df_pr_1 = bronx_df.loc[bronx_df["room_type"] == 1]
bronx_df_app_1 = bronx_df.loc[bronx_df["room_type"] == 2]
bronx_df_sr_1 = bronx_df.loc[bronx_df["room_type"] == 3]

staten_Island_df_1 = avalibility_df.loc[avalibility_df["neighbourhood_group"] == "Staten Island"]
staten_Island_pr_1 = staten_Island_df.loc[staten_Island_df["room_type"] == 1]
staten_Island_app_1 = staten_Island_df.loc[staten_Island_df["room_type"] == 2]
staten_Island_sr_1 = staten_Island_df.loc[staten_Island_df["room_type"] == 3]


# In[156]:


fig = go.Figure(data=[
    go.Bar(name='Без закрытых объявлений', x=["Manhattan","Brooklyn","Queens","Bronx","Staten Island"], y=[manheten_df_app_1["price"].mean(),brooklyn_df_app_1["price"].mean(),queens_df_app_1["price"].mean(),bronx_df_app_1["price"].mean(),staten_Island_app_1["price"].mean()],marker_color='#330C73',
    opacity=0.75),
    go.Bar(name='С закрытыми', x=boro_dt['Region'].to_list(), y=[manheten_df_app["price"].mean(),brooklyn_df_app["price"].mean(),queens_df_app["price"].mean(),bronx_df_app["price"].mean(),staten_Island_app["price"].mean()],     marker_color='#EB89B5',
    opacity=0.75)
])
# Change the bar mode
fig.update_layout(barmode = 'group',
                 title_x = 0.5,
                 xaxis_title="Аппартаменты",
                yaxis_title="Средняя цена",
                 )
fig.show()


# In[157]:


fig = go.Figure(data=[
    go.Bar(name='Без закрытых объявлений', x=["Manhattan","Brooklyn","Queens","Bronx","Staten Island"], y=[manheten_df_pr_1["price"].mean(),brooklyn_df_pr_1["price"].mean(),queens_df_pr_1["price"].mean(),bronx_df_pr_1["price"].mean(),staten_Island_pr_1["price"].mean()],marker_color='#330C73',
    opacity=0.75),
    go.Bar(name='С закрытыми', x=boro_dt['Region'].to_list(), y=[manheten_df_pr["price"].mean(),brooklyn_df_pr["price"].mean(),queens_df_pr["price"].mean(),bronx_df_pr["price"].mean(),staten_Island_pr["price"].mean()],     marker_color='#EB89B5',
    opacity=0.75)
])
# Change the bar mode
fig.update_layout(barmode = 'group',
                 title_x = 0.5,
                 xaxis_title="Отдельная комната",
                yaxis_title="Средняя цена",
                 )
fig.show()


# In[158]:


fig = go.Figure(data=[
    go.Bar(name='Без закрытых объявлений', x=["Manhattan","Brooklyn","Queens","Bronx","Staten Island"], y=[manheten_df_sr_1["price"].mean(),brooklyn_df_sr_1["price"].mean(),queens_df_sr_1["price"].mean(),bronx_df_sr_1["price"].mean(),staten_Island_sr_1["price"].mean()],marker_color='#330C73',
    opacity=0.75),
    go.Bar(name='С закрытыми', x=boro_dt['Region'].to_list(), y=[manheten_df_sr["price"].mean(),brooklyn_df_sr["price"].mean(),queens_df_sr["price"].mean(),bronx_df_sr["price"].mean(),staten_Island_sr["price"].mean()],     marker_color='#EB89B5',
    opacity=0.75)
])
# Change the bar mode
fig.update_layout(barmode = 'group',
                 title_x = 0.5,
                 xaxis_title="Совместная комната",
                yaxis_title="Средняя цена",
                 )
fig.show()


# In[180]:


# № 5 чем меньше последняя дата оценки, тем меньше отзывов на объявлении. 
dict_host_date = {} # Словарь, где ключ - id арендодателя, значение - список из боро в которых он сдает(по каждому объявлению)
for i, row in df_norm1.iterrows(): 
    dict_host_date[row['last_review']]= dict_host_date.get(row['last_review'], []) + [row['number_of_reviews']]
print(dict_host_date)


# In[217]:


import statistics
date_1, meann = [],[]
list_keys = list(dict_host_date.keys())
list_keys.sort()
for key in list_keys:
    date_1.append(key)
    meann.append(statistics.median(dict_host_date[key]))    
del date_1[0], meann[0] # удалить дату 2001-01-01 которая означает что на объявлении нет даты последней оценки
print(len(date_1), len(meann))


# In[218]:


import plotly.express as px
df = pd.DataFrame({'Date': date_1, 'Mean': meann})


# In[219]:


fig = px.line(df, y="Mean", x="Date")
fig.show()


# In[220]:


from datetime import datetime, date
dict_host_date2 = {} # рассмотрим 2019 год
for i, row in df_norm1.iterrows(): 
    if row['last_review'] > date(2018, 12, 31):
        dict_host_date2[row['last_review']]= dict_host_date2.get(row['last_review'], []) + [row['number_of_reviews']]
print(dict_host_date2)

date_1, meann = [],[]
list_keys = list(dict_host_date2.keys())
list_keys.sort()
for key in list_keys:
    date_1.append(key)
    meann.append(statistics.median(dict_host_date2[key]))    
del date_1[0], meann[0] # удалить дату 2001-01-01 которая означает что на объявлении нет даты последней оценки
print(len(date_1), len(meann))


# In[221]:


df = pd.DataFrame({'Date': date_1, 'Mean': meann})
fig = px.line(df, y="Mean", x="Date")
fig.show()

