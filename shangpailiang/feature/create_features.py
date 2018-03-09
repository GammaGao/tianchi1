# coding: utf-8
# In[1]:
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# TRAIN
parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('trainb.csv', parse_dates=['date_map'], date_parser=parser)
df['year'] = df.date_map.dt.year
df['month'] = df.date_map.dt.month
df['day'] = df.date_map.dt.day
df['day_of_week'] = df['day_of_week'] - 1
df['brandid'] = df['brand']
df['brand'] = df['brand'] - 1
df = df[['brand', 'date_type', 'day_of_week', 'year', 'month', 'week_of_year', 'cnt', 'date', 'date_map', 'day',
         'day_of_year', 'brandid']]

df.to_csv('train_tmp.csv', index=False)

t = pd.read_csv('train_tmp.csv')
t = t[['brand', 'date_type', 'day_of_week', 'year', 'month', 'week_of_year', 'cnt', 'day', 'day_of_year', 'date',
       'brandid']]

oenc = OneHotEncoder(categorical_features=np.array([0, 1, 2]), n_values=[10, 5, 7])
oenc.fit(t)
t1 = oenc.transform(t)
t1df = pd.DataFrame(t1.toarray())
t1df.to_csv('train_ohe.csv',
            header=['brand1', 'brand2', 'brand3', 'brand4', 'brand5', 'brand6', 'brand7', 'brand8', 'brand9', 'brand10',
                    'date_type1', 'date_type2', 'date_type3', 'date_type4', 'date_type5', 'day_of_week1',
                    'day_of_week2', 'day_of_week3', 'day_of_week4', 'day_of_week5', 'day_of_week6', 'day_of_week7'
                , 'year', 'month', 'week_of_year', 'cnt', 'day', 'day_of_year', 'date', 'brandid'], index=False)

# TESTA
parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('testa.csv', parse_dates=['date_map'], date_parser=parser)
df['year'] = df.date_map.dt.year
df['month'] = df.date_map.dt.month
df['day'] = df.date_map.dt.day
df['day_of_week'] = df['day_of_week'] - 1
df['brandid'] = df['brand']
df['brand'] = df['brand'] - 1
df = df[['brand', 'date_type', 'day_of_week', 'year', 'month', 'week_of_year', 'date', 'date_map', 'brandid', 'day',
         'day_of_year', ]]

df.to_csv('testa_tmp.csv', index=False)

t = pd.read_csv('testa_tmp.csv')
t = t[['brand', 'date_type', 'day_of_week', 'year', 'month', 'week_of_year', 'day', 'day_of_year', 'date', 'brandid']]

oenc = OneHotEncoder(categorical_features=np.array([0, 1, 2]), n_values=[10, 5, 7])
oenc.fit(t)
t1 = oenc.transform(t)
t1df = pd.DataFrame(t1.toarray())
t1df.to_csv('testa_ohe.csv',
            header=['brand1', 'brand2', 'brand3', 'brand4', 'brand5', 'brand6', 'brand7', 'brand8', 'brand9', 'brand10',
                    'date_type1', 'date_type2', 'date_type3', 'date_type4', 'date_type5', 'day_of_week1',
                    'day_of_week2', 'day_of_week3', 'day_of_week4', 'day_of_week5', 'day_of_week6', 'day_of_week7'
                , 'year', 'month', 'week_of_year', 'day', 'day_of_year', 'date', 'brandid'], index=False)
# TESTB
parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('testb.csv', parse_dates=['date_map'], date_parser=parser)
df['year'] = df.date_map.dt.year
df['month'] = df.date_map.dt.month
df['day'] = df.date_map.dt.day
df['day_of_week'] = df['day_of_week'] - 1
df['brandid'] = df['brand']
df['brand'] = df['brand'] - 1
df = df[['brand', 'date_type', 'day_of_week', 'year', 'month', 'week_of_year', 'date', 'date_map', 'brandid', 'day',
         'day_of_year', ]]

df.to_csv('testb_tmp.csv', index=False)

t = pd.read_csv('testb_tmp.csv')
t = t[['brand', 'date_type', 'day_of_week', 'year', 'month', 'week_of_year', 'day', 'day_of_year', 'date', 'brandid']]

oenc = OneHotEncoder(categorical_features=np.array([0, 1, 2]), n_values=[10, 5, 7])
oenc.fit(t)
t1 = oenc.transform(t)
t1df = pd.DataFrame(t1.toarray())
t1df.to_csv('testb_ohe.csv',
            header=['brand1', 'brand2', 'brand3', 'brand4', 'brand5', 'brand6', 'brand7', 'brand8', 'brand9', 'brand10',
                    'date_type1', 'date_type2', 'date_type3', 'date_type4', 'date_type5', 'day_of_week1',
                    'day_of_week2', 'day_of_week3', 'day_of_week4', 'day_of_week5', 'day_of_week6', 'day_of_week7'
                , 'year', 'month', 'week_of_year', 'day', 'day_of_year', 'date', 'brandid'], index=False)
