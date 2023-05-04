# -*- coding: utf-8 -*-
"""deploy-streamlit-parking.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/CyJERIDI/0bad283ee60619f31f551ad4bd1ea5bf/deploy-streamlit-parking.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %pip install prophet
import prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv('/content/verdun_MAJ.csv' )
 
df['y'] = pd.array(df.y, dtype=pd.Int64Dtype())
df['ds'] = pd.to_datetime(df.ds, format='%Y-%m-%d', errors='coerce')

df1 = df.loc[df.ds <"2020-01-01"].copy()
df2 =df.loc[df.ds> "2020-12-31"].copy()
frames = [df1, df2]
split_date = '2023-04-15'  
df = pd.concat(frames)
df_train = df.loc[df.ds <=split_date].copy()
df_test =df.loc[df.ds> split_date].copy()

from prophet import Prophet


def predict(date_future):
 df_train_prophet = df_train.reset_index()  

 

 model = Prophet ( )
 model.add_country_holidays(country_name='FR')

 model.fit(df_train_prophet)
 future_date = pd.date_range(date_future , periods=30, freq='D')
 future_date = pd.DataFrame({'ds': future_date })
 pred = model.predict(future_date )
 return pred
 predict('2023-05-01')