import streamlit as st
import pandas as pd
import prophet
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/verdun_MAJ.csv' )
 
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
 

from pandas.api.types import CategoricalDtype

cat_type = CategoricalDtype(categories=['Monday','Tuesday',
                                        'Wednesday',
                                        'Thursday','Friday',
                                        'Saturday','Sunday'],
                            ordered=True)

def create_features_saison(df, label=None):
  
    
    df = df.copy()
    
     
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['weekday'] = df['ds'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week
    df['date_offset'] = (df.ds.dt.month*100 + df.ds.dt.day - 320)%1300
    df['date'] = df.index
    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], 
                          labels=['Printemps' ,'Été', 'Automne' ,'Hiver']
                   )
    X = df[[ 'dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear','weekday',
           'season']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features_saison(df,label='y')
features_and_target = pd.concat([X, y], axis=1)
 

st.title("A Simple Streamlit Web App")


dat = st.text_input("Faire la prédiction à partir de la date : ", '') 
d=predict(dat)
st.dataframe(d['yhat'])
