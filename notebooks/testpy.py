import streamlit as st
import pandas as pd
import prophet
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

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


def create_features(df, label=None):
     
    df = df.copy()
    df['datetime'] = df['ds']
    
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week
    df['ds'] = df.index
    
    X = df[['datetime','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(df, label='y')


features_and_target = pd.concat([X, y], axis=1)


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
 

def pilot() :
  sns.pairplot(features_and_target_sCOV.dropna(),
             hue='year',  palette='hls',
             x_vars=['dayofweek',
                     'year','weekofyear'],
             y_vars='y',
             height=5,
             plot_kws={'alpha':0.15, 'linewidth':0}  
            )
  plt.suptitle('Nombre entrée parking par jour , année et semaine ')
 
  return plt
def pilot3(): 
  fig, ax = plt.subplots(figsize=(10, 5))
  sns.boxplot(data=features_and_target.dropna(),
            x='weekday',
            y='y',
            hue='season',
            ax=ax,
            linewidth=1)
  ax.set_title('Nombre Entrée Parking par saison  ')
  ax.set_xlabel('jour de la semaine')
  ax.set_ylabel('Nombre Entrée Parking')
  ax.legend(bbox_to_anchor=(1, 1))
  return plt 

 
 
 
 
 
 
 
 
 
 
st.title("A Simple Streamlit Web App")


dat = st.text_input("Faire la prédiction à partir de la date : ", '') 
d=predict(dat)


fig1 = pilot() 

st.pyplot(fig1)
fig = pilot3() 

st.pyplot(fig)



st.dataframe(d['yhat'])
