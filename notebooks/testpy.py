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
d=predict('2023-05-01')
st.dataframe(d['yhat'])


st.title("A Simple Streamlit Web App")
name = st.text_input("Enter your name", '')
st.write(f"Hello {name}!")
x = st.slider("Select an integer x", 0, 10, 1)
y = st.slider("Select an integer y", 0, 10, 1)
df = pd.DataFrame({"x": [x], "y": [y] , "x + y": [x + y]}, index = ["addition row"])
st.write(df)
