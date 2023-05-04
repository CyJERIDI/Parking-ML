import streamlit as st
import pandas as pd
import prophet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('https://github.com/CyJERIDI/Parking-ML/blob/codespace-cyjeridi-literate-computing-machine-gg66r7x9gvjhv5jj/data/verdun_MAJ.csv' )
 
df['y'] = pd.array(df.y, dtype=pd.Int64Dtype())
df['ds'] = pd.to_datetime(df.ds, format='%Y-%m-%d', errors='coerce')

df1 = df.loc[df.ds <"2020-01-01"].copy()
df2 =df.loc[df.ds> "2020-12-31"].copy()
frames = [df1, df2]
split_date = '2023-04-15'  
df = pd.concat(frames)
df_train = df.loc[df.ds <=split_date].copy()
df_test =df.loc[df.ds> split_date].copy()



st.title("A Simple Streamlit Web App")
name = st.text_input("Enter your name", '')
st.write(f"Hello {name}!")
x = st.slider("Select an integer x", 0, 10, 1)
y = st.slider("Select an integer y", 0, 10, 1)
df = pd.DataFrame({"x": [x], "y": [y] , "x + y": [x + y]}, index = ["addition row"])
st.write(df)
