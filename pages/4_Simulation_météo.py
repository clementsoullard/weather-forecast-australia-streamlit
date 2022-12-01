import pandas as pd 
import seaborn as sns 
import geopandas as gpd
import streamlit as st 
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import bottleneck as bn
import time

model=joblib.load("simplified_model.sav")
### using Markdown
st.markdown("### Prédiction de la pluie à partir des données de la veille")

#



if st.checkbox("Missing values") : 
    st.dataframe(df.isna().sum())


### Preprocessing

with st.form("my_form"):
    st.write("Quelle météo fera-t-il demain ?")
    Humidity3pm_val = st.slider("Humidity3pm")
    Sunshine_val = st.slider("Sunshine",min_value=0.0,max_value=14.0,step=.2)
    varhumidity_val = st.slider("varhumidity",min_value=-100,max_value=100)
    Pressure9am_val = st.slider("Pressure9am",min_value=970,max_value=1050)
    Rainfall_val = st.slider("Rainfall",min_value=0,max_value=200)
    WindGustSpeed_val = st.slider("WindGustSpeed",min_value=0,max_value=140)
    Pressure3p_val = st.slider("Pressure3p",min_value=970,max_value=1050)
    submitted = st.form_submit_button("Submit")
    if submitted:
        with st.spinner('Simulation...'):
            time.sleep(3)
            st.success('Simulation effectuée')
        inputvalue=np.array([Humidity3pm_val,Sunshine_val,varhumidity_val,Pressure9am_val,Rainfall_val,WindGustSpeed_val,Pressure3p_val]).reshape(1,-1)
        out=any(model.predict(inputvalue)>0)
        st.write("Pleuvra-t-il demain ?",( "Oui"if out else "Non"))
        if out == True:
            st.image("images/rainy.gif")
        else :
            st.image("images/sunny.gif")
