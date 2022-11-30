# Core Pkg
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
# from streamlit_autorefresh import st_autorefresh

st.title("Etude d'un prédicteur de pluie")
st.header("Sur le continent autralien")
st.write("Clément Soullard, Stanley Armel, Hamza Moulaye, Théo Porcher")

### Add a picture
st.image("rain.jpg",width=400)
model=joblib.load(".\simplified_model.sav")