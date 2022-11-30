import pandas as pd 
import seaborn as sns 
import geopandas as gpd
import streamlit as st 
import numpy as np
import joblib
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib as mpl
from matplotlib.patches import Rectangle 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import bottleneck as bn
import time

sns.set_style('darkgrid')

# dataset_path = 'C:/Users/theop/Documents/000AAADATASCIENTIST/weather-forecast-australia-streamlit-main/australie.csv'
dataset_path = 'australie.csv'
df = pd.read_csv(dataset_path, parse_dates=['Date'])
df['month'] = pd.to_datetime(df['Date']).dt.month
st.title('Présentation des données')
df['RainToday'].replace({'No': 0}, inplace=True)
df['RainToday'].replace({'Yes': 1}, inplace=True)
df['RainTomorrow'].replace({'No': 0}, inplace=True)
df['RainTomorrow'].replace({'Yes': 1}, inplace=True)
# df.dropna(inplace=True)
twelve = []
for i in range(1,13):
    twelve.append(i)
df_as = df[df['Location'] == 'AliceSprings']
df_ad = df[df['Location'] == 'Adelaide']
df_da = df[df['Location'] == 'Darwin']
df_sy = df[df['Location'] == 'Sydney']

ad_ti = 'Adelaide - climat tempéré'
as_ti = 'Alice Springs - climat désertique'
da_ti = 'Darwin - climat tropical'
sy_ti = 'Sydney - climat subtropical'

st.header('Approche comparative')
st.markdown("4 climats étudiés")
    
with st.expander('Températures moyennes'):
    fig = plt.figure(figsize=(14,10))
    plt.subplot(221)
    ax1 = sns.barplot(data = df_ad.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax1 = sns.barplot(data = df_ad.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(ad_ti)
    plt.xlabel('Mois')
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    ax2 = sns.barplot(data = df_as.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax2 = sns.barplot(data = df_as.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(as_ti)
    plt.xlabel('Mois')
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(223)
    ax3 = sns.barplot(data = df_da.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax3 = sns.barplot(data = df_da.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(da_ti)
    plt.xlabel('Mois')
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(224)
    ax4 = sns.barplot(data = df_sy.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax4 = sns.barplot(data = df_sy.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(sy_ti)
    plt.xlabel('Mois')
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()
    st.pyplot(fig)

with st.expander('Moyenne quotidienne des précipitations'):
    fig = plt.figure(figsize=(14,10))
    plt.subplot(221)
    sns.barplot(data = df_ad.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(ad_ti)
    plt.xlabel('Mois')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(222)
    sns.barplot(data = df_as.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(as_ti)
    plt.xlabel('Mois')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_da.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(da_ti)
    plt.xlabel('Mois')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(224)
    sns.barplot(data = df_sy.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(sy_ti)
    plt.xlabel('Mois')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')
    st.pyplot(fig)

with st.expander("Durée moyenne d'ensoleillement quotidien"):
    fig = plt.figure(figsize=(14,10))
    plt.subplot(221)
    sns.barplot(data = df_ad.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.xlabel('Mois')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(ad_ti)

    plt.subplot(222)
    sns.barplot(data = df_as.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.xlabel('Mois')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(as_ti)

    plt.subplot(223)
    sns.barplot(data = df_da.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.xlabel('Mois')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(da_ti)

    plt.subplot(224)
    sns.barplot(data = df_sy.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.xlabel('Mois')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(sy_ti)
    st.pyplot(fig)

with st.expander('Moyenne des pressions quotidiennes'):
    fig = plt.figure(figsize=(14,10))
    plt.subplot(221)
    ax5 = sns.lineplot(data = df_ad.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax5 = sns.lineplot(data = df_ad.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(ad_ti)
    plt.xlabel('Mois')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()


    plt.subplot(222)
    ax6 = sns.lineplot(data = df_as.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax6 = sns.lineplot(data = df_as.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(as_ti)
    plt.xlabel('Mois')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()

    plt.subplot(223)
    ax7 = sns.lineplot(data = df_da.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax7 = sns.lineplot(data = df_da.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(da_ti)
    plt.xlabel('Mois')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()

    plt.subplot(224)
    ax8 = sns.lineplot(data = df_sy.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax8 = sns.lineplot(data = df_sy.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(sy_ti)
    plt.xlabel('Mois')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig)




tab1, tab2, tab3, tab4 = st.tabs(["Adelaide", "Alice Springs", "Darwin", "Sydney"])

with tab1:
    st.header('Adelaide : climat tempéré')
    fig1 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_ad.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_ad.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.xlabel('Mois')
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_ad.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.xlabel('Mois')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_ad.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.xlabel('Mois')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_ad.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_ad.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.xlabel('Mois')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig1)

with tab2:
    st.header('Alice Springs : climat désertique')
    fig2 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_as.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_as.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.xlabel('Mois')
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_as.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.xlabel('Mois')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_as.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.xlabel('Mois')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_as.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_as.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.xlabel('Mois')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig2)

with tab3:
    st.header('Darwin : climat tropical')
    fig3 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_da.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_da.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.xlabel('Mois')
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_da.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.xlabel('Mois')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_da.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.xlabel('Mois')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_da.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_da.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.xlabel('Mois')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig3)

with tab4:
    st.header('Sydney : climat subtropical')
    fig4 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_sy.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_sy.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.xlabel('Mois')
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_sy.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.xlabel('Mois')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_sy.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.xlabel('Mois')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_sy.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_sy.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.xlabel('Mois')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig4)
    