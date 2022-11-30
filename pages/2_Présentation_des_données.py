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

twelve = ['Janv', 'Fev', 'Mars', 'Avril', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Dec']
df_as = df[df['Location'] == 'AliceSprings']
df_ad = df[df['Location'] == 'Adelaide']
df_da = df[df['Location'] == 'Darwin']
df_sy = df[df['Location'] == 'Sydney']
df_co = df[df['Location'] == 'Cobar']

ad_ti = 'Adelaide - climat tempéré'
as_ti = 'Alice Springs - climat désertique'
da_ti = 'Darwin - climat tropical'
sy_ti = 'Sydney - climat subtropical'
co_ti = 'Cobar - climat aride'

st.header('Approche comparative')
st.markdown("5 climats étudiés")
    
with st.expander('Températures moyennes'):
    fig = plt.figure(figsize=(14,12))
    plt.subplot(321)
    ax1 = sns.barplot(data = df_ad.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax1 = sns.barplot(data = df_ad.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(ad_ti)
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(322)
    ax2 = sns.barplot(data = df_as.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax2 = sns.barplot(data = df_as.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(as_ti)
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(323)
    ax5 = sns.barplot(data = df_co.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax5 = sns.barplot(data = df_co.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(co_ti)
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()
    

    plt.subplot(324)
    ax4 = sns.barplot(data = df_sy.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax4 = sns.barplot(data = df_sy.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(sy_ti)
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()
    
    plt.subplot(325)
    ax3 = sns.barplot(data = df_da.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax3 = sns.barplot(data = df_da.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.title(da_ti)
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()
    st.pyplot(fig)

with st.expander('Moyenne quotidienne des précipitations'):
    fig = plt.figure(figsize=(14,12))
    plt.subplot(321)
    sns.barplot(data = df_ad.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(ad_ti)
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(322)
    sns.barplot(data = df_as.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(as_ti)
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(323)
    sns.barplot(data = df_co.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(co_ti)
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(324)
    sns.barplot(data = df_sy.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(sy_ti)
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')
    
    plt.subplot(325)
    sns.barplot(data = df_da.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.title(da_ti)
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')
    st.pyplot(fig)


with st.expander("Durée moyenne d'ensoleillement quotidien"):
    fig = plt.figure(figsize=(14,12))
    plt.subplot(321)
    sns.barplot(data = df_ad.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(ad_ti)

    plt.subplot(322)
    sns.barplot(data = df_as.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(as_ti)

    plt.subplot(323)
    sns.barplot(data = df_co.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(co_ti)

    plt.subplot(324)
    sns.barplot(data = df_sy.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(sy_ti)

    plt.subplot(325)
    sns.barplot(data = df_da.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement moyen quotidien (h)')
    plt.title(da_ti)
    st.pyplot(fig)

with st.expander('Moyenne des pressions quotidiennes'):
    fig = plt.figure(figsize=(14,12))
    plt.subplot(321)
    ax5 = sns.lineplot(data = df_ad.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax5 = sns.lineplot(data = df_ad.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(ad_ti)
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()


    plt.subplot(322)
    ax6 = sns.lineplot(data = df_as.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax6 = sns.lineplot(data = df_as.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(as_ti)
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()

    plt.subplot(323)
    ax7 = sns.lineplot(data = df_co.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax7 = sns.lineplot(data = df_co.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(co_ti)
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()

    plt.subplot(324)
    ax8 = sns.lineplot(data = df_sy.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax8 = sns.lineplot(data = df_sy.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(sy_ti)
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()

    plt.subplot(325)
    ax7 = sns.lineplot(data = df_da.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'blue', label = 'Pression 9:00')
    ax7 = sns.lineplot(data = df_da.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.title(da_ti)
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig)




tab1, tab2, tab3, tab4, tab5 = st.tabs(["Adelaide", "Alice Springs", "Cobar", "Darwin", "Sydney"])

with tab1:
    st.header('Adelaide : climat tempéré')
    st.image('images/aerial_view_Adelaide.jpg')
    fig1 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_ad.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_ad.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_ad.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_ad.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_ad.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_ad.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig1)

with tab2:
    st.header('Alice Springs : climat désertique')
    st.image('images/480px-Alice_Springs,_2015_(01).jpg')
    fig2 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_as.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_as.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_as.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_as.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_as.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_as.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig2)

with tab3:
    st.header('Cobar : climat de plaine aride')
    st.image('images/640px-Aerial_view_of_Cobar,New_South_Wales,_2009-03-06.jpg')
    fig2 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_co.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_co.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_co.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_co.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_co.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_co.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig2)

with tab4:
    st.header('Sydney : climat subtropical')
    st.image('images/vue sydney.webp')
    fig4 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_sy.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_sy.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_sy.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_sy.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_sy.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_sy.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig4)
    

with tab5:
    st.header('Darwin : climat tropical')
    st.image('images/Darwin.jpg')
    fig3 = plt.figure(figsize = (10,8))
    plt.subplot(221)
    ax = sns.barplot(data = df_da.groupby('month').mean('MaxTemp'), x = twelve, y = 'MaxTemp', color = 'orange', label = "Maximales")
    ax = sns.barplot(data = df_da.groupby('month').mean('MinTemp'), x = twelve, y = 'MinTemp', color = 'lightblue', label = "Minimales")
    plt.ylabel('Températures moyennes (°C)')
    plt.legend()

    plt.subplot(222)
    sns.barplot(data = df_da.groupby('month').mean('Rainfall'), x = twelve, y = 'Rainfall', color = 'b')
    plt.ylabel('Moyenne quotidienne des précipitations (mm)')

    plt.subplot(223)
    sns.barplot(data = df_da.groupby('month').mean('Sunshine'), x = twelve, y = 'Sunshine', color = 'gold')
    plt.ylabel('Ensoleillement quotidien moyen (h)')

    plt.subplot(224)
    ax0 = sns.lineplot(data = df_da.groupby('month').mean('Pressure9am'), x = twelve, y = 'Pressure9am', color = 'b', label = 'Pression 9:00')
    ax0 = sns.lineplot(data = df_da.groupby('month').mean('Pressure3pm'), x = twelve, y = 'Pressure3pm', color = 'r', label = 'Pression 15:00')
    plt.ylabel('Pression quotidienne moyenne')
    plt.legend()
    st.pyplot(fig3)