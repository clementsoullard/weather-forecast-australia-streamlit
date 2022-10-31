#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:07:36 2022

@author: yohancohen
"""

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



def demo_australie():

    ### Create Title
    st.title("Etude d'un prédicteur de pluie")
    st.header("Sur le continent autralien")
    st.write("Clément Soullard, Stanley, Moulaye, Théo Porcher")

    ### Add a picture
    st.image("rain.jpg",width=400)

    model=joblib.load('simplified_model.sav')
    ### using Markdown
    st.markdown("### Prédiction de la pluie à partir des données de la veille")

    ### reading dataset
    # Normally, you will store all the necessary path and env variables in a .env file
    dataset_path = 'australie.csv'
    locationsCoords={'Albury': (-36.083333333333336, 146.95),
 'BadgerysCreek': (-33.87972222222222, 150.75222222222223),
  'Cobar': (-31.483333333333334, 145.8), 'CoffsHarbour': (-30.3, 153.11666666666667), 'Moree': (-29.466666666666665, 149.83333333333334), 'Newcastle': (-32.93333333333333, 151.73333333333332), 'NorahHead': (-33.2825, 151.57416666666666), 'NorfolkIsland': (-29.03, 167.95), 'Penrith': (-33.75, 150.71666666666667), 'Richmond': (-37.823, 144.998), 'Sydney': (-33.85611111111111, 151.1925), 'SydneyAirport': (-33.925900320771255, 151.18998051518105), 'WaggaWagga': (-35.13, 147.3536111111111), 'Williamtown': (-32.815, 151.8427777777778), 'Wollongong': (-34.42722222222222, 150.89388888888888), 'Canberra': (-35.293055555555554, 149.12694444444446), 'Tuggeranong': (-35.4244, 149.0888), 'MountGinini': (-35.53333333333333, 148.78333333333333), 'Ballarat': (-37.56083333333333, 143.8475), 'Bendigo': (-36.75, 144.26666666666668), 'Sale': (-38.1, 147.06666666666666), 'MelbourneAirport': (-37.81666666666667, 144.96666666666667), 'Melbourne': (-37.81666666666667, 144.96666666666667), 'Mildura': (-34.18888888888888, 142.15833333333333), 'Nhil': (-36.333333333333336, 141.65), 'Portland': (-38.333333333333336, 141.6), 'Watsonia': (-37.708, 145.083), 'Dartmoor': (-37.93333333333333, 141.28333333333333), 'Brisbane': (-27.467777777777776, 153.02805555555557), 'Cairns': (-16.92, 145.78), 'GoldCoast': (-28.02583333333333, 153.38972222222222), 'Townsville': (-19.25, 146.8), 'Adelaide': (-34.93, 138.59972222222223), 'MountGambier': (-37.82944444444445, 140.7827777777778), 'Nuriootpa': (-34.46666666666667, 138.98333333333332), 'Woomera': (-31.2, 136.81666666666666), 'Albany': (-35.016666666666666, 117.88333333333334), 'Witchcliffe': (-34.03, 115.1), 'PearceRAAF': (-31.66777777777778, 116.015), 'PerthAirport': (-31.942222222222224, 115.95583333333333), 'Perth': (-31.95, 115.86666666666666), 'SalmonGums': (-32.98, 121.645), 'Walpole': (-34.98, 116.7), 'Hobart': (-42.88583333333333, 147.33138888888888), 'Launceston': (-41.43333333333333, 147.13333333333333), 'AliceSprings': (-23.702222222222222, 133.87666666666667), 'Darwin': (-12.436111111111112, 130.84111111111113), 'Katherine': (-14.466666666666667, 132.26666666666668), 'Uluru': (-25.345, 131.0363888888889)}
  
    dict={'E':90, 'ENE':68, 'ESE':112, 'N':0, 'NE':45, 'NNE':22, 'NNW':-22, 'NW':-45, 'S':180, 'SE':135, 'SSE':158,
       'SSW':-158, 'SW':-135, 'W':-90, 'WNW':-68, 'WSW':-112}


    ### Showing code
    #st.text("importing dataset with the folowing command: ")
    #with st.echo(): 
    #    df = pd.read_csv(dataset_path, parse_dates=['Date'])
    #    df['month'] = pd.to_datetime(df['Date']).dt.month
    df = pd.read_csv(dataset_path, parse_dates=['Date'])
    df['month'] = pd.to_datetime(df['Date']).dt.month
    
    df['RainToday'].replace({'No': 0}, inplace=True)
    df['RainToday'].replace({'Yes': 1}, inplace=True)
    df['RainTomorrow'].replace({'No': 0}, inplace=True)
    df['RainTomorrow'].replace({'Yes': 1}, inplace=True)



    ### Showing the data
    #if st.checkbox("Showing the data") :
    #     st.dataframe(df.head())
    locations=list(df.Location.unique())
    locations.sort()
    
    #st.dataframe(locations)
    location=st.sidebar.selectbox("Location",locations)
    
    stationsmeteos=pd.DataFrame.from_dict(locationsCoords, orient='index')
      
    #st.write(stationsmeteos)
    gdf = gpd.GeoDataFrame(
    stationsmeteos, geometry=gpd.points_from_xy(stationsmeteos[1], stationsmeteos[0]))
    # gdflocation = gpd.GeoDataFrame(
    # stationsmeteos, geometry=gpd.points_from_xy(stationsmeteos[1], stationsmeteos[0]))

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdfcity=gdf.loc[location]
    #st.write(gdfcity)

    # We restrict to Australia. 
    st.markdown('## Analyse par localité')
    st.markdown('Observez les indicateurs clé en sélectionnant la localité à droite')
    
    fig, ((ax1,ax),(ax2,ax3)) = plt.subplots(2,2)
    world[world.name == 'Australia'].plot(
    color='white', edgecolor='black',figsize=(18,12),ax=ax1)
    ax1.set_xlim(110,160)
    ax1.set_ylim(-40,-10)
    ax1.set_title('Localisation')
       
    gdf.plot(ax=ax1, color='red')
    ax1.scatter(gdfcity[1],gdfcity[0], color='blue',s=60)   

    barplot = sns.barplot(data=df[df.Location==location],ax=ax, x="month", y="Rainfall",color='b')
    barplot.set_ylim(bottom=0, top=20);
    ax.set_title('Pluviométrie (mm/jour)')
    
    #st.pyplot(fig1)

    #st.markdown('## Diagramme de température')
    
    
    temperature=df[df['Location']==location][['Date','MaxTemp','MinTemp']]
    temperature["rollavgmax"]=bn.move_mean(temperature.MaxTemp,30)
    temperature["rollavgmin"]=bn.move_mean(temperature.MinTemp,30)
    temperature['dayofyear']=temperature['Date'].apply(lambda x: x.dayofyear)
    temperatureYear=temperature.groupby('dayofyear')['rollavgmax','rollavgmin'].agg('mean')
    ax2.plot(temperatureYear.rollavgmax,label="Maxima",color='r')
    ax2.plot(temperatureYear.rollavgmin,label="Minima",color='b')
    ax2.set_title('Températures annuelles')
 
    dffiltered=df.loc[df['Location'] == location]
    winds=pd.get_dummies(dffiltered['WindGustDir'])
    winds['RainTomorrow']=dffiltered['RainTomorrow']
    corr=winds.corr()
    bringrain=pd.DataFrame(corr.iloc[:-1,-1])
    bringrain.rename(columns={'RainTomorrow':'WindContr'},inplace=True)
    
    for dirvent,w in bringrain.iterrows():
        force=w.WindContr
        if (force > 0):
            c='b'
        else:
            c='r'
            force=-force
        rectangle=Rectangle((-.05, .25),.1,3*force, color=c)
        t2 = mpl.transforms.Affine2D().rotate_deg(dict[dirvent]) + ax3.transData
        rectangle.set_transform(t2)
        ax3.add_patch(rectangle)
    ax3.set_xlim([-1.5,1.5])
    ax3.set_ylim([-1,1])
    ax3.set_title("Corrélation pluie/vent")

    fig.tight_layout()
    st.pyplot(fig)



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
           inputvalue=np.array([Humidity3pm_val,Sunshine_val,varhumidity_val,Pressure9am_val,Rainfall_val,WindGustSpeed_val,Pressure3p_val]).reshape(1,-1)
           out=any(model.predict(inputvalue)>0)
           st.write("Pleuvra-t-il demain ?",( "Oui"if out else "Non"))

  

