FROM aminehy/docker-streamlit-app:latest
RUN pip install geopandas
RUN pip install seaborn
RUN pip install sklearn
