FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install --upgrade pip
RUN pip3 install geopandas
RUN pip3 install seaborn
RUN pip3 install sklearn
RUN pip3 install altair
RUN pip3 install pandas
RUN pip3 install streamlit
RUN pip3 install bottleneck
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
