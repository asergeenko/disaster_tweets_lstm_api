FROM python:3.6-slim

RUN apt-get update && apt-get install -y unixodbc-dev gcc g++
RUN apt-get install -y git

CMD mkdir app
WORKDIR app

RUN git clone https://github.com/asergeenko/disaster_tweets_lstm_api.git /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords

RUN pip install h5py==2.9.0

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "deploy/api.py"]