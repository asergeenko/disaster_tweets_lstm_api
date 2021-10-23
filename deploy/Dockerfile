FROM python:3.6-slim

RUN apt-get update && apt-get install -y unixodbc-dev gcc g++
RUN apt-get install -y git

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords

CMD mkdir app
WORKDIR app

RUN git clone https://github.com/asergeenko/disaster_tweets_lstm_api.git /app

CMD ["python", "disaster_tweets_lstm_api/deploy/api.py"]