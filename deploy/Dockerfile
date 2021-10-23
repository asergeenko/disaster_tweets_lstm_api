FROM python:3.6-slim

RUN apt-get update && apt-get install -y unixodbc-dev gcc g++
RUN apt-get install git

RUN pip install \
    pytest==6.2.5 \
    uvicorn==0.13.4 \
    fastapi==0.68.1 \
    pandas==1.1.5 \
    scikit-learn==0.21.2 \
    tensorflow==1.9.0 \
    nltk==3.2.1 \

RUN python -m nltk.downloader stopwords

CMD mkdir app
WORKDIR app

RUN git clone https://github.com/asergeenko/disaster_tweets_lstm_api.git /app

CMD ["python", "disaster_tweets_lstm_api/deploy/api.py"]