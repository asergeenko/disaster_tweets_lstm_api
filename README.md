NLP with Disaster Tweets using LSTM
==============================
Source notebook: [NLP with Disaster Tweets using LSTM](https://www.kaggle.com/sandhyakrishnan02/nlp-with-disaster-tweets-using-lstm)

Kaggle competition: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)


## Deployment using Docker
~~~
cd deploy
docker build -t disaster_tweets_lstm_api .
docker run -d -p 8000:8000 disaster_tweets_lstm_api
~~~

## Local installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
## Usage:

### Console run

#### Training
~~~
python main.py --train_file="data/raw/train.csv"
~~~

#### Prediction
~~~
python main.py --predict_message="Damage to school bus on 80 in multi car crash"
~~~

### REST API run
~~~
python deploy/api.py
~~~

#### Endpoints

- *GET /predict/{message}* - predict disaster for message
- *POST /train/* - train model with given CSV file


## Test:
~~~
python -m pytest tests/
~~~

Project structure is based on [ML Project Example](https://github.com/Mikhail-M/ml_project_example).
