import json
import logging

import pandas as pd
import os
import pickle
import logging

#Library for Splitting Dataset
from sklearn.model_selection import train_test_split

#Libraries for NN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout

from preprocessing import preprocess_text


with open('config/config.json') as fp:
    config = json.load(fp)

logging.basicConfig(filename=config.get('LOG_FILE',None), level=config.get('LOG_LEVEL','DEBUG'), datefmt='%Y.%m.%d %H:%M:%S')

def tokenize(input_text, max_features=3000, seq_len=None):
    '''Tokenize text'''
    if seq_len:
        with open(config["SAVED_TOKENIZER_PATH"], 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(input_text)
        with open(config["SAVED_TOKENIZER_PATH"], 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X = tokenizer.texts_to_sequences(input_text)
    X = pad_sequences(X, seq_len) if seq_len else pad_sequences(X)
    return pad_sequences(X)

def train(X_train,X_test,y_train,y_test, max_features=3000):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=32, input_length=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.4))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.002)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save(config['SAVED_MODEL_PATH'])
    return model

def train_pipeline(input_filepath):
    train_data = pd.read_csv(input_filepath)
    train_data = preprocess_text(train_data)

    X = tokenize(train_data['clean_text'].values, config['MAX_FEATURES'])

    # Model creation using LSTM
    y = train_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =41)

    model = train(X_train,X_test,y_train,y_test, config['MAX_FEATURES'])

    return model

def get_model():
    '''Load or create model'''
    if os.path.exists(config['SAVED_MODEL_PATH']) and os.path.isfile(config['SAVED_MODEL_PATH']):
        try:
            model = load_model(config['SAVED_MODEL_PATH'])
        except ValueError as e:
            logging.exception('Cannot load model. Trying to train again...')
            model = train_pipeline(config['RAW_DATA_PATH'])
    else:
        model = train_pipeline(config['RAW_DATA_PATH'])
    return model

def predict(message, model):
    '''Predict result for "message" using "model"'''
    df = pd.DataFrame({'text': [message]})
    df = preprocess_text(df)
    X = tokenize(df['clean_text'].values, config['MAX_FEATURES'], model.input_shape[1])
    y_pred = model.predict(X).round()
    return int(y_pred[0][0])