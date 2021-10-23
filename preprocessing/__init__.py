from .punctuation import toclean_text
from .noise import clean_tweet
from .stopwords import toremove_stopword

def preprocess_text(train_data):
    '''Text preprocessing'''

    # Removing punctuations
    train_data['clean_text'] = train_data['text'].apply(toclean_text)

    # Removing noise
    train_data["clean_text"] = train_data["clean_text"].apply(clean_tweet)

    # Removing stopwords
    train_data['clean_text'] = train_data['clean_text'].apply(toremove_stopword)

    return train_data