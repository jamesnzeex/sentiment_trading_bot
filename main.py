import pickle
import os
import NLPmodel
import webscraper

import pandas as pd
import numpy as np
import tensorflow as tf
# import alpaca_trade_api as tradeapi

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

ticker = "TSLA"

oov_tok = '<OOV>'
trunc_type = 'post'
padding_type='post'
vocab_size =1000
max_length = 150

if os.path.exists('./tokenizer.pickle'):
    with open('./tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    NLPmodel.train_model()
    with open('./tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

if os.path.exists('./trained_model'):
    model = tf.keras.models.load_model('./trained_model')
else:
    NLPmodel.train_model()
    model = tf.keras.models.load_model('./trained_model')

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type,
                          truncating=trunc_type)
    return padded

df = webscraper.get_news(ticker)

news_title = preprocess_text(df.News_Title)
news_title = model.predict(news_title)
df['sentiment'] = np.argmax(news_title, axis=-1)

api = tradeapi.REST("key in your own token", "key in your own token", "https://paper-api.alpaca.markets")
account = api.get_account()
# print(account)

mode_sentiment = df.sentiment.mode().iloc[0]

if mode_sentiment == 1:
    print("Neutral Sentiment, Nothing to do!")
elif mode_sentiment == 0:
    api.submit_order(
        symbol=ticker,
        qty=qty,
        side='sell',
        type='market',
        time_in_force='gtc'
    )
    print("SELL")
elif mode_sentiment == 2:
    api.submit_order(
        symbol=ticker,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    print("BUY")