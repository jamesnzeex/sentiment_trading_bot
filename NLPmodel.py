import keras
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_model():
    df = pd.read_csv('./data/financial_news.csv')
    df.columns = ['sentiment', 'text']
    mapper = {'negative': 0,
            'neutral': 1,
            'positive': 2,}

    df.sentiment = df.sentiment.map(mapper)

    train, valid = train_test_split (df, test_size = 0.2)

    train_text = np.array(train.text.tolist().copy())
    train_label = keras.utils.to_categorical(train.sentiment.astype('int64'))

    valid_text = np.array(valid.text.tolist().copy())
    valid_label = keras.utils.to_categorical(valid.sentiment.astype('int64'))

    vocab_size = 1000
    embedding_dim = 16
    max_length = 150
    trunc_type = 'post'
    padding_type = 'post'
    oov_token = '<OOV>'

    tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
    tokenizer.fit_on_texts(train_text)
    #tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_text)
    train_padded = pad_sequences(train_sequences, maxlen =  max_length, padding = padding_type, truncating = trunc_type)

    valid_sequences = tokenizer.texts_to_sequences(valid_text)
    valid_padded = pad_sequences(valid_sequences, maxlen =  max_length, padding = padding_type, truncating = trunc_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(embedding_dim, activation = 'relu'),
        tf.keras.layers.Dense(3, activation = 'softmax')
    ])

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary ()

    epochs = 25
    history = model.fit(train_padded, train_label, epochs = epochs, validation_data = (valid_padded, valid_label))

    model.save('trained_model')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)