# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import json
import os
import gensim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding

DATA_DIRECTORY = f"{os.getcwd()}/COP_filt3_sub"

# Load the data

with open (f'{DATA_DIRECTORY}/filtered_data.json') as file:
    data = json.load(file)

# Prep the data

article_body = []
Y = []
for item in data:
    article_body.append(gensim.utils.simple_preprocess(item['headline']))
    Y.append(item['political_orientation'])
classes = sorted(list(set(Y)))
#Y = label_binarize(Y, classes)
Y = np.array(Y)


# Embeddings

def get_embedding(text, embeddings):
    total_embedding = []
    for word in text:
        try:
            total_embedding.append(embeddings[word.lower()])
        except KeyError:
            total_embedding.append(embeddings['UNK'])
    joined_embeddings = [total_embedding[0]]
    for item in total_embedding[1:]:
        joined_embeddings[0].extend(item)
    return(joined_embeddings[0])



X = []
embeddings = json.load(open('embeddings.json', 'r'))
for text in article_body[:100]:
    body_embedding = get_embedding(text, embeddings)
    X.append(body_embedding)

# Splitting the data into training and test

split_point = int(0.75 * len(X))
Xtrain = X[:split_point]
Xtest = X[split_point:]
Ytrain = Y[:split_point]
#Adapt Y to 100 aswell for testing purposes.
#Ytest = Y[split_point:]
Y_for_testing = Y[:100]
Ytest = Y_for_testing[split_point:]
print(Ytrain)
print(len(Xtrain))
print(len(Ytrain))
print(len(Xtest))
print(len(Ytest))

# create the neural network

model = keras.Sequential()
#embedding layer
model.add(layers.Embedding(input_dim=1000, output_dim=64))
#recurrent layer
model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#output layer
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the network

trained = model.fit(Xtrain, Ytrain, epochs = 20, batch_size = 10, shuffle = True, verbose = 1)

# Testing the network

Yguess = model.predict(Xtest, batch_size=10)
print('Classification accuracy on test: {0}'.format(accuracy_score(y_true=Ytest, y_pred=Yguess)))
#use macro to value categories evenly, because of uneven distribution
print('Classification precision on test: {0}'.format(precision_score(y_true=Ytest, y_pred=Yguess, average = 'macro')))
print('Classification recall on test: {0}'.format(recall_score(y_true=Ytest, y_pred=Yguess, average = 'macro')))
print('Classification F1-score on test: {0}'.format(f1_score(y_true=Ytest, y_pred=Yguess, average = 'macro')))


