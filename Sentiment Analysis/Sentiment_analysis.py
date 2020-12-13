# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 23:07:37 2020

@author: user
"""

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# All the ratings you want your neural network to detect
RATINGS = ["1", "2", "3", "4", "5"]

# Need to retrain after adding 'don't' back into sentences
clean_train_df = pd.read_csv("clean_train.csv")
clean_train_df = clean_train_df.drop(['review_id'], axis=1)
clean_test_df = pd.read_csv("clean_test.csv")
review_id = clean_test_df[["review_id"]]
clean_test_df = clean_test_df.drop('review_id',axis=1)

clean_train_list = clean_train_df.values.tolist()
clean_test_list = clean_test_df.values.tolist()
review_id = review_id.values.tolist()
clean_test_list = [str(x) for x in clean_test_list]

random.shuffle(clean_train_list)

X = [] #features
y = [] #labels

for features, label in clean_train_list:
	X.append(features)
	y.append(label)

X_size = len(X)
tokenizer = Tokenizer()

for i in range(0, X_size):
    X[i] = str(X[i])

tokenizer.fit_on_texts(X)
tokenizer.fit_on_texts(clean_test_list)

train_data = tokenizer.texts_to_sequences(X)
test_data = tokenizer.texts_to_sequences(clean_test_list)

#max_review_length = max([len(txt) for txt in test_data])
#print(max_review_length)

max_review_length = 175

train_pad = pad_sequences(train_data, maxlen = max_review_length)

test_pad = pad_sequences(test_data, maxlen = max_review_length)

vocab_size = len(tokenizer.word_index) + 1

#print(vocab_size)

model = Sequential()

model.add(Embedding(vocab_size, 100, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_pad, y, epochs=10, verbose=2)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_pad)

ratings_prediction = []
for i in range(0, len(predictions)):
    class_num = np.argmax(predictions[i])
    ratings_prediction.append(class_num)

review_df = pd.DataFrame(review_id, columns=['review_id'])
ratings_df = pd.DataFrame(ratings_prediction, columns=['rating'])
full_df = pd.concat([review_df, ratings_df], axis=1)
full_df.to_csv('test_predictions.csv', index = False)