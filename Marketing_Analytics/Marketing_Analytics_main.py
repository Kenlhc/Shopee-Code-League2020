# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:34:28 2020

@author: user
"""

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow import keras
from keras.regularizers import l2

# Label is either true or false
OPEN = ["0", "1"]

#Need to try removing more features/cleaning data better
clean_train_df = pd.read_csv("clean_final_train.csv")
clean_train_df = clean_train_df.drop(['row_id'], axis=1)
labels_df = clean_train_df['open_flag']
clean_train_df = clean_train_df.drop(['open_flag'], axis=1)
clean_test_df = pd.read_csv("clean_final_test.csv")
row_id = clean_test_df[["row_id"]]
clean_test_df = clean_test_df.drop(['row_id'],axis=1)

#remove user_id features
clean_train_df = clean_train_df.drop(['user_id'], axis=1)
clean_test_df = clean_test_df.drop(['user_id'], axis=1)


clean_train_list = clean_train_df.values.tolist()
clean_test_list = clean_test_df.values.tolist()
row_id = row_id.values.tolist()
labels = labels_df.values.tolist()

X = [] #features
y = [] #labels

size = len(labels)

for i in range(0, size):
	X.append(clean_train_list[i])
	y.append(labels[i])

X = np.array(X)

model = Sequential()

model.add(Input(shape=(20,)))
model.add(Dense(16, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=5, verbose=1)

Xnew = np.array(clean_test_list)
# make a prediction
ynew = model.predict_classes(Xnew)

open_prediction = ynew

row_df = pd.DataFrame(row_id, columns=['row_id'])
predict_df = pd.DataFrame(open_prediction, columns=['open_flag'])
full_df = pd.concat([row_df, predict_df], axis=1)
full_df.to_csv('test_predictions.csv', index = False)