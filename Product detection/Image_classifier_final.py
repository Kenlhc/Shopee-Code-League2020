# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:10:11 2020

@author: user
"""

import numpy as np
import pandas as pd
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow import keras


file_list = []
class_list = []

DATADIR = "C:/Users/user/Desktop/Shopee_Code_League_Stuff/shopee-product-detection-dataset/train/train"

# All the categories you want your neural network to detect
CATEGORIES = ["00", "01", "02", "03", "04",
	      "05", "06", "07", "08", "09",
	      "10", "11", "12", "13", "14", "15", "16",
          "17", "18", "19", "20", "21", "22", "23",
          "24", "25", "26", "27", "28", "29", "30",
          "31", "32", "33", "34", "35", "36", "37",
          "38", "39", "40", "41"]

# Can try increasing to improve accuracy
# 48 about 1hr+
IMG_SIZE = 64

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        count = 0
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                count += 1
                if count == 1600:
                    break
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0

model = Sequential()
#3 convolutional layers

model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#1 hidden layers

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Activation("relu"))

#output layer
model.add(Dense(42, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=20)

TESTDIR = "C:/Users/user/Desktop/Shopee_Code_League_Stuff/shopee-product-detection-dataset/test/test"

test_data = []
filenames = []
class_predictions = []
test_data_in_csv = pd.read_csv("C:/Users/user/Desktop/Shopee_Code_League_Stuff/shopee-product-detection-dataset/test.csv")
test_data_filenames = test_data_in_csv['filename'].values.tolist()

def create_test_data():
    for img in os.listdir(TESTDIR):
        try:
            if img in test_data_filenames:
                filenames.append(img)
                img_array = cv2.imread(os.path.join(TESTDIR, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append(new_array)
        except Exception as e:
            pass

create_test_data()

X_test = [] #features

for features in test_data:
    X_test.append(features)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test / 255.0

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
#softmax layer normalizes output into a probability distribution

predictions = probability_model.predict(X_test)
for i in range(0, len(predictions)):
    class_num = np.argmax(predictions[i])
    class_predictions.append(class_num)

filename_df = pd.DataFrame(filenames, columns=['filename'])
category_df = pd.DataFrame(class_predictions, columns=['category'])
category_df['category'] = category_df.category.apply(lambda c: "{:02d}".format(c))
full_df = pd.concat([filename_df, category_df], axis=1)
full_df.to_csv('test_predictions1.csv', index = False)