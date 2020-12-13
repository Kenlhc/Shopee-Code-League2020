import numpy as np
import pandas as pd
import os
import cv2
import re
import string
import emoji
import random 

clean_test_df = pd.read_csv("clean_test.csv")
clean_test_df = clean_test_df.drop('review',axis=1)

clean_test_list = clean_test_df.values.tolist() 

test_size = len(clean_test_list)

my_randoms = [] 

for i in range(test_size):
	my_randoms.append(random.randrange(4,6,1))

review_df = pd.DataFrame(clean_test_list, columns=['review_id'])
ratings_df = pd.DataFrame(my_randoms, columns=['rating'])
full_df = pd.concat([review_df, ratings_df], axis=1)
full_df.to_csv('test_predictions_45.csv', index = False)