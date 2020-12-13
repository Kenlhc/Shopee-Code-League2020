import numpy as np
import pandas as pd
import os
import cv2
import re
import string
import emoji

en_df = pd.read_csv("dev_en.csv")
tcn_df = pd.read_csv("dev_tcn.csv", encoding='utf-8')

tcn_df = tcn_df.drop(['split'], axis=1)

en_list = en_df.values.tolist() 
tcn_list = tcn_df.values.tolist()

en_size = len(en_list)

def getChinese(context):
	filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
	context = filtrate.sub(r'', context) # remove all non-Chinese characters
	return context

def Punctuation(string): 
  
	# punctuation marks 
	punctuations = '''!()-[]{};:'"\\,<>./?@#$%^&*_~【】'''
  
	# traverse the given string and if any punctuation 
	# marks occur replace it with null 
	for x in string.lower(): 
		if x in punctuations: 
			string = string.replace(x, "")
	return string

for i in range(0, en_size):
	en_list[i] = str(en_list[i])
	#en_list[i] = alphanumeric(en_list[i])
	en_list[i] = re.sub("\d", "", en_list[i])
	en_list[i] = Punctuation(en_list[i])
	en_list[i] = en_list[i].lower()

tcn_size = len(tcn_list)

for i in range(0, tcn_size):
	tcn_list[i] = str(tcn_list[i])
	tcn_list[i] = re.sub("\d", "", tcn_list[i])
	tcn_list[i] = Punctuation(tcn_list[i])
	tcn_list[i] = getChinese(tcn_list[i])  
	

final_en_df = pd.DataFrame(en_list, columns=['translation_output'])
final_en_df.to_csv('cleaned_dev_en.csv', index = False)
final_tcn_df = pd.DataFrame(tcn_list, columns=['text'])
final_tcn_df.to_csv('cleaned_dev_tcn.csv', encoding='utf_8_sig', index = False)