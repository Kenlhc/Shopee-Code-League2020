# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:30:29 2020

@author: user
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow import keras

batch_size = 64  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

input_df = pd.read_csv("cleaned_dev_tcn.csv", encoding='utf-8')
target_df = pd.read_csv("cleaned_dev_en.csv")

input_texts = input_df.values.tolist()
target_texts = target_df.values.tolist()
input_characters = set()
target_characters = set()

input_size = len(input_texts)

for i in range(0, input_size):
    input_texts[i] = str(input_texts[i])
    target_texts[i] = str(target_texts[i])
    for char in input_texts[i]:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_texts[i]:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

'''
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
'''

# mapping token to index， easily to vectors
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# np.zeros(shape, dtype, order)
# shape is an tuple, in here 3D
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# input_texts contain all english sentences
# output_texts contain all chinese sentences
# zip('ABC','xyz') ==> Ax By Cz, looks like that
# the aim is: vectorilize text, 3D
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        # 3D vector only z-index has char its value equals 1.0
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            # igone t=0 and start t=1, means
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
# input prodocts keras tensor, to fit keras model!
# 1x73 vector
# encoder_inputs is a 1x73 tensor!
encoder_inputs = Input(shape=(None, num_encoder_tokens))

# units=256, return the last state in addition to the output
encoder_lstm = LSTM((latent_dim), return_state=True)

# LSTM(tensor) return output, state-history, state-current
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM((latent_dim), return_sequences=True, return_state=True)

# obtain output
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

# dense 2580x1 units full connented layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

# why let decoder_outputs go through dense ?
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn, groups layers into an object
# with training and inference features
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
# model(input, output)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
# compile -> configure model for training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model optimizsm
model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('seq2seq.h5')

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.
    # this target_seq you can treat as initial state

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        # argmax: Returns the indices of the maximum values along an axis
        # just like find the most possible char
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # find char using index
        sampled_char = reverse_target_char_index[sampled_token_index]
        # and append sentence
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        # append then ?
        # creating another new target_seq
        # and this time assume sampled_token_index to 1.0
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        # update states, frome the front parts
        states_value = [h, c]

    return decoded_sentence

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

val_input_df = pd.read_csv("test_tcn.csv", encoding='utf-8')
val_input_df = val_input_df.drop(['split'], axis=1)
val_input_texts = val_input_df.values.tolist()

val_input_size = len(val_input_texts)

for i in range(0, val_input_size):
	val_input_texts[i] = str(val_input_texts[i])
	val_input_texts[i] = re.sub("\d", "", val_input_texts[i])
	val_input_texts[i] = Punctuation(val_input_texts[i])
	val_input_texts[i] = getChinese(val_input_texts[i])

val_encoder_input_data = np.zeros(
  (len(val_input_texts), max([len(txt) for txt in val_input_texts]),
   num_encoder_tokens), dtype='float32')

for i, input_text in enumerate(val_input_texts):
  for t, char in enumerate(input_text):
    val_encoder_input_data[i, t, input_token_index[char]] = 1.

final_output = []

for seq_index in range(10000):
  input_seq = val_encoder_input_data[seq_index: seq_index + 1]
  decoded_sentence = decode_sequence(input_seq)
  final_output.append(decoded_sentence)

final_output_df = pd.DataFrame(final_output, columns=['translation_output'])
final_output_df.to_csv('test_translations.csv', index = False)