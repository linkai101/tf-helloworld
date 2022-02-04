import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM , Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import string
import pickle

#importing the dataset
with open('intents.json') as intentsFile:
  intents = json.load(intentsFile)['intents']
#getting all the data to lists
tags = []
inputs = []
responses={}
for intent in intents:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])
#converting to dataframe
data = pd.DataFrame({ "inputs":inputs,"tags":tags })

#removing punctuations
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
#tokenize the data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

#apply padding
x_train = pad_sequences(train)

#encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

#input length
input_shape = x_train.shape[1]
#define vocabulary
vocabulary = len(tokenizer.word_index)
#output length
output_length = le.classes_.shape[0]

# loading
model = tf.keras.models.load_model('model.h5')

#chatting
import random
while True:
  texts_p = []
  prediction_input = input('You: ')
  #print('You: ' + prediction_input)
  #if prediction_input.strip() == '': break
  #removing punctuation and converting to lowercase
  prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)
  texts_p.append(prediction_input)
  #tokenizing and padding
  prediction_input = tokenizer.texts_to_sequences(texts_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  prediction_input = pad_sequences([prediction_input],input_shape)
  #getting output from model
  output = model.predict(prediction_input)
  output = output.argmax()
  #finding the right tag and predicting
  response_tag = le.inverse_transform([output])[0]
  print("Bot: ",random.choice(responses[response_tag]))
  if response_tag == "goodbye":
    break