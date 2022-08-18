from tokenize import Token
from keras.models import load_model
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

'''
https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
https://stackoverflow.com/questions/59910151/keras-printing-out-the-predicted-class-label
'''

model_path = os.getcwd() + '\\main\\cnn_models\\model.h5'
dataset_path = os.getcwd() + '\\main\\cnn_models\\df_train.csv'

#kata = ['saya akan bunuh kamu sampai kamu mati di penjara']

def tokenize_df_train():
    d = pd.read_csv(dataset_path)
    d = d['Konten']
    return d

def word_to_sequence(kata):
    max_words=10000
    tokenizer=Tokenizer(max_words)
    d = tokenize_df_train()
    tokenizer.fit_on_texts(d)
    sequence_kata=tokenizer.texts_to_sequences(kata)
    padding_sequence = sequence_padding(sequence_kata)
    return padding_sequence
    
def sequence_padding(kata):
    paddy = pad_sequences(kata,maxlen=37)
    return paddy

def loading_model():
    model = load_model(model_path)
    return model

def prediksi(kata):
    model = loading_model()
    prediksi = word_to_sequence(kata)
    prediksi = model.predict(prediksi)
    MaxPosition=np.argmax(prediksi)
    classes = ['Bukan Ancaman', 'Ancaman'] 
    prediction_label=classes[MaxPosition]
    persentase = np.max(prediksi)
    persentase = persentase * 100
    print(prediction_label, persentase)
    return prediction_label, int(persentase)

