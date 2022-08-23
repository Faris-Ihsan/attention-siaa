from tokenize import Token
from keras.models import load_model, Model
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from main.Attention import Attention
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D

'''
https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
https://stackoverflow.com/questions/59910151/keras-printing-out-the-predicted-class-label
'''

cnn_model_path = os.getcwd() + '\\main\\cnn_models\\cnn_models.h5'
lstm_model_path = os.getcwd() + '\\main\\cnn_models\\lstm_models.h5'
dataset_path = os.getcwd() + '\\main\\cnn_models\\df_train.csv'
embedding_matrix_path = os.getcwd() + '\\main\\cnn_models\\embedding_matrix.npy'
embed_size = 300 # how big is each word vector
max_features = 1774 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use


#kata = ['saya akan bunuh kamu sampai kamu mati di penjara']

def tokenize_df_train():
    d = pd.read_csv(dataset_path)
    # print(d)
    # d = d['Konten']
    return d

def word_to_sequence(kata):
    tokenizer=Tokenizer(num_words=max_features)
    d = tokenize_df_train()
    tokenizer.fit_on_texts(d)
    sequence_kata=tokenizer.texts_to_sequences(kata)
    padding_sequence = pad_sequences(sequence_kata, padding='post', maxlen=maxlen)
    return padding_sequence
    
def sequence_padding(kata):
    paddy = pad_sequences(kata,maxlen=37)
    return paddy

def load_cnn_model():
    cnn_model = load_model(cnn_model_path)
    return cnn_model

def load_matrix():
    embedding_matrix = np.load(embedding_matrix_path)
    return embedding_matrix

def build_model():
    embedding_matrix = load_matrix()
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model

def load_lstm_model():
    model = build_model()
    model.load_weights(lstm_model_path)
    return model

def prediksi(kata):
    cnn_model = load_cnn_model()
    lstm_model = load_lstm_model()
    sequence = word_to_sequence(kata)
    prediksi_cnn = cnn_model.predict(sequence)
    prediksi_lstm = lstm_model.predict(sequence)
    # MaxPosition=np.argmax(prediksi)
    # classes = ['Bukan Ancaman', 'Ancaman'] 
    # prediction_label=classes[MaxPosition]
    # persentase = np.max(prediksi)
    # persentase = persentase * 100
    print(prediksi_cnn, prediksi_lstm)
    # print(prediction_label, persentase)

    #sementara biar ga error
    prediction_label = 'yeah'
    persentase = 100
    return prediction_label, int(persentase)

