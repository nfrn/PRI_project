import pandas as pd
import numpy as np
from keras import Input, Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding, Bidirectional, \
    SpatialDropout1D, Convolution1D, GlobalMaxPool1D, Dropout, Flatten, \
    MaxPooling1D, regularizers, Activation
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

from textRepresentation import retrieveProcessedData

TRAIN = 1

PATH = "pt_docs_clean.csv"
MAX_WORDS_DOC = 100
EMBEDDINGS_DATA = "glove_s100.txt"
WORD_EMBEDDINGS_SIZE = 100
MODEL_PATH = "model.h5"
BEST_PATH = "model.h5"
# Training parameters
EPOCHS = 15
BATCH_SIZE = 32
VALIDATION = 0.1
DROPOUTRATE = 0.2

# Model parameters
ACTIVATION = 'softmax'
LOSS_FUNCTION = 'categorical_crossentropy'
OPTIMIZER = 'rmsprop'

# Global Early stopping
EARLYSTOP = [EarlyStopping(monitor='val_loss', patience=10)]

'''6 different outputs'''
LABELS = 6


def processLabels(labels):
    training_labels = []
    for label in labels:
        if label == 0:
            training_labels.append([1,0,0,0,0,0])
        elif label == 1:
            training_labels.append([0,1,0,0,0,0])
        elif label == 2:
            training_labels.append([0,0,1,0,0,0])
        elif label == 3:
            training_labels.append([0,0,0,1,0,0])
        elif label == 4:
            training_labels.append([0,0,0,0,1,0])
        elif label == 5:
            training_labels.append([0,0,0,0,0,1])

    return training_labels

def decodeOutput(numb):
    if numb == 0:
        return 'Portuguese Communist Party'
    elif numb == 1:
        return 'Socialist Party'
    elif numb == 2:
        return "Ecologist Party ‘The Greens'"
    elif numb == 3:
        return 'Social Democratic Party'
    elif numb == 4:
        return 'Left Bloc'
    elif numb == 5:
        return 'Social Democratic Center-Popular Party'


def createModel(voc_size):
    print("Create Model")

    model = Sequential()

    model.add(Dense(128, input_dim=voc_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation('softmax'))

    # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

    print(model.summary())
    return model


def trainModel(encoder, data, target):
    print("Train Model")
    print(data.shape)
    target = np.array(target)
    target = target.reshape(target.shape[0], 6)
    print(target.shape)

    encoder.fit(data, target, epochs=EPOCHS, batch_size=BATCH_SIZE,
                shuffle=True, validation_split=VALIDATION,
                verbose=2, callbacks=EARLYSTOP)
    encoder.save_weights(MODEL_PATH)
    return encoder


def makePrediction(model, data):
    result = model.predict(data)
    print(result)
    output = decodeOutput(np.argmax(result))
    print(output)
    return output


if __name__ == '__main__':
    processed_corpus, processed_labels = retrieveProcessedData(True, True, True, 0.0001, 0.6, 1, 5000)
    processed_labels = processLabels(processed_labels)
    voc_size = np.shape(processed_corpus)[1]
    model = createModel(voc_size)

    if TRAIN:
        model_trained = trainModel(model, processed_corpus, processed_labels)
    else:
        model.load_weights(BEST_PATH)

    print(processed_corpus[1])
    print(processed_corpus[1])
    makePrediction(model, processed_corpus[1])
