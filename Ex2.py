import pandas as pd
import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

PATH = "pt_docs_clean.csv"
MAX_WORDS_DOC = 121
EMBEDDINGS_DATA = "glove.6B.100d.txt"
WORD_EMBEDDINGS_SIZE = 50
MODEL_PATH = "model.h5"
#Training parameters
EPOCHS = 15
BATCH_SIZE = 64
VALIDATION = 0.2
DROPOUTRATE = 0.2

#Model parameters
ACTIVATION = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
OPTIMIZER = 'rmsprop'

# Global Early stopping
EARLYSTOP = [EarlyStopping(monitor='val_loss', patience=3)]

'''Index(['text','manifesto_id', 'party', 'date', 'title'], dtype='object')'''

'''['Portuguese Communist Party' 'Socialist Party'
 "Ecologist Party ‘The Greens'" 'Social Democratic Party' 'Left Bloc'
 'Social Democratic Center-Popular Party']'''

'''6 different outputs'''
LABELS = 6
def processLabels(texts ):
    training_labels = []
    for name in texts:
        if name == 'Portuguese Communist Party':
            training_labels.append(0)
        elif name == 'Socialist Party':
            training_labels.append(1)
        elif name == "Ecologist Party ‘The Greens'":
            training_labels.append(2)
        elif name == 'Social Democratic Party':
            training_labels.append(3)
        elif name == 'Left Bloc':
            training_labels.append(4)
        elif name == 'Social Democratic Center-Popular Party':
            training_labels.append(5)

    return training_labels


def prepareData():
    print("Start reading data")
    data = pd.read_csv(PATH, encoding="utf8",header=0)

    training = data.values
    texts = np.append(training[:,0],training[:,2])
    t = Tokenizer()
    t.fit_on_texts(texts)

    rowsCounter = int(np.size(texts, 0) / 2)

    train_input_texts = texts[:rowsCounter]
    train_labels_texts = texts[rowsCounter:]

    training_inputs = t.texts_to_sequences(train_input_texts)
    training_inputs = pad_sequences(training_inputs, maxlen=MAX_WORDS_DOC,
                           padding='post')
    training_labels = processLabels(train_labels_texts)

    voc_size = len(t.word_index)
    return training_inputs, training_labels, voc_size


def prepareEmbeddings(data):
    print("Start embeddings")

def createModel(voc_size):
    print("Create Model")

    ### build encoder
    enc_input = Input(shape=(MAX_WORDS_DOC,), dtype='int32',
                      name='encoder_input')
    enc_embed = Embedding(voc_size, WORD_EMBEDDINGS_SIZE
                          , input_length=MAX_WORDS_DOC,
                          trainable=True)(enc_input)
    enc_lstm = LSTM(WORD_EMBEDDINGS_SIZE, return_state=True,
                    return_sequences=True,
                    input_shape=(None, MAX_WORDS_DOC,
                                 WORD_EMBEDDINGS_SIZE), dropout=DROPOUTRATE)
    sequence, state1, state2 = enc_lstm(enc_embed)
    enc_states = [state1, state2]

    ### build decoder

    dec_sequence = LSTM(WORD_EMBEDDINGS_SIZE, return_sequences=True,
                        dropout=DROPOUTRATE)(sequence, enc_states)

    decoder_dense = Dense(LABELS, activation=ACTIVATION)(dec_sequence)

    ### build model
    model = Model(input=enc_input, output=decoder_dense)
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy'])

    print(model.summary())
    return model

def trainModel(encoder, data, target):
    print("Train Model")
    target = target.reshape(target.shape[0], MAX_WORDS_DOC, 1)
    encoder.fit(data, target, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, validation_split=VALIDATION,
                verbose=2, callbacks=EARLYSTOP)

    encoder.save_weights(MODEL_PATH)
    return encoder

def makePrediction():
    print("Make Prediction")

if __name__ == '__main__' :
    training_inputs, training_labels, voc_size = prepareData()
    model = createModel(voc_size)
    model_trained = trainModel(model,training_inputs,training_labels)

