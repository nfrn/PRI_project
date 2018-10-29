import pandas as pd
import numpy as np
import unicodedata
import re
from gensim.models import KeyedVectors
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding, Bidirectional, \
    SpatialDropout1D, Convolution1D, GlobalMaxPool1D, Dropout, Flatten, \
    MaxPooling1D, regularizers
from keras_preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


TRAIN = 1

PATH = "pt_docs_clean.csv"
MAX_WORDS_DOC = 100
EMBEDDINGS_DATA = "glove_s100.txt"
WORD_EMBEDDINGS_SIZE = 100
MODEL_PATH = "model.h5"
BEST_PATH = "model.h5"
#Training parameters
EPOCHS = 15
BATCH_SIZE = 50
VALIDATION = 0.2
DROPOUTRATE = 0.2

#Model parameters
ACTIVATION = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
OPTIMIZER = 'rmsprop'

# Global Early stopping
EARLYSTOP = [EarlyStopping(monitor='val_loss', patience=10)]


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


def decodeOutput(numb):
    if numb==0:
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

def prepareData(embeddings):
    print("Start reading data")
    data = pd.read_csv(PATH, encoding="utf8",header=0)
    training = data.values
    texts = np.append(training[:,0],training[:,2])
    rowsCounter = int(np.size(texts, 0) / 2)

    train_labels_texts = texts[rowsCounter:]
    training_labels = processLabels(train_labels_texts)

    for idx, sentence in enumerate(texts):
        cleansentence = text_to_word_sequence(sentence, lower=True, split=" ")
        finalsentence = ''
        for word in cleansentence:
            if word in embeddings.wv.vocab:
                finalsentence = finalsentence + " " + word
        texts[idx] = finalsentence

    t = Tokenizer()
    t.fit_on_texts(texts)

    voc_size = len(t.word_index) + 1
    print("Voc_size:" + str(voc_size))


    embedding_matrix = np.zeros((voc_size, WORD_EMBEDDINGS_SIZE))
    print("emb_size:" + str(embedding_matrix.shape))

    for word, i in t.word_index.items():
        if word in embeddings.wv.vocab:
            embedding_vector = embeddings.wv.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    train_input_texts = texts[:rowsCounter]

    training_inputs = t.texts_to_sequences(train_input_texts)
    training_inputs = pad_sequences(training_inputs, maxlen=MAX_WORDS_DOC,
                           padding='post')

    return training_inputs, training_labels, voc_size, embedding_matrix, \
            t


def loadEmbeddings():
    print("Load embeddings")
    model = KeyedVectors.load_word2vec_format(EMBEDDINGS_DATA)
    return model


def createModel(voc_size,embeddings):
    print("Create Model")
    sequence_input = Input(shape=(MAX_WORDS_DOC,), dtype='int32')
    embedded_sequences = Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings]
                          , input_length=MAX_WORDS_DOC,
                          trainable=False)(sequence_input)

    l_cov1 = Convolution1D(128,3, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(3)(l_cov1)
    l_cov2 = Convolution1D(128, 3, activation='relu')(l_pool1)
    l_pool2 = GlobalMaxPool1D()(l_cov2)
    l_dense = Dense(128, activation='relu')(l_pool2)
    preds = Dense(LABELS, activation=ACTIVATION)(l_dense)
    model = Model(sequence_input, preds)

    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy'])

    print(model.summary())
    return model

def trainModel(encoder, data, target):
    print("Train Model")
    print(data.shape)
    target = np.array(target)
    target = target.reshape(target.shape[0],1)
    print(target.shape)

    encoder.fit(data, target, epochs=EPOCHS, batch_size=BATCH_SIZE,
                shuffle=True, validation_split=VALIDATION,
                verbose=2, callbacks=EARLYSTOP)
    encoder.save_weights(MODEL_PATH)
    return encoder

def makePrediction(model,data):
    result = model.predict(data)
    output = decodeOutput(np.argmax(result))
    print(output)
    return output

if __name__ == '__main__' :
    embeddings = loadEmbeddings()
    training_inputs, training_labels, voc_size,embedding_matrix,\
    t = prepareData(embeddings)
    model = createModel(voc_size,embedding_matrix)

    if TRAIN:
        model_trained = trainModel(model, training_inputs, training_labels)
    else:
        model.load_weights(BEST_PATH)

    print(training_inputs[1])
    print(training_labels[1])
    makePrediction(model,training_inputs[1])


class Bi_rnn():

    def __init__(self):
        print("Loading embeddings")
        self.embeddings = loadEmbeddings()
        self.training_inputs, self.training_labels, self.voc_size, \
        self.embedding_matrix, \
        self.t = prepareData(self.embeddings)
        print("Loading model")
        self.model = createModel(self.voc_size, self.embedding_matrix)
        self.model.load_weights(BEST_PATH)

    def makePrediction(self,doc):
        cleansentence = text_to_word_sequence(doc,lower=True, split=" ")
        encoded = []
        for word in cleansentence:
            if word in self.embeddings.wv.vocab:
                if word in self.t.word_index:
                    encoded.append(self.t.word_index[word])
        encoded1 = pad_sequences([encoded], maxlen=MAX_WORDS_DOC,
                                  padding='post')
        result = makePrediction(self.model,encoded1)
        return result
