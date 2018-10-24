import pandas as pd
import numpy as np
import unicodedata
import re
from gensim.models import KeyedVectors
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding, Bidirectional
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


def prepareData(embeddings):
    print("Start reading data")
    data = pd.read_csv(PATH, encoding="utf8",header=0)

    training = data.values
    texts = np.append(training[:,0],training[:,2])
    rowsCounter = int(np.size(texts, 0) / 2)

    train_labels_texts = texts[rowsCounter:]
    training_labels = processLabels(train_labels_texts)

    for idx, sentence in enumerate(texts):
        newsentence = cleanText(sentence)
        cleansentence = ''
        for word in newsentence:
            if word in embeddings.wv.vocab:
                cleansentence += word
        texts[idx] = cleansentence
    t = Tokenizer()
    t.fit_on_texts(texts)

    voc_size = len(t.word_index) + 1
    print("Voc_size:" + str(voc_size))

    print("Start adapt embeddings")
    embedding_matrix = np.zeros((voc_size, 50))

    print("emb_size:" + str(embedding_matrix.shape))

    for word, i in t.word_index.items():
        if word in embeddings.wv.vocab:
            embedding_vector = embeddings.wv.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    reverse_word_map = dict(map(reversed, t.word_index.items()))

    train_input_texts = texts[:rowsCounter]

    training_inputs = t.texts_to_sequences(train_input_texts)
    training_inputs = pad_sequences(training_inputs, maxlen=MAX_WORDS_DOC,
                           padding='post')

    return training_inputs, training_labels, voc_size, embedding_matrix, \
           reverse_word_map


def loadEmbeddings():
    print("Load embeddings")
    model = KeyedVectors.load_word2vec_format('skip_s50.txt')
    return model


def remove_accents(input_str):
    nkfd_form = unicodedata.normalize('NFKD', np.unicode(input_str))
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])


def cleanText(text):
    # Punctuation list
    punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')

    # ##### #
    # Regex #
    # ##### #
    re_remove_brackets = re.compile(r'\{.*\}')
    re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
    re_transform_numbers = re.compile(r'\d', re.UNICODE)
    re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
    re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
    # Different quotes are used.
    re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
    re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
    re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
    re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
    re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
    re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
    re_tree_dots = re.compile(u'…', re.UNICODE)
    # Differents punctuation patterns are used.
    re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                           (punctuations, punctuations), re.UNICODE)
    re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                             (punctuations, punctuations), re.UNICODE)
    re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
    re_changehyphen = re.compile(u'–')
    re_doublequotes_1 = re.compile(r'(\"\")')
    re_doublequotes_2 = re.compile(r'(\'\')')
    re_trim = re.compile(r' +', re.UNICODE)

    text = text.lower()
    text = text.replace('\xa0', ' ')
    text = re_tree_dots.sub('...', text)
    text = re.sub('\.\.\.', '', text)
    text = re_remove_brackets.sub('', text)
    text = re_changehyphen.sub('-', text)
    text = re_remove_html.sub(' ', text)
    text = re_transform_numbers.sub('0', text)
    text = re_transform_url.sub('URL', text)
    text = re_transform_emails.sub('EMAIL', text)
    text = re_quotes_1.sub(r'\1"', text)
    text = re_quotes_2.sub(r'"\1', text)
    text = re_quotes_3.sub('"', text)
    text = re.sub('"', '', text)
    text = re_dots.sub('.', text)
    text = re_punctuation.sub(r'\1', text)
    text = re_hiphen.sub(' - ', text)
    text = re_punkts.sub(r'\1 \2 \3', text)
    text = re_punkts_b.sub(r'\1 \2 \3', text)
    text = re_punkts_c.sub(r'\1 \2', text)
    text = re_doublequotes_1.sub('\"', text)
    text = re_doublequotes_2.sub('\'', text)
    text = re_trim.sub(' ', text)
    return text.strip()

def createModel(voc_size,embeddings):
    print("Create Model")
    sequence_input = Input(shape=(MAX_WORDS_DOC,), dtype='int32')
    embedded_sequences = Embedding(voc_size, WORD_EMBEDDINGS_SIZE, weights=[embeddings]
                          , input_length=MAX_WORDS_DOC,
                          trainable=False)(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    preds = Dense(LABELS, activation=ACTIVATION)(l_lstm)
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

    encoder.fit(data, target, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, validation_split=VALIDATION,
                verbose=2, callbacks=EARLYSTOP)
    encoder.save_weights(MODEL_PATH)
    return encoder

def makePrediction():
    print("Make Prediction")

if __name__ == '__main__' :
    embeddings = loadEmbeddings()
    training_inputs, training_labels, voc_size,embedding_matrix,\
    reverse_word_map = prepareData(embeddings)
    model = createModel(voc_size,embedding_matrix)
    model_trained = trainModel(model,training_inputs,training_labels)
