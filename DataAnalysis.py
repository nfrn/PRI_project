import pandas
import numpy as np
import matplotlib.pyplot as plt
import nltk
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Convolution1D, regularizers, GlobalMaxPool1D, Dense
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from keras_preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


#nltk.download('stopwords')
#nltk.download('rslp')
from keras.preprocessing.text import text_to_word_sequence


PATH = "pt_docs_clean.csv"
MAX_WORDS_DOC = 60
LABELS = 6
ACTIVATION = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
OPTIMIZER = 'rmsprop'
#Training parameters
EPOCHS = 50
BATCH_SIZE = 50
VALIDATION = 0.2
DROPOUTRATE = 0.2
# Global Early stopping
EARLYSTOP = [EarlyStopping(monitor='val_loss', patience=50)]

# Description
# Index(['text','manifesto_id', 'party', 'date', 'title'], dtype='object')

# 0 'Portuguese Communist Party'
# 1 'Socialist Party'
# 2 "Ecologist Party ‘The Greens'"
# 3 'Social Democratic Party'
# 4 'Left Bloc'
# 5 'Social Democratic Center-Popular Party'

def visualize(df,string):
    # Visualize class imbalance
    classes = df.groupby('party')
    classes.agg(np.size).plot.bar(rot=0)
    plt.title('Number of texts per party' + string)
    plt.show(block=False)

    # Visualize len of texts
    df.plot.scatter(x='id',y='text_length')
    plt.title('Size of each Text' + string)
    plt.show(block=False)

    # Visualize mean len of texts per party
    classes = df.groupby('party')
    classes.agg({'text_length': 'mean'}).plot.bar(rot=0)
    plt.title('Mean size of texts per party' + string)
    plt.show(block=False)

def process_data(texts):
    stop_words = set(stopwords.words('portuguese'))
    st = RSLPStemmer()
    processeddocs = []

    for idx, sentence in enumerate(texts):
        sentence = text_to_word_sequence(sentence)
        filtered_sentence = []
        for w in sentence:
            if w not in stop_words:
                nw = st.stem(w)
                if w not in stop_words:
                    filtered_sentence.append(nw)
        processeddocs.append(filtered_sentence)

    return texts

def create_model():
    print("Create Model")
    sequence_input = Input(shape=(MAX_WORDS_DOC,1))
    conv_layer1 = Convolution1D(300, 5, activation="relu" )(
        sequence_input)
    pooling_layer1 = pool(conv_layer1)
    conv_layer1 = Convolution1D(300, 5, activation="relu")(
        sequence_input)
    pooling_layer1 = GlobalMaxPool1D()(conv_layer1)

    preds = Dense(LABELS, activation=ACTIVATION)(pooling_layer1)
    model = Model(sequence_input, preds)
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == '__main__':
    dataframe = pandas.read_csv(PATH, encoding="utf8", header=0)
    dataframe = dataframe.loc[:, ['text', 'party']]

    dataframe['party'] = dataframe['party'].map({
        'Portuguese Communist Party': 0,
        'Socialist Party': 1,
        'Social Democratic Party': 2,
        'Left Bloc': 3,
        'Social Democratic Center-Popular Party': 4,
        "Ecologist Party ‘The Greens'": 5
    })
    dataframe['text_length'] = dataframe['text'].str.split().str.len()
    dataframe['id'] = range(1, len(dataframe) + 1)

    #Visualizing
    print("Number of texts:" + str(len(dataframe.index)))
    #visualize(dataframe," before")
    #dataframe = dataframe[dataframe['text_length'] <= 1000]
    print("Number of texts:" + str(len(dataframe.index)))
    #visualize(dataframe, " after adjustment")

    dataframe = dataframe.loc[:, ['text', 'party']]
    training = dataframe.values
    data = np.append(training[:, 0], training[:, 1])
    rowsCounter = int(np.size(data, 0) / 2)
    texts = data[:rowsCounter]

    training_labels = data[rowsCounter:]
    training_texts = process_data(texts)

    t = Tokenizer()
    t.fit_on_texts(training_texts)
    training_inputs = t.texts_to_sequences(training_texts)
    training_inputs = pad_sequences(training_inputs, maxlen=MAX_WORDS_DOC,
                                    padding='post')

    training_texts = np.reshape(training_inputs,(training_inputs.shape[0],
                                MAX_WORDS_DOC,1))

    model = create_model()
    model.fit(training_texts, training_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
              shuffle=True, validation_split=VALIDATION,
                verbose=2, callbacks=EARLYSTOP)
    model.save_weights('text.h5')

