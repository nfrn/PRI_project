import numpy as np
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from nltk import RSLPStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import *
import winsound

PATH = "pt_docs_clean.csv"
MAX_WORDS_DOC = 60
LABELS = 6

def process_data(texts, swFlag, stFlag):
    stop_words = set()
    if swFlag:
        stop_words = set(stopwords.words('portuguese'))
    if stFlag:
        st = RSLPStemmer()

    processeddocs = []

    for idx, sentence in enumerate(texts):
        sentenceList = text_to_word_sequence(sentence)
        processedSentence = ''
        for word in sentenceList:
            if word not in stop_words:
                if stFlag:
                    newWord = st.stem(word)
                    if newWord not in stop_words:
                        processedSentence = processedSentence + ' ' + newWord
                else:
                    processedSentence = processedSentence + ' ' + word
        processeddocs.append(processedSentence)

    return processeddocs


def retrieveProcessedData(swFlag,stFlag, tf_idf, minf, maxf, ngram, max = ''):
    print("Process data with input: StopWord:=" + str(swFlag) + ", Stemming=" + str(stFlag)
          + ", Tf_idf=" + str(tf_idf) + ", MinFREQ=" + str(minf) + ", MaxFREQ=" + str(maxf) + ", MaxNgram=" + str(ngram))
    dataframe = pd.read_csv(PATH, encoding="utf8", header=0)
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

    dataframe = dataframe.loc[:, ['text', 'party']]
    training = dataframe.values
    data = np.append(training[:, 0], training[:, 1])
    rowsCounter = int(np.size(data, 0) / 2)

    texts = data[:rowsCounter]
    texts_labels = data[rowsCounter:]
    texts_labels = texts_labels.astype('int')

    #Diferent Process Mechanisms
    corpus = process_data(texts,swFlag,stFlag)

    #Effect on corpus count values
    if tf_idf:
        if max != '':
            vectorizer = TfidfVectorizer(sublinear_tf=True,norm='l2',strip_accents='ascii', lowercase=True, ngram_range=(1, ngram), min_df=minf, max_df=maxf, max_features=max)
            model = vectorizer.fit_transform(corpus)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True,norm='l2',strip_accents='ascii', lowercase=True, ngram_range=(1, ngram), min_df=minf, max_df=maxf)
            model = vectorizer.fit_transform(corpus)
    else:
        vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, ngram_range=(1, ngram), min_df=minf, max_df=maxf)
        model = vectorizer.fit_transform(corpus)

    print("Number of features: " + str(len(vectorizer.get_feature_names())))
    return model, texts_labels, vectorizer
