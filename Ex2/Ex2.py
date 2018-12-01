from datetime import time

from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import svm
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from itertools import product
import pickle
from sklearn.svm import LinearSVC
from textRepresentation import retrieveProcessedData
import sys

dic = {
    0:'Portuguese Communist Party',
    1:'Socialist Party',
    2:'Social Democratic Party',
    3:'Left Bloc',
    4:'Social Democratic Center-Popular Party',
    5:"Ecologist Party ‘The Greens'"
}

if __name__ == '__main__':


    #input = sys.argv[1]
    #print("Input: " + input)
    input = "Direita"

    vect = joblib.load('bestModel/IDF.sav')

    input2 = vect.transform([input])

    model = joblib.load('bestModel/model.sav')

    output = model.predict(input2)

    party = dic[output[0]]
    print(party)
