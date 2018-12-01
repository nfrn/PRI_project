import time

import pickle
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

import pandas as pd
import numpy as np
from Ex2.textRepresentation import retrieveProcessedData


def classification_report_csv(report, name):
    #Credits to https://stackoverflow.com/a/41044355
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}

        row_data = line.split(' ')
        row_data = list(filter(None, row_data))

        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(name, index = False)

def getModel(name):
    if name == "BernoulliNB":
        return BernoulliNB()
    elif name == "MultinomialNB":
        return MultinomialNB()
    elif name == "GaussianNB":
        return GaussianNB()
    elif name == "SVC":
        return SGDClassifier()

if __name__ == '__main__':

    swFlagList = [True, False]
    stFlagList = [True, False]
    tf_idfList = [True, False]
    minfList = [0.0001, 5, 0.005]
    maxfList = [0.50,0.99]
    ngramList = [1,2,3]
    models = ["MultinomialNB", "BernoulliNB", "GaussianNB", "SVC"]
    for swFlag in swFlagList:
        for stFlag in stFlagList:
            for tf_idf in tf_idfList:
                for minf in minfList:
                    for maxf in maxfList:
                        for ngram in ngramList:

                            processed_corpus, processed_labels = retrieveProcessedData(swFlag, stFlag, tf_idf, minf, maxf, ngram)
                            X_train, X_test, y_train, y_test = train_test_split(processed_corpus, processed_labels, test_size=0.2)

                            label0 = 0
                            label1 = 0
                            label2 = 0
                            label3 = 0
                            label4 = 0
                            label5 = 0

                            for label in y_test:
                                if label == 0:
                                    label0 += 1
                                if label == 1:
                                    label1 += 1
                                if label == 2:
                                    label2 += 1
                                if label == 3:
                                    label3 += 1
                                if label == 4:
                                    label4 += 1
                                if label == 5:
                                    label5 += 1

                            total = label0+label1+label2+label3+label4+label5
                            print(total)
                            print(str(label0 / total))
                            print(str(label1 / total))
                            print(str(label2 / total))
                            print(str(label3 / total))
                            print(str(label4 / total))
                            print(str(label5 / total))
                            for model in models:

                                print(model)
                                start = time.time()
                                clf = getModel(model)
                                for i in range(1, 21):
                                    print('T_' + str(i))
                                    clf.partial_fit(X_train[int(X_train.shape[0] / 20) * (i - 1):int(
                                        X_train.shape[0] / 20) * i].toarray(),
                                                    y_train[int(y_train.shape[0] / 20) * (i - 1):int(
                                                        y_train.shape[0] / 20) * i], classes=np.unique(
                                            y_train))

                                end = time.time()
                                print("Model took %0.2f seconds to train" % (end - start))

                                predicted = []
                                print("Total: " + str(X_test.shape[0]))
                                for i in range(0, 3):
                                    beg = int(X_test.shape[0] / 3) * i
                                    to = int(X_test.shape[0] / 3) * (i+1)
                                    print("Beg: " + str(beg) + "  To: " + str(to))
                                    predicted = np.append(predicted,clf.predict(
                                        (X_test[beg:to].toarray())
                                    ))
                                if to < X_test.shape[0]:
                                    predicted = np.append(predicted, clf.predict(
                                        (X_test[to:X_test.shape[0]].toarray())
                                    ))

                                print(predicted)
                                report = classification_report(y_test, predicted)
                                print(report)

                                report_name = "results2/" + str(swFlag) + str(stFlag) + str(tf_idf)+ str(minf)+ str(maxf)+ str(ngram)+ str(model) + ".csv"
                                classification_report_csv(report,report_name)

                                filename = "models2/" + str(swFlag) + str(stFlag) + str(tf_idf)+ str(minf)+ str(maxf)+ str(ngram)+ str(model) + ".sav"
                                pickle.dump(clf, open(filename, 'wb'))

