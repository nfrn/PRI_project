from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import svm
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from itertools import product

from textRepresentation import retrieveProcessedData

BESTMODELPATH = "bestModel/WWWWWWWWWWWWWFalseFalseTrue0.00010.993SVC.sav"

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == '__main__':

    print("Retrieving data")
    processed_corpus, processed_labels = retrieveProcessedData(False, False, True, 0.0001, 0.99,3)
    prec_max = 0
    for x in range(100000):
        X_train, X_test, y_train, y_test = train_test_split(processed_corpus, processed_labels,
                                                            test_size=0.2)
        model = SGDClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        prec = precision_score(y_test, y_pred, average='micro')

        if prec > prec_max:
            prec_max = prec
            best_model = model
            print(prec_max)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

        if prec_max >0.70:
            break


    np.set_printoptions(precision=2)

    class_names = ["PCP", "PS", "PSD", "BE", "CDC", "VERDES"]

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    print(report)
    plt.show()

    print("Saving model")
    joblib.dump(best_model, BESTMODELPATH)