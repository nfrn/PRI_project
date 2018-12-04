import sys
from collections import OrderedDict
from operator import itemgetter   
import numpy as np
import nltk
import sklearn.feature_extraction.text as te
import pandas as pd

doc_stats = None
documents = list()
inv_index = dict()
idf_vals = dict()
df_vals = dict()
min_tfs = dict()
max_tfs = dict()

def build_inverted_indexes(features, matrix, n_docs):
    for c in range(len(features)): #each term (each column)
        inv_index[features[c]] = dict()

        tf = []
        for l in range (len(matrix)):
            val  = matrix[l][c]
            tf.append(val)

            if (val != 0):
                inv_index[features[c]][l+1] = matrix[l][c]

        df = len(inv_index[features[c]])
        df_vals[features[c]] = df
        idf_vals[features[c]] = np.log(n_docs/df)

        min_tfs[features[c]] = min(tf)
        max_tfs[features[c]] = max(tf)

    return inv_index

def read_documents(data):
    for index, row in data.iterrows():
        documents.append(row['text'])

    #print ("data", documents)
    count_vect = te.CountVectorizer()
    transformed = count_vect.fit_transform(documents)

    features = count_vect.get_feature_names()
    matrix = transformed.toarray()
    #getting counts
    i_index = build_inverted_indexes(features, matrix, len(documents) )

    return documents, i_index

def dot_product_similarity(inv_index, terms):
    A = dict()

    for term in terms:
        if(term not in inv_index):
            print("term \'"+ term +"\' not in vocab")
            continue
        i_term_list = inv_index[term]
        idf = idf_vals[term]

        for doc, tf in i_term_list.items():
            if(doc not in A):
                Ad = 0
                A[doc] = Ad
        
            A[doc] += tf * idf
    #order by value
    return sorted(A.items(), key = itemgetter(1), reverse=True )

def search(data):
    docs, i_index = read_documents(data)
    return docs, i_index
