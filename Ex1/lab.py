import sys
from collections import OrderedDict
from operator import itemgetter   
import numpy as np
import nltk
import sklearn.feature_extraction.text as te
import pandas as pd

class Stats():

    def __init__(self, documents, inv_index):
        self.documents = documents
        self.inv_index = inv_index

        self.update()

    def update(self):
        self.n_docs = len(self.documents)
        self.n_terms = self.calc_terms()
        self.n_ind_terms = len(self.inv_index)

    def calc_terms(self):
        terms= 0
        for doc in self.documents:
            terms += len(nltk.word_tokenize(doc))

        return terms

    def print_doc_index(self):
        print(self.documents)
        print(self.inv_index)

    def __str__(self):
        string = ""

        string += "num of docs > " + str(self.n_docs) + "\n"
        string += "num of terms > "+ str(self.n_terms) + "\n"
        string += "num of individual terms > "+ str(self.n_ind_terms) + "\n"

        return string

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
    documents, i_index = read_documents(data)

    doc_stats = Stats(documents, i_index)
    #stats.print_doc_index()
    #print(idf_vals)
    #print(doc_stats)

    return documents, i_index

    '''
    while(True):
        query = input("search query: ")
        if(query == "x"):
            break

        query = nltk.word_tokenize(query)
        similar_docs = dot_product_similarity(query)
        print(similar_docs)
        print()

    return
    '''
    '''
    for term in sys.argv[1:]:
        print("-term \'"+term+"\'")

        df_val = 0
        min_tf = 0
        max_tf = 0
        idf_val = 0
        if (term in i_index):
            df_val = df_vals[term]
            min_tf = min_tfs[term]
            max_tf = max_tfs[term]
            idf_val = idf_vals[term]

        print("df >", df_val)
        print("min tf >", min_tf)
        print("max tf >", max_tf)
        print("idf >", idf_val)
        print("")
        '''
    #print(documents)
