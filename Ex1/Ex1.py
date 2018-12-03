import pandas as pd
import os
import sys
import nltk
import whoosh.index as index
import whoosh.query as query
from whoosh.fields import Schema, TEXT, STORED, BOOLEAN, NUMERIC, DATETIME,ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import *
from collections import Counter
import time
from lab import search, dot_product_similarity

PATH="en_docs.csv"
CLEANED_PATH = "en_docs_clean_test.csv"
QUERY_KW = "QUERY"
FLAG_CREATE = False
WHOOSH_SEARCH = False

'''
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))'''

if not os.path.exists("indexdir"):
   os.mkdir("indexdir")

def readData():
    print("Start reading data...")
    data = pd.read_csv(PATH, encoding="utf8")
    data.fillna("0", inplace=True)
    #data = data.groupby('id', as_index=False).agg(lambda x: x.tolist())
    print("making cleaning csv...")
    cleaned = data.groupby("md5sum_text", as_index=False).agg(lambda x: x.tolist())

    for index, row in cleaned.iterrows():
        realtext = ' '.join(row['text'])
        row['text'] = realtext

        #just check for potential errors, do nothing for now
        for potential_error_column in ("title", "date", "party"):
            if not all(x==row[potential_error_column][0] for x in row[potential_error_column]):
                print("not cleaned properly row:", index, "| column name:", potential_error_column)

        #compensate for poor joining becaus ims stewpid
        for column in ("party", "title"):
            row[column] = row[column][0]

    cleaned.to_csv( CLEANED_PATH) 
    return cleaned

def createSchema(data):
    print("Create schema")
    schema = Schema(number=NUMERIC,
                    text=TEXT(stored = True, vector=True),
                    party=NUMERIC(stored = True),
                    title=TEXT(stored = True, vector=True))

    ix = create_in("indexdir", schema)
    writer = ix.writer()
    print("Writing in schema", end='')
    for index, row in data.iterrows():
        if(int(index)%5==0):
            print(".", end='', flush=True)

        writer.add_document(number=row[1],
                    text=row["text"],
                    party=row["party"],
                    title=row["title"])
    print("done")
    print("commiting schema...")
    writer.commit()
    return ix

def result_statistics(results, searcher, qr):
    query_tokens = nltk.word_tokenize(qr)
    party_counts = dict()
    keyword_per_party_counts = dict()
    for docnum in results.docs():
        # party counts
        party = searcher.stored_fields(docnum)["party"]
        if(party not in party_counts):
            party_counts[party] = 0
        party_counts[party] += 1

        #keyword_per_party_counts
        term_freq_text = Counter(dict(searcher.vector_as("frequency", docnum, "text")))
        term_freq_title = Counter(dict(searcher.vector_as("frequency", docnum, "title")))
        total_term_freq = term_freq_text + term_freq_title

        if( party not in keyword_per_party_counts):
            keyword_per_party_counts[party] = dict()
            for tok in query_tokens:
                keyword_per_party_counts[party][ tok ] = 0

        for tok in query_tokens:
            if(tok in total_term_freq):
                keyword_per_party_counts[party][ tok ] += total_term_freq[tok]
        
    return party_counts, keyword_per_party_counts
            
def whoosh_search(searcher, qr, ix):
    qp = MultifieldParser(["title", "text"], schema=ix.schema)
    #qp = QueryParser("title", schema=ix.schema)
    
    q = qp.parse(qr)
    results = searcher.search(q, terms = True)

    return qp,q,results


def searchManifest(ix):
    qr = input("[METHOD_WHOOSH] " + QUERY_KW + ": ")
    while(qr != '-exit'):
        with ix.searcher() as searcher:
            start_time = time.time()

            qp,q,results = whoosh_search(searcher, qr, ix)
            print("---search time: %s seconds ---" % (time.time() - start_time))

            ndocs = len(results.docs())
            # print stats
            if( ndocs <= 0 ):
                print("no results found") 
            else:
                print("top results:", results)
                print("number of total results:", ndocs)
                #print("scores:", results.top_n )
                
                party_counts, word_per_party = result_statistics(results, searcher, qr)
                print("number of manifestos from party: ",party_counts)
                print("keyword per party: ",word_per_party)


        qr = input("\n" + "[METHOD_WHOOSH] " + QUERY_KW + ": ")

def search_lab(CLEANED_PATH):

    documents , i_index = search(CLEANED_PATH)

    while True:
        qr = input("[METHOD_LAB] " + QUERY_KW + ": ")

        start_time = time.time()
        print(qr)
        qr = nltk.word_tokenize(qr)
        similar_docs = dot_product_similarity(i_index, qr)
        print(similar_docs)
        print()

        print("---search time: %s seconds ---" % (time.time() - start_time))
        #qp,q,results,ndocs = lab.search(lab_search(searcher, qr, ix)

        continue
        # print stats
        '''
        if( ndocs <= 0 ):
            print("no results found") 
        else:
            print("top results:", results)
            print("number of total results:", ndocs)
            #print("scores:", results.top_n )
            
            party_counts, word_per_party = result_statistics(results, searcher, qr)
            print("number of manifestos from party: ",party_counts)
            print("keyword per party: ",word_per_party)
        '''

        qr = input("\n" + QUERY_KW + ": ")

if __name__ == '__main__':
    data = readData()

    if (len(sys.argv) > 1 and sys.argv[1]=="whoosh"):
        WHOOSH_SEARCH = True

    if WHOOSH_SEARCH and len(sys.argv) > 2 and sys.argv[2]=="generate":
        FLAG_CREATE = True

    if WHOOSH_SEARCH:
        if FLAG_CREATE:
            ix = createSchema(data)
        else:
            ix = index.open_dir("indexdir")

    if WHOOSH_SEARCH :
        searchManifest(ix)
    else:#search using lab 2
        search_lab(PATH)
        pass

