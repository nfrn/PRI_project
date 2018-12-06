import pandas as pd
import os
import sys
import nltk
import whoosh.index as index
import whoosh.query as query
from whoosh.fields import Schema, TEXT, STORED, BOOLEAN, NUMERIC, DATETIME,ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import *
#from whoosh.scoring.Weighting import BM25F, tf
from collections import Counter
import time
from lab import search, dot_product_similarity

PATH="en_docs.csv"
CLEANED_PATH = "en_docs_clean.csv"
CLEAN_DATA = True
QUERY_KW = "QUERY"
FLAG_CREATE = False
WHOOSH_SEARCH = True

'''
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))'''

if not os.path.exists("indexdir"):
   os.mkdir("indexdir")

def readData(path):
    print("Start reading data...")
    data = pd.read_csv(path, encoding="utf8")
    data.fillna(" ", inplace=True)
    #data = data.groupby('id', as_index=False).agg(lambda x: x.tolist())
    if not CLEAN_DATA:
        return data

    print("making cleaning csv...")
    cleaned = data.groupby("manifesto_id", as_index=False).agg(lambda x: x.tolist())

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
    start_time = time.time()
    schema = Schema(number=NUMERIC,
                    text=TEXT(stored = True, vector=True),
                    party=TEXT(stored = True),
                    title=TEXT(stored = True, vector=True))

    ix = create_in("indexdir", schema)
    writer = ix.writer()
    print("Writing in schema", end='')
    
    if CLEAN_DATA:
        num_column = 1
    else:
        num_column = 0

    for index, row in data.iterrows():
        if(int(index)%(5 if CLEAN_DATA else 50)==0):
            print(".", end='', flush=True)

        writer.add_document(
                    text=row["text"],
                    party=row["party"],
                    title=row["title"])
    print("done")
    print("commiting schema...")
    writer.commit()
    print("---schema indexing time: ", (time.time()-start_time), "s ---")
    return ix

def keyword_party_counts(query_tokens, keyword_per_party_counts, total_term_freq, party):
    if( party not in keyword_per_party_counts):
        keyword_per_party_counts[party] = dict()
        for tok in query_tokens:
            keyword_per_party_counts[party][ tok ] = 0

    for tok in query_tokens:
        if(tok in total_term_freq):
            keyword_per_party_counts[party][ tok ] += total_term_freq[tok]

    return keyword_per_party_counts


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

        keyword_per_party_counts = keyword_party_counts(query_tokens, keyword_per_party_counts, total_term_freq, party)
        #print("total_term_freq", total_term_freq)
    return party_counts, keyword_per_party_counts
            
def whoosh_search(searcher, qr, ix):
    qp = MultifieldParser(["title", "text"], schema=ix.schema)
    #qp = QueryParser("title", schema=ix.schema)

    q = qp.parse(qr)
    results = searcher.search(q, terms = True)

    return results

def searchManifest(ix):
    qr = input("[METHOD_WHOOSH] " + QUERY_KW + ": ")
    while(qr != '-exit'):
        with ix.searcher() as searcher: #weighting=<class 'whoosh.scoring.BM25F'>
            start_time = time.time()

            results = whoosh_search(searcher, qr, ix)
            search_time = time.time() - start_time

            stats_start_time = time.time()
            
            ndocs = len(results.docs())
            party_counts, word_per_party = result_statistics(results, searcher, qr)
            stats_time = time.time() - stats_start_time

            print_stats(ndocs, results, party_counts, word_per_party)

        print("---search time: %s seconds ---" % (search_time))
        print("---stats time: %s seconds ---" % (stats_time))
        qr = input("\n" + "[METHOD_WHOOSH] " + QUERY_KW + ": ")

def search_lab(data):
    start_time = time.time()
    documents , i_index = search(data)
    print("---lab indexing time: %s seconds ---" % (time.time() - start_time))
    while True:
        qr = input("[METHOD_LAB] " + QUERY_KW + ": ")
        party_counts = dict()
        keyword_per_party_counts = dict()

        start_time = time.time()
        qr = nltk.word_tokenize(qr)
        results = dot_product_similarity(i_index, qr)
        search_time = time.time() - start_time
        
        start_search_time = time.time()
        ndocs = len(results)
        for doc in results:
            # party counts
            doc_row = doc[0]
            doc_score = doc[1]
            full_doc = data[ data["text"] == documents[doc_row-1] ]
            party = full_doc["party"].values[0]
            if party not in party_counts:
                party_counts[party] = 0
            party_counts[party] += 1
        
            #keyword_per_party_counts full_doc
            term_freq_text = dict()
            for kw in full_doc["text"].values[0].split():
                if kw.lower() not in term_freq_text:
                    term_freq_text[kw.lower()] = 0
                term_freq_text[kw.lower()] += 1
            #term_freq_title = Counter(dict(searcher.vector_as("frequency", docnum, "title")))
            total_term_freq = Counter(term_freq_text) # + term_freq_title

            #print("total_term_freq", total_term_freq)

            keyword_per_party_counts = keyword_party_counts(qr, keyword_per_party_counts, total_term_freq, party )
        stats_time = time.time() - start_search_time
        print_stats(ndocs, results, party_counts, keyword_per_party_counts)
        print("---search time: %s seconds ---" % (search_time))
        print("---stats time: %s seconds ---" % (stats_time))
        print()

def print_stats(ndocs, results, party_counts, word_per_party):
    # print stats
    if( ndocs <= 0 ):
        print("no results found") 
    else:
        print("top results:", results)
        print("number of total results:", ndocs)
        #print("scores:", results.top_n )
        print("number of manifestos from party: ",party_counts)
        print("keyword per party: ",word_per_party)

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        if arg =="lab":
            WHOOSH_SEARCH = False
        elif arg=="generate":
            FLAG_CREATE = True
        elif arg=="raw_data":
            CLEAN_DATA = False
    
    print("=============================================")
    print("PARAMS:")
    print("METHOD:", "whoosh" if WHOOSH_SEARCH  else "lab version")
    print("CREATED WHOOSH INDEX:", "yes" if FLAG_CREATE else "no (previously created)")
    print("DATA:", "cleaned data" if CLEAN_DATA else "raw data")
    print("=============================================")

    data = readData(PATH)

    if WHOOSH_SEARCH :
        if FLAG_CREATE:
            ix = createSchema(data)

        ix = index.open_dir("indexdir")
        searchManifest(ix)

    else:#search using lab 2
        search_lab(data)

