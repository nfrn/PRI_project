import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import whoosh.index as index
import whoosh.query as query
from whoosh.fields import Schema, TEXT, STORED, BOOLEAN, NUMERIC, DATETIME,ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import *



PATH="en_docs.csv"
FLAG_CREATE = False

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
            if( not all(x==row[potential_error_column][0] for x in row[potential_error_column])):
                print("not cleaned properly row:", index, "| column name:", potential_error_column)

        #compensate for poor joining becaus ims stewpid
        for column in ("party", "title"):
            row[column] = row[column][0]

    cleaned.to_csv("test.csv")
    #print(data.columns.values)
    #print(len(data.columns.values))
    #print(data.head(1))
    return cleaned

def createSchema(data):
    print("Create schema")
    schema = Schema(number=NUMERIC,
                    text=TEXT(stored = True),
                    party=NUMERIC(stored = True),
                    title=TEXT(stored = True))

    ix = create_in("indexdir", schema)
    writer = ix.writer()
    print("Writing in schema", end='')
    for index, row in data.iterrows():
        if(int(index)%500==0):
            print(".", end='', flush=True)

        writer.add_document(number=row[1],
                    text=row["text"],
                    party=row["party"],
                    title=row["title"])
    print(" done")
    print("commiting schema")
    writer.commit()
    return ix
def searchManifest(ix):
    qr = input("query: ")
    while(qr != '-exit'):
        with ix.searcher() as searcher:
            qp = MultifieldParser(["title", "text"], schema=ix.schema)
            #qp = QueryParser("title", schema=ix.schema)
            q = qp.parse(qr)
            results = searcher.search(q)

            print("results:", results)
            print("nresults:", len(results.docs()))
            print("scores:", results.top_n )
            
            for docnum in results.docs():
                pass
                #print(searcher.stored_fields(docnum)["title"])
        qr = input("\nquery: ")


if __name__ == '__main__':
    data = readData()

    if (len(sys.argv) > 1):
        if(sys.argv[1]=="generate"):
            FLAG_CREATE= True

    if FLAG_CREATE:
        ix = createSchema(data)
    else:
        ix = index.open_dir("indexdir")

    searchManifest(ix)

