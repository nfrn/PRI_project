import pandas as pd
import os
from sklearn.model_selection import train_test_split
import whoosh.index as index
import whoosh.query as query
from whoosh.fields import Schema, TEXT, STORED, BOOLEAN, NUMERIC, DATETIME,ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import *



PATH="en_docs.csv"
FLAG_CREATE = True

if not os.path.exists("indexdir"):
   os.mkdir("indexdir")

def readData():
    data = pd.read_csv(PATH, encoding="utf8")
    data.fillna("0", inplace=True)
    return data

def createSchema(data):
    print("Create schema")
    schema = Schema(number=NUMERIC,
                    text=TEXT,
                    cmp_code=NUMERIC,
                    eu_code=NUMERIC,
                    pos=NUMERIC,
                    manifesto_id=TEXT,
                    party=NUMERIC,
                    date=NUMERIC,
                    language=TEXT,
                    source=TEXT,
                    has_eu_code=BOOLEAN,
                    is_primary_doc=BOOLEAN,
                    may_contradict_core_dataset=BOOLEAN,
                    url_original=TEXT,
                    md5sum_text=TEXT,
                    md5sum_original=TEXT,
                    annotations=BOOLEAN,
                    handbook=NUMERIC,
                    is_copy_of=ID,
                    title=TEXT,
                    id=TEXT)

    ix = create_in("indexdir", schema)
    writer = ix.writer()
    print("Write in schema")
    for index, row in data.iterrows():
        print(index)
        writer.add_document(number=row[0],
                            text=row[1],
                            cmp_code=row[2],
                            eu_code=row[3],
                            pos=row[4],
                            manifesto_id=row[5],
                            party=row[6],
                            date=row[7],
                            language=row[8],
                            source=row[9],
                            has_eu_code=row[10],
                            is_primary_doc=row[11],
                            may_contradict_core_dataset=row[12],
                            url_original=row[13],
                            md5sum_text=row[14],
                            md5sum_original=row[15],
                            annotations=row[16],
                            handbook=row[17],
                            is_copy_of=row[18],
                            title=row[19],
                            id=row[20])

    print()
    writer.commit()
    return ix
def searchManifest(ix,qr):

    print("Search query")
    results = ix.searcher().search(query.Every())
    print("Unordered results")
    print(results)

if __name__ == '__main__':
    print("Start reading data")
    data = readData()

    if FLAG_CREATE:
        ix = createSchema(data)
    else:
        ix = index.open_dir("indexdir")

    searchManifest(ix,u"world")

