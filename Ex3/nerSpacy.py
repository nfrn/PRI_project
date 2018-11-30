import itertools
import spacy
import csv
#pip install -U spacy
#python -m spacy download pt
from Info import Info
from Ex2.textRepresentation import process_data
import pandas as pd
PATH = "pt_docs_clean.csv"

def readData():
    print("Start reading data...")
    data = pd.read_csv(PATH, encoding="utf8")
    return data



if __name__ == '__main__':
    nlp = spacy.load('pt')

    data = readData()
    
    outputData ={}
    i=0
    for i in range(1):
        manifest = data['text'][i]
        print(manifest)
        manifest_id =  data['manifesto_id'][i]
        party = data['party'][i]
        title = data['title'][i]
        i+=1
        print("ID:" + str(i) + " :" + str(title))
        doc = nlp(manifest)

        for token in doc.ents:
            index = hash(token.text)
            if token in outputData.keys():
                outputData[index].addWord(token.text)
                outputData[index].addManifest(manifest_id,title,party)
            else:
                info = Info(token.label_)
                outputData[index] = info
                outputData[index].addWord(token.text)
                outputData[index].addManifest(manifest_id, title, party)

    for value in outputData.values():
        print(value)
