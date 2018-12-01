import spacy
import pandas as pd
import pickle

class ListInfo:
    def __init__(self,info):
        self.occorrence = [info]

    def addManifest(self, info):
        self.occorrence.append(info)

    def __str__(self):
        output = ''
        for x in self.occorrence:
            output += " " + str(x) + " "

        return output


class Info:

    def __init__(self, manID, manTitle, party, label):
        self.manID = manID
        self.manTitle = manTitle
        self.party = party
        self.label = label

    def __str__(self):
        return "MID: " + str(self.manID) + " MTITLE: " + str(self.manTitle) + " PARTY: " + str(
            self.party) + " LABEL: " + str(self.label)



def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



if __name__ == '__main__':
    nlp = spacy.load('pt')
    data = pd.read_csv("en_docs.csv", encoding="utf8")
    data.dropna()

    text = data["text"]

    entitiesDict = {}

    for i in range(len(text)):
        manifest = str(data['text'][i])
        manifest_id =  data['manifesto_id'][i]
        party = data['party'][i]
        title = data['title'][i]
        print("ID:" + str(i) + " :" + str(title))

        doc = nlp(manifest)

        for token in doc.ents:
            processedEntity = token.text.lower()

            if processedEntity in entitiesDict.keys():
                entitiesDict[processedEntity].addManifest(Info(manifest_id,title,party,token.label_))
            else:
                entitiesDict[processedEntity] = ListInfo(Info(manifest_id, title, party,token.label_))

    for k, v in entitiesDict.items():
        print(k, v)

    save_obj(entitiesDict,"DicionaryEntitiesSpacy")

    dicio = load_obj("DicionaryEntitiesSpacy")
    for k, v in dicio.items():
        print(k, v)