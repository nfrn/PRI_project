import pickle
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import pandas as pd


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

    def __init__(self, manID, manTitle, party):
        self.manID = manID
        self.manTitle = manTitle
        self.party = party

    def __str__(self):
        return "MID: " + str(self.manID) + " MTITLE: " + str(self.manTitle) + " PARTY: " + str(
            self.party)



def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_continuous_chunks(text):
    #https://stackoverflow.com/questions/31836058/nltk-named-entity-recognition-to-a-python-list
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
           if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
           elif current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity)
                            current_chunk = []
           else:
                    continue
    return continuous_chunk

if __name__ == '__main__':
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

        entities = get_continuous_chunks(manifest)

        for entity in entities:
            processedEntity = entity.lower()

            if processedEntity in entitiesDict.keys():
                entitiesDict[processedEntity].addManifest(Info(manifest_id,title,party))
            else:
                entitiesDict[processedEntity] = ListInfo(Info(manifest_id, title, party))

    for k, v in entitiesDict.items():
        print(k, v)

    save_obj(entitiesDict,"DicionaryEntities")

    dicio = load_obj("DicionaryEntities")
    for k, v in dicio.items():
        print(k, v)