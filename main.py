import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def quicksort(array, low, high):
    if (low < high) :
        pivot_location = partition(array,low,high)
        quicksort(array, low, pivot_location)
        quicksort(array, pivot_location + 1, high)

def partition(array,low,high):
    pivot = array[low]
    leftwall = low
    for i in range(low+1,high+1):
        if array[i] < pivot :
            leftwall += 1
            array[leftwall],  array[i] = array[i], array[leftwall]

    array[leftwall], array[low] = array[low], array[leftwall]

    return leftwall

def readarraynumeric(path):
    f = open(path, encoding="utf8")
    array = []
    for line in f:
        array.append(int(line.split()[0]))
    f.close()

    print(array)
    quicksort(array,0,len(array)-1)
    print(array)

def readarraytext(path):
    f = open(path, encoding="utf8")
    thisdict = {}
    for line in f:
        line = line.split()
        for word in line:
            if word in thisdict.keys():
                thisdict[word] += 1
            else:
                thisdict[word] = 1
    f.close()
    return thisdict

def readarraytextcompare(path1, path2):
    dic1 =readarraytext(path1)
    dic2 = readarraytext(path2)
    i = 0
    for key in dic1.keys():
        if key in dic2.keys():
            i+=1
    print(i)


def useNTLK():
    f = open('text13.txt', encoding="utf8")
    doc = f.read()
    processed = nltk.pos_tag(nltk.word_tokenize(doc))
    print(Counter([j for i, j in processed]))


def compareDocs(path1, path2):
    f = open(path1, encoding="utf8")
    doc1 = f.readlines()
    f.close()

    f = open(path2, encoding="utf8")
    doc2 = f.readlines()
    f.close()

    bow_transformer = TfidfVectorizer()
    bow_transformer.fit(doc1,doc2)

    data1 = bow_transformer.transform(doc1)
    data2 = bow_transformer.transform(doc2)

    print(data1)
    print("_")
    print(data2)

    print(cosine_similarity(data1, data2))



if __name__ == '__main__':
    readarraynumeric('text12.txt')
    print(readarraytext('text13.txt'))
    readarraytextcompare('text13.txt','text14.txt')
    useNTLK()
    compareDocs('text13.txt','text14.txt')
