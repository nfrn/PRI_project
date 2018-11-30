import nltk
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in' \
     'the mobile phone market and ordered the company to alter its practices'
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

if __name__ == '__main__':
    nltk.download()
    sent = preprocess(ex)
    print(sent)

    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)

    iob_tagged = tree2conlltags(cs)
    pprint(iob_tagged)

    ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
    print(ne_tree)