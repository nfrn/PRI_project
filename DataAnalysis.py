import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = "pt_docs_clean.csv"

# Description
# Index(['text','manifesto_id', 'party', 'date', 'title'], dtype='object')

# 0 'Portuguese Communist Party'
# 1 'Socialist Party'
# 2 "Ecologist Party ‘The Greens'"
# 3 'Social Democratic Party'
# 4 'Left Bloc'
# 5 'Social Democratic Center-Popular Party'

def visualize(df,string):
    # Visualize class imbalance
    print("Visualize class imbalance" + string)
    classes = df.groupby('party')
    classes.agg(np.size).plot.bar(rot=0)
    plt.title('Number of texts per party' + string)
    plt.show(block=False)

    # Visualize len of texts
    print("Visualize texts imbalance"+ string)
    df['text_length'] = df['text'].str.split().str.len()
    df['id'] = range(1, len(df) + 1)
    df.plot.scatter(x='id',y='text_length')
    plt.title('Size of each Text' + string)
    plt.show(block=False)

    # Visualize mean len of texts per party
    print("Visualize mean text len per party"+ string)
    classes = df.groupby('party')
    classes.agg({'text_length': 'mean'}).plot.bar(rot=0)
    plt.title('Mean size of texts per party' + string)
    plt.show(block=False)

if __name__ == '__main__':
    df = pd.read_csv(PATH, encoding="utf8", header=0)
    df = df.loc[:, ['text', 'party']]

    df['party'] = df['party'].map({
        'Portuguese Communist Party': 'PCP',
        'Socialist Party': 'PS',
        'Social Democratic Party': 'PSD',
        'Left Bloc': 'BE',
        'Social Democratic Center-Popular Party': 'CDS',
        "Ecologist Party ‘The Greens'": 'VERDES'
    })

    print("Number of texts:" + str(len(df.index)))
    visualize(df," before")


    df = df[df['text_length'] <= 150]
    print("Number of texts:" + str(len(df.index)))
    visualize(df, " after adjustment")

    plt.show(block=True)





