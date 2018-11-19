import os
import pandas as pd
from tabulate import tabulate

if __name__ == '__main__':

    bestPrecision = [0,0,0,0,0,0]
    bestPrecisionFile = ['','','','','','']
    bestRecall = [0,0,0,0,0,0]
    bestRecallFile = ['','','','','','']
    bestSupport = [0,0,0,0,0,0]
    bestSupportFile = ['','','','','','']
    bestF1_Score = [0,0,0,0,0,0]
    bestF1_ScoreFile = ['','','','','','']

    bestPrecisionOverall = 0
    bestPrecisionOverallFile = ''
    bestRecallOverall = 0
    bestRecallOverallFile = ''
    bestSupportOverall = 0
    bestSupportOverallFile = ''
    bestF1_ScoreOverall = 0
    bestF1_ScoreOverallFile = ''

    for file in os.listdir("results"):

        # (0.359*a)+(0.256*b)+(0.205*c)+(0.087*d)+(0.073*e)+(0.016*f)
        df = pd.read_csv("results/"+file)

        for i in range(0,6):
            if bestF1_Score[i] < df["f1_score"][i]:
                bestF1_Score[i] = df["f1_score"][i]
                bestF1_ScoreFile[i]=file
            if bestPrecision[i] < df["precision"][i]:
                bestPrecision[i] = df["precision"][i]
                bestPrecisionFile[i] = file
            if bestRecall[i] < df["recall"][i]:
                bestRecall[i] = df["recall"][i]
                bestRecallFile[i] = file
            if bestSupport[i] < df["support"][i]:
                bestSupport[i] = df["support"][i]
                bestSupportFile[i] = file

        currPrecision = 0
        currRecall = 0
        currSupport = 0
        currF1_Score = 0

        for idx,value in enumerate([0.359,0.256,0.205,0.087,0.073,0.016]):
            currF1_Score += (value * df["f1_score"][idx])
            currPrecision += (value * df["precision"][idx])
            currRecall += (value * df["recall"][idx])
            currSupport += (value * df["support"][idx])

        if currPrecision > bestPrecisionOverall:
            bestPrecisionOverall=currPrecision
            bestPrecisionOverallFile = file
        if currRecall > bestRecallOverall:
            bestRecallOverall=currRecall
            bestRecallOverallFile = file
        if currSupport > bestSupportOverall:
            bestSupportOverall=currSupport
            bestSupportOverallFile = file
        if currF1_Score > bestF1_ScoreOverall:
            bestF1_ScoreOverall=currF1_Score
            bestF1_ScoreOverallFile = file

    bestPrecision.insert(0,"Precision")
    bestPrecisionFile.insert(0, "Precision")
    bestRecall.insert(0, "Recall")
    bestRecallFile.insert(0, "Recall")
    bestSupport.insert(0, "Support")
    bestSupportFile.insert(0, "Support")
    bestF1_Score.insert(0, "F1_SCORE")
    bestF1_ScoreFile.insert(0, "F1_SCORE")

    tableSpecific = [["","Class0","Class1","Class2","Class3","Class4","Class5"],
                     bestPrecision,bestPrecisionFile,bestRecall,bestRecallFile,
                     bestSupport,bestSupportFile,bestF1_Score,bestF1_ScoreFile]

    tableGeneral = [ ["Precision Best","Recall Best","Support Best","F1_Score Best"],
                     [bestPrecisionOverall,bestRecallOverall,bestSupportOverall,bestF1_ScoreOverall],
                    [bestPrecisionOverallFile,bestRecallOverallFile,bestSupportOverallFile,bestF1_ScoreOverallFile]]

    print(tabulate(tableSpecific))
    print(tabulate(tableGeneral))
