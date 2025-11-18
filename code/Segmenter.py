import pandas
import Dataloader
import os


def segmentData(dataPath, outputPath, segemnt_frames,trialName):

    data = Dataloader.loadData(dataPath)
    os.mkdir(outputPath)
    
    ##slicing the data into the 10 respective trials
    for  i in range(9):
        temp = data.iloc[segemnt_frames[i]:segemnt_frames[i+1]]
        tempname =trialName + str(i) + ".csv"
        f = open(os.path.join(outputPath, tempname), "x")
        
        temp.to_csv(os.path.join(outputPath, tempname), sep=",", index = False)


