import pandas
import Dataloader
import os


def sliceData(dataPath, outputPath, segemnt_frames):

    data = Dataloader.loadData(dataPath)
    internalFolder = os.path.join(outputPath,dataPath)
    os.makedirs(internalFolder)
    ##slicing the data into the 10 respective trials
    for  i in range(len(segemnt_frames)-2):
        temp = data.iloc[segemnt_frames[i]:segemnt_frames[i+1]]
        tempname ="sliced" + str(i) + ".csv"
        open(os.path.join(internalFolder, tempname), "x")
        
        temp.to_csv(os.path.join(internalFolder, tempname), sep=",", index = False)


