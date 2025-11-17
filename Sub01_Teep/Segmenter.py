import pandas
import Dataloader


def segmentData(dataPath, segemnt_frames):
    data = Dataloader.loadData(dataPath)
    segments = []
    ##slicing the data into the 10 respecive trials
    for  i in range(9):
        segments[i] = data.iloc(segemnt_frames[segemnt_frames[i]:segemnt_frames[i+1]])

    return segments


