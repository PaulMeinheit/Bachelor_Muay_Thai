import Slicer
import os

dirPath = "Sub01_Teep"
resultPath = "results"
segmentframes = [0,100,200,300,400,500,600,700,800,900]


os.mkdir(resultPath)
for file in os.listdir(dirPath):
    Slicer.sliceData(os.path.join(dirPath,file),resultPath,segmentframes)

