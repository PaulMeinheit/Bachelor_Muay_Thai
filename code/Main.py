import Slicer
import os
import SegmentFinder

dirPath = "Sub01_Teep"
intermediaryResultPath = "results"
segmentBeginFrames = [0,100,200,300,400,500,600,700,800,900,1000]
segmentImpactFrames = [50,150,250,350,450,550,650,750,850,950,1050]

for file in os.listdir(dirPath):
    Slicer.sliceData(os.path.join(dirPath,file),intermediaryResultPath,segmentBeginFrames)

grfPath = "results/" + dirPath + "/GRF_filtered12hz_resampled120hz.txt"
SegmentFinder.findSegments(grfPath, segmentImpactFrames)


