import Slicer
import os
import SegmentFinder

dirPath = "Sub01_Teep"
intermediaryResultPath = "results"
segmentBeginFrames = [520,1042,1321,1662,1995,2325,2600,2873,3125,3400]
segmentImpactFrames = [645,1162,1438,1769,2063,2391,2668,2942,3193]

for file in os.listdir(dirPath):
   Slicer.sliceData(os.path.join(dirPath,file),intermediaryResultPath,segmentBeginFrames)

grfPath = "results/" + dirPath + "/GRF_filtered12hz_resampled120hz.txt"
Segments = SegmentFinder.findTeepSegments(grfPath, segmentImpactFrames,segmentBeginFrames)
print(Segments)

