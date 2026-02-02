import Slicer
import os
import SegmentFinder
import Scaler
import Averager
import matplotlib.pyplot as plt
dirPath = "Sub01_Teep"
SlicedResultsPath = "slicedResults"
scaledResultPath = "scaledResults"
segmentBeginFrames = [520,1042,1321,1662,1995,2325,2600,2873,3125,3400]
segmentLiftFrames =[]
segmentImpactFrames = [645,1162,1438,1769,2063,2391,2668,2942,3193]

for file in os.listdir(dirPath):
   Slicer.sliceData(os.path.join(dirPath,file),SlicedResultsPath,segmentBeginFrames)

grfPath = os.path.join(SlicedResultsPath, dirPath) + "/GRF_filtered12hz_resampled120hz.txt"
Segments = SegmentFinder.findTeepSegments(grfPath, segmentImpactFrames,segmentBeginFrames)

for directory in sorted(os.listdir(os.path.join(SlicedResultsPath, dirPath))):
     print(directory)
     dict = Scaler.load_csvs_from_dir(os.path.join(SlicedResultsPath, dirPath, directory))
     Scaler.scaleToFourPhases(dict, Segments, scaledResultPath,directory)
Averager.run(scaled_results_dir=scaledResultPath, output_dir="AveragedResults")

