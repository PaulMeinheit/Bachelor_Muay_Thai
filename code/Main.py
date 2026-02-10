import Slicer
import os
import Scaler
import Averager
import matplotlib.pyplot as plt

dirPath = "Sub01_Teep"
SlicedResultsPath = dirPath + "/sliced"
scaledResultPath = dirPath + "/scaled"

# Frame numbers for each segment phase boundary
segmentLiftFrames = [645, 1162, 1438, 1769, 2063, 2391, 2668, 2942, 3193, 3475]
segmentImpactFrames = [645, 1162, 1438, 1769, 2063, 2391, 2668, 2942, 3193, 3475]
segmentFootDownFrames = [750, 1250, 1530, 1850, 2150, 2475, 2750, 3025, 3275, 3550]
segmentBeginFrame = Slicer.calcBeginnframe(segmentLiftFrames)

for file in os.listdir(dirPath):
   Slicer.sliceData(os.path.join(dirPath, file), SlicedResultsPath, segmentBeginFrame)

grfPath = os.path.join(SlicedResultsPath, dirPath) + "/GRF_filtered12hz_resampled120hz.txt"
Segments = Slicer.findTeepSegments(
    grfPath,
    segmentLiftFrames,
    segmentImpactFrames,
    segmentFootDownFrames,
    
)

for directory in sorted(os.listdir(os.path.join(SlicedResultsPath, dirPath))):
     print(directory)
     dict = Scaler.load_csvs_from_dir(os.path.join(SlicedResultsPath, dirPath, directory))
     Scaler.scaleToFourPhases(dict, Segments, scaledResultPath, directory)

#Averager.run(scaled_results_dir=scaledResultPath, output_dir="AveragedResults")



