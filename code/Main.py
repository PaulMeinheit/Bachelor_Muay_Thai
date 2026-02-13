import Slicer
import os
import Scaler
import Averager
import matplotlib.pyplot as plt

trialPath ="/N4/roundhouse"
dataPath = "Raw_Data" + trialPath
SlicedResultsPath = "processed_Data/" +trialPath + "/sliced"
scaledResultPath = "processed_Data/" + trialPath + "/scaled"

# Frame numbers for each segment phase boundary
segmentLiftFrames = [415,754,1038,1322,1608,1892,2230,2520,2799,3300]
segmentImpactFrames = [458,795,1080,1366,1649,1935,2277,2563,2842,3345]
segmentFootDownFrames = [504,845,1136,1424,1712,2005,2332,2621,2943,3400]

segmentBeginFrame = Slicer.calcBeginnframe(segmentLiftFrames)

for file in os.listdir(dataPath):
   Slicer.sliceData(os.path.join(dataPath, file), SlicedResultsPath, segmentBeginFrame)

grfPath = os.path.join(SlicedResultsPath + "/CoG_Velocity")
Segments = Slicer.findTeepSegments(
    grfPath,
    segmentLiftFrames,
    segmentImpactFrames,
    segmentFootDownFrames,
    
)

for directory in sorted(os.listdir(SlicedResultsPath)):
     print(directory)
     dict = Scaler.load_csvs_from_dir(os.path.join(SlicedResultsPath, directory))
     Scaler.scaleToFourPhases(dict, Segments, scaledResultPath, directory)

#Averager.run(scaled_results_dir=scaledResultPath, output_dir="AveragedResults")



