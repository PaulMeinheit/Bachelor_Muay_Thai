import Slicer
import os
import Scaler
import Averager
import matplotlib.pyplot as plt

trialPath ="/E1/roundhouse"
dataPath = "Raw_Data" + trialPath
SlicedResultsPath = "processed_Data/" +trialPath + "/sliced"
scaledResultPath = "processed_Data/" + trialPath + "/scaled"

# Frame numbers for each segment phase boundary
segmentLiftFrames = [703,1075,1542,2072,2667,3132,3642,4153,4521]
segmentImpactFrames = [734,1105,1576,2106,2700,3163,3675,4185,4548]
segmentFootDownFrames = [806,1176,1635,2174,2753,3235,3741,4238,4604]

segmentBeginFrame = Slicer.calcBeginnframe(segmentLiftFrames)

for file in os.listdir(dataPath):
   Slicer.sliceData(os.path.join(dataPath, file), SlicedResultsPath, segmentBeginFrame)

grfPath = os.path.join(SlicedResultsPath + "/CoG_Velocity")
Segments = Slicer.findTeepSegments(
    segmentBeginFrame,
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



