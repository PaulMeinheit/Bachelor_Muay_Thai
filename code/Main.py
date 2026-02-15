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
segmentLiftFrames = [519,989,1245,1563,1855,2335,2621,2937,3230,3507]
segmentImpactFrames = [560,1024,1280,1595,1887,2369,2656,2969,3262,3540]
segmentFootDownFrames = [760,1104,1372,1667,1959,2453,2762,3038,3331,3617]

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



