import Slicer
import os
import Scaler
import Averager
import Dataloader
import matplotlib.pyplot as plt

trialPath ="/N2/roundhouse"
dataPath = "Raw_Data" + trialPath
SlicedResultsPath = "processed_Data/" +trialPath + "/sliced"
scaledResultPath = "processed_Data/" + trialPath + "/scaled"

# Frame numbers for each segment phase boundary
segmentLiftFrames = [259,541,871,1130,1412,1751,2065,2431,2718,2972]
segmentImpactFrames = [295,575,905,1165,1445,1789,2098,2469,2753,3009]
segmentFootDownFrames = [395,770,1070,1200,1550,1872,2185,2566,2845,3100]

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
     Scaler.scaleDirectoryToFourPhases(os.path.join(SlicedResultsPath, directory), Segments, scaledResultPath, directory)
    
#Averager.run(scaled_results_dir=scaledResultPath, output_dir="AveragedResults")



