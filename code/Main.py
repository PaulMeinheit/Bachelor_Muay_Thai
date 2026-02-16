import Slicer
import os
import Scaler
import Averager
import Dataloader
import matplotlib.pyplot as plt

trialPath ="/E1/teep"
dataPath = "Raw_Data" + trialPath
SlicedResultsPath = "processed_Data/" +trialPath + "/sliced"
scaledResultPath = "processed_Data/" + trialPath + "/scaled"

# Frame numbers for each segment phase boundary
segmentLiftFrames = [456,1082,1458,1870,2328,2723,3136,3578,4025,4469,4896]
segmentImpactFrames = [545,1133,1507,1912,2377,2769,3182,3628,4071,4515,4940]
segmentFootDownFrames = [610,1176,1553,1956,2424,2815,3229,3674,4119,4569,5011]


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



