import pandas
import Dataloader
import os
from pathlib import Path


def sliceData(dataPath, outputPath, segmentBeginnframes):

    stem = Path(dataPath).stem
    
    data = Dataloader.loadData(dataPath)
    internalFolder = os.path.join(outputPath,stem)

    
    os.makedirs(internalFolder)
    ##slicing the data into the respective trials
    for  i in range(len(segmentBeginnframes)-1):
        temp = data.iloc[segmentBeginnframes[i]:segmentBeginnframes[i+1]]
        tempname ="sliced" + str(i) + ".csv"
        open(os.path.join(internalFolder, tempname), "x")
        temp.to_csv(os.path.join(internalFolder, tempname), sep=",", index = False)


def calcBeginnframe(liftOffFrame):
    return [frame - 40 for frame in liftOffFrame]

def findTeepSegments(segmentBeginnframe, groundReactionPath, liftOffFrames, impactFrames, footDownFrames):
    """
    Build segment data from manually-provided frame numbers.
    
    Args:
        groundReactionPath: path to ground reaction force files (for trial count)
        liftOffFrames: array of lift-off frame numbers (phase 0->1 transition)
        impactFrames: array of impact/max extension frame numbers (phase 1->2 transition)
        footDownFrames: array of foot-down frame numbers (phase 2->3 transition)
        trialEndFrames: array of trial end frame numbers (phase 3 end)
    
    Returns:
        segmentData: list of [phase0_start, phase0_end, phase1_end, phase2_end, phase3_end]
    """
    segmentData = []
    counter = 0
    for file in sorted(os.listdir(groundReactionPath)):
        data = pandas.read_csv(os.path.join(groundReactionPath, file), header=[0, 1])
        
        # Build segment as [phase0_start, phase0_end, phase1_end, phase2_end, phase3_end]
        # Phase 0: 0 to liftOffFrame
        # Phase 1: liftOffFrame to impactFrame
        # Phase 2: impactFrame to footDownFrame
        # Phase 3: footDownFrame to trialEndFrame
        segmentData.append([
            0,
            liftOffFrames[counter]-segmentBeginnframe[counter],
            impactFrames[counter]-segmentBeginnframe[counter],
            footDownFrames[counter]-segmentBeginnframe[counter],
            footDownFrames[counter] +50  - segmentBeginnframe[counter]  # trial end is 50 frames after foot down
        ])
        counter += 1
    
    return segmentData
