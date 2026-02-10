import pandas
import Dataloader
import os


def sliceData(dataPath, outputPath, segemnt_frames):

    data = Dataloader.loadData(dataPath)
    internalFolder = os.path.join(outputPath,dataPath)
    os.makedirs(internalFolder)
    ##slicing the data into the 10 respective trials
    for  i in range(len(segemnt_frames)-2):
        temp = data.iloc[segemnt_frames[i]:segemnt_frames[i+1]]
        tempname ="sliced" + str(i) + ".csv"
        open(os.path.join(internalFolder, tempname), "x")
        
        temp.to_csv(os.path.join(internalFolder, tempname), sep=",", index = False)

def calcBeginnframe(liftOffFrame):
    beginnframe = []
    for i in liftOffFrame.size():
        beginnframe[i] = (liftOffFrame[i]- 100)
    return beginnframe


def findTeepSegments(groundReactionPath, liftOffFrames, impactFrames, footDownFrames):
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
            liftOffFrames[counter],
            impactFrames[counter],
            footDownFrames[counter],
            len(data) - 1  # trial end is last frame in data
        ])
        counter += 1
    
    return segmentData
