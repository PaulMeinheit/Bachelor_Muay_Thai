##Recall the five phases and points of segmentation:
##
##|       Preparation             |          Extension              |                Retraction                    |            Termination               |
##0 [stand still] ------> 1 [lift kick foot] 0 ------> 1 [strike / max knee extension] 0 ----------------> 1 [kick foot on gnd] 0 -------------> [Kick ends]ta)
import pandas
import os
def findTeepSegments(groundReactionPath,impactFrames,segmentBeginFrames):
    segmentData = []
    counter = 0
    for file in sorted(os.listdir(groundReactionPath)):
        data = pandas.read_csv(os.path.join(groundReactionPath,file),header=[0, 1])
        
        Liftthreshold = 8
        col = ("FP1", "Z")
        mask = data[col] < Liftthreshold
        liftOffFrame = mask.idxmax() if mask.any() else None
        print(liftOffFrame)
        Downthreshold = 15
        # liftoff is assumed to always be found — locate its integer
        # position in the DataFrame index and search only rows after it
        subset = data.iloc[liftOffFrame+1:]
        mask = subset[col] > Downthreshold
        # per assumption a foot-down will be found; take the first True
        footDownFrame = mask.idxmax()

        segmentData.append([0,liftOffFrame,impactFrames[counter] - segmentBeginFrames[counter], footDownFrame,data.shape[0]])
        counter += 1
    return segmentData