
##Recall the five phases and points of segmentation:
##
##|       Preparation             |          Extension              |                Retraction                    |            Termination               |
##0 [stand still] ------> 1 [lift kick foot] 0 ------> 1 [strike / max knee extension] 0 ----------------> 1 [kick foot on gnd] 0 -------------> [Kick ends]ta)
import pandas
import os
def findTeepSegments(groundReactionPath,impactFrames):
    segmentData = [[]]
    counter = 0
    for file in os.listdir(groundReactionPath):
        data = pandas.read_csv(os.path.join(groundReactionPath,file))
    #lift off frame
    segmentData[counter][1] = impactFrames[counter]
    #TODO
    #impact frame
    segmentData[counter][1] = impactFrames[counter]
    #foot down frame
    #TODO
    segmentData[counter][1] = impactFrames[counter]
    counter =+ 1
    return segmentData