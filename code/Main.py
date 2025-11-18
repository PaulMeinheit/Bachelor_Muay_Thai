import Segmenter
filePath = "Sub01_Teep/JointAngles.txt"
resultPath = "results"
segmentframes = [0,100,200,300,400,500,600,700,800,900]
trialName = "Sub01_Teep_finishedJointAngles"
Segmenter.segmentData(filePath,resultPath,segmentframes,trialName)

