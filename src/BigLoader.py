import os
from os import path

import pandas as pd


subjects = ["E1", "E2", "E3", "N1", "N2", "N3", "N4"]


movements = ["roundhouse", "teep"]
FailExclusions = {
    ("E1", "teep") : ["averaged.csv"],
    ("E1", "roundhouse") : ["averaged.csv"],
    ("E1", "elbow") : ["averaged.csv"],
    ("E1", "uppercut") : ["averaged.csv"],

    ("E2", "teep") : ["averaged.csv"],
    ("E2", "roundhouse") : ["averaged.csv"],
    ("E2", "elbow") : ["averaged.csv"],
    ("E2", "uppercut") : ["averaged.csv"],

    ("E3", "teep") : ["averaged.csv"],
    ("E3", "roundhouse") : ["averaged.csv"],
    ("E3", "elbow") : ["averaged.csv"],
    ("E3", "uppercut") : ["averaged.csv"],

    ("N1", "teep") : ["scaled4.csv","averaged.csv"],
    ("N1", "roundhouse") : ["scaled1.csv","averaged.csv"],
    ("N1", "elbow") : ["scaled4.csv","averaged.csv"],
    ("N1", "uppercut") : ["scaled1.csv","averaged.csv"],

    ("N2", "teep") : ["averaged.csv"],
    ("N2", "roundhouse") : ["scaled2.csv", "scaled3.csv", "scaled4.csv", "scaled5.csv", "averaged.csv"],
    ("N2", "elbow") : ["averaged.csv"],
    ("N2", "uppercut") : ["scaled2.csv", "scaled3.csv", "scaled4.csv", "scaled5.csv", "averaged.csv"],

    ("N3", "teep") : ["scaled8.csv","averaged.csv"],
    ("N3", "roundhouse") : ["scaled1.csv","averaged.csv"],
    ("N3", "elbow") : ["scaled8.csv","averaged.csv"],
    ("N3", "uppercut") : ["scaled1.csv","averaged.csv"],

    ("N4", "teep") : ["averaged.csv"],
    ("N4", "roundhouse") : ["averaged.csv"],
    ("N4", "elbow") : ["averaged.csv"],
    ("N4", "uppercut") : ["averaged.csv"],
}


def loadallspecified(subjects, movement) -> pd.DataFrame:
    BigData = pd.DataFrame()
    subjectDataList = []
    
    for subject in subjects:
        
        subjectData = pd.DataFrame()
        
        if subject == "E2" and movement == "roundhouse":
            continue
        if subject == "N1" and movement == "teep":
            continue
        path = "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/scaled_Data/processed_AngMomData/"+subject+"/"+movement+"/scaled/AMACscalar"
        if not os.path.exists(path):
            continue
        files = [f for f in os.listdir(path) if f not in FailExclusions[(subject, movement)]]
        if not files:
	        raise ValueError("No scaled* files found in directory.")
        dflist = []
        keyList = []
        for file in sorted(files):
            i = 0
            dflist.append(pd.read_csv(os.path.join(path, file)))
            keyList.append(str(i))
            i += 1
        subjectData = pd.concat(dflist, axis=1, keys=keyList)
        subjectDataList.append(subjectData)
    BigData = pd.concat(subjectDataList, axis=1, keys=subjects)
  
    return BigData   


loadallspecified(["E1", "E2", "E3"], "elbow")