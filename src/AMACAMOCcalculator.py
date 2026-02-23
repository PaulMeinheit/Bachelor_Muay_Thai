import os
import pandas
subjects = ["E1", "E2", "E3", "N1", "N2", "N3", "N4"]
movements = ["roundhouse", "teep","elbow","uppercut"]

for subject in subjects:
    for movement in movements:
        if subject == "E2" and movement == "roundhouse":
            continue
        rawDataPath = "newAngMom/" + subject + "/" + movement + ".csv"
        loadedAngMomData = pandas.read_csv(rawDataPath)
        full_body_angmom = loadedAngMomData["FullBody_AngMom"]
        print(full_body_angmom)