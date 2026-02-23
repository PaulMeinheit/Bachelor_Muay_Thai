import os
import pandas
import numpy as np
import Dataloader  
subjects = ["E1", "E2", "E3", "N1", "N2", "N3", "N4"]
movements = ["roundhouse", "teep","elbow","uppercut"]
bodyparts = [
    "FullBody_AngMom",
    "L_Hand",
    "R_Hand",
    "L_FA",
    "R_FA",
    "L_UA",
    "R_UA",
    "Head",
    "Trunk",
    "Pelvis",
    "L_Thigh",
    "R_Thigh",
    "L_Shank",
    "R_Shank",
    "L_Foot",
    "R_Foot",
]



for subject in subjects:
    for movement in movements:
        if subject == "E2" and movement == "roundhouse":
            continue
        rawDataPath = "newAngMom/" + subject + "/" + movement + ".csv"
        loadedAngMomData = pandas.read_csv(rawDataPath)
        full_body_angmom_x = loadedAngMomData["('FullBody_AngMom', 'X')"]
        full_body_angmom_y = loadedAngMomData["('FullBody_AngMom', 'Y')"]
        full_body_angmom_z = loadedAngMomData["('FullBody_AngMom', 'Z')"]  

        newAMACData = pandas.DataFrame()
        
        for bodypart in bodyparts:
            if bodypart == "FullBody_AngMom":
                continue
            
            col = (bodypart + "AMAC")
            scalarAMACBodyPart = []
            for time in range(len(loadedAngMomData)):
                angmomx = loadedAngMomData["('" + bodypart + "', 'X')"].iloc[time]
                angmomy = loadedAngMomData["('" + bodypart + "', 'Y')"].iloc[time]
                angmomz = loadedAngMomData["('" + bodypart + "', 'Z')"].iloc[time]
                HbodyPart = np.array([angmomx, angmomy, angmomz])
                HfullBody = np.array([full_body_angmom_x.iloc[time], full_body_angmom_y.iloc[time], full_body_angmom_z.iloc[time]])
                HfullBody_norm = np.linalg.norm(HfullBody, axis=0)
                
                scalarAMACBodyPart.append(np.vdot(HbodyPart, HfullBody) / HfullBody_norm if HfullBody_norm > 0 else 0)


            newAMACData[col] = scalarAMACBodyPart   
           
    
        out_dir = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/AMACscalar/{subject}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{movement}.csv")
        newAMACData.to_csv(out_path, index=False, header=True)
                    
                
            
