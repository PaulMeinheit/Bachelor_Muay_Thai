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



def calculate_AMAC():
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
            
        
            out_dir = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"AMACscalar.csv")
            newAMACData.to_csv(out_path, index=False, header=True)
                        
                    
def calculateAMACVectors():

    for subject in subjects:
        for movement in movements:
            if subject == "E2" and movement == "roundhouse":
                continue
            rawDataPath = "newAngMom/" + subject + "/" + movement + ".csv"
            rawAMACpath = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}/AMACscalar.csv"
            loadedAngMomData = pandas.read_csv(rawDataPath)
            loadedAMACData = pandas.read_csv(rawAMACpath)

        
            full_body_angmom_x = loadedAngMomData["('FullBody_AngMom', 'X')"]
            full_body_angmom_y = loadedAngMomData["('FullBody_AngMom', 'Y')"]
            full_body_angmom_z = loadedAngMomData["('FullBody_AngMom', 'Z')"]

            newAMACVectors = pandas.DataFrame()
            for bodypart in bodyparts:
                if bodypart == "FullBody_AngMom":
                    continue
                AMAC_x = []
                AMAC_y = []
                AMAC_z = []

                AMACBodyPart = loadedAMACData[bodypart + "AMAC"]

                AMAC_x = []
                AMAC_y = []
                AMAC_z = []
                    
                for time in range(len(loadedAMACData)):
                    AMAC_x.append(AMACBodyPart.iloc[time] * full_body_angmom_x.iloc[time])
                    AMAC_y.append(AMACBodyPart.iloc[time] * full_body_angmom_y.iloc[time])
                    AMAC_z.append(AMACBodyPart.iloc[time] * full_body_angmom_z.iloc[time])

                for idx, axis in enumerate(['X', 'Y', 'Z']):
                        col = (bodypart + "AMACV" + axis)
                        if col not in newAMACVectors.columns:
                            newAMACVectors[col] = np.nan
                        if axis == 'X':
                            newAMACVectors[col] = AMAC_x
                        elif axis == 'Y':
                            newAMACVectors[col] = AMAC_y
                        elif axis == 'Z':
                            newAMACVectors[col] = AMAC_z

            out_dir = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"AMACVector.csv")
            newAMACVectors.to_csv(out_path, index=False, header=True)


def calculateScalarAMOC():

    for subject in subjects:
        for movement in movements:
            if subject == "E2" and movement == "roundhouse":
                continue

            rawAMOCpath = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}/AMOCVector.csv"
            loadedAMOCData = pandas.read_csv(rawAMOCpath)

            newScalarAMOC = pandas.DataFrame()
            for bodypart in bodyparts:
                if bodypart == "FullBody_AngMom":
                    continue
                col = (bodypart + "AMOC")
                scalarAMOCBodyPart = []
                for time in range(len(loadedAMOCData)):
                    amoc_x = loadedAMOCData[bodypart + "AMOCVX"].iloc[time]
                    amoc_y = loadedAMOCData[bodypart + "AMOCVY"].iloc[time]
                    amoc_z = loadedAMOCData[bodypart + "AMOCVZ"].iloc[time]
                    AMOCVector = np.array([amoc_x, amoc_y, amoc_z])
                    scalarAMOCBodyPart.append(np.linalg.norm(AMOCVector))
                newScalarAMOC[col] = scalarAMOCBodyPart
            out_dir = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"AMOCscalar.csv")
            newScalarAMOC.to_csv(out_path, index=False, header=True)

            
def calculateAMOCVectors():
    for subject in subjects:
        for movement in movements:
            if subject == "E2" and movement == "roundhouse":
                continue
            rawDataPath = "newAngMom/" + subject + "/" + movement + ".csv"
            loadedAngMomData = pandas.read_csv(rawDataPath)
            full_body_angmom_x = loadedAngMomData["('FullBody_AngMom', 'X')"]
            full_body_angmom_y = loadedAngMomData["('FullBody_AngMom', 'Y')"]
            full_body_angmom_z = loadedAngMomData["('FullBody_AngMom', 'Z')"]  

            rawAMACpath = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}/AMACVector.csv"
            loadedAMACData = pandas.read_csv(rawAMACpath)

            newAMOCVectors = pandas.DataFrame()
            
            for bodypart in bodyparts:
                if bodypart == "FullBody_AngMom":
                    continue
                
                col = (bodypart + "AMOCV")
                AMAC_x = []
                AMAC_y = []
                AMAC_z = []

                for time in range(len(loadedAngMomData)):
                    AMOCVectorX = full_body_angmom_x.iloc[time] - loadedAMACData[bodypart + "AMACVX"].iloc[time]
                    AMOCVectorY = full_body_angmom_y.iloc[time] - loadedAMACData[bodypart + "AMACVY"].iloc[time]
                    AMOCVectorZ = full_body_angmom_z.iloc[time] - loadedAMACData[bodypart + "AMACVZ"].iloc[time]
                    AMAC_x.append(AMOCVectorX)
                    AMAC_y.append(AMOCVectorY)
                    AMAC_z.append(AMOCVectorZ)
                newAMOCVectors[col + "X"] = AMAC_x
                newAMOCVectors[col + "Y"] = AMAC_y
                newAMOCVectors[col + "Z"] = AMAC_z

            out_dir = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"AMOCVector.csv")
            newAMOCVectors.to_csv(out_path, index=False, header=True)
            
def calculateTheta(): 
    
    for subject in subjects:
        for movement in movements:
            if subject == "E2" and movement == "roundhouse":
                continue
            rawAMACpath = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}/AMACVector.csv"
            loadedAMACData = pandas.read_csv(rawAMACpath)

            rawDataPath = "newAngMom/" + subject + "/" + movement + ".csv"
            loadedAngMomData = pandas.read_csv(rawDataPath)
            
            newThetaData = pandas.DataFrame()
            for bodypart in bodyparts:
                if bodypart == "FullBody_AngMom":
                    continue
                col = (bodypart + "Theta")
                thetaBodyPart = []
                for time in range(len(loadedAMACData)):
                    amac_x = loadedAMACData[bodypart + "AMACVX"].iloc[time]
                    amac_y = loadedAMACData[bodypart + "AMACVY"].iloc[time]
                    amac_z = loadedAMACData[bodypart + "AMACVZ"].iloc[time]
                    AMACVector = np.array([amac_x, amac_y, amac_z])
                    bodypartAngMomX = loadedAngMomData["('" + bodypart + "', 'X')"].iloc[time]
                    bodypartAngMomY = loadedAngMomData["('" + bodypart + "', 'Y')"].iloc[time]
                    bodypartAngMomZ = loadedAngMomData["('" + bodypart + "', 'Z')"].iloc[time]
                    bodypartVector = np.array([bodypartAngMomX, bodypartAngMomY, bodypartAngMomZ])
                    inside = np.vdot(AMACVector, bodypartVector) / (np.linalg.norm(AMACVector) * np.linalg.norm(bodypartVector))
                    theta = np.arccos(inside)
    
                    thetaBodyPart.append(theta)

                newThetaData[col] = thetaBodyPart
            out_dir = f"/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/{subject}/{movement}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"theta.csv")
            newThetaData.to_csv(out_path, index=False, header=True)

def main():
    calculate_AMAC()
    print("Finished calculating AMAC scalars.")
    calculateAMACVectors()
    print("Finished calculating AMAC vectors.")
    calculateAMOCVectors()
    print("Finished calculating AMOC vectors.")
    calculateScalarAMOC()
    print("Finished calculating AMOC scalars.")
    calculateTheta()
    print("Finished calculating Theta.")
main()