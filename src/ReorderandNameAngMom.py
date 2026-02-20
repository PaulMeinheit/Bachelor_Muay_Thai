import Dataloader
import pandas
import os
from typing import List, Dict
import numpy as np

_TRUNK_DEN = 0.142 + 0.355
_PELVIS_RATIO = 0.142 / _TRUNK_DEN
_TORSO_RATIO  = 0.355 / _TRUNK_DEN
weight = 75.0  



        
trunk = 0.4346
body_part_mass_ratios = {
            "Head": 0.0694,
            "Pelvis": trunk * _PELVIS_RATIO,
            "Trunk":  trunk * _TORSO_RATIO,
            "R_UA": 0.0271, "L_UA": 0.0271,
            "R_FA": 0.0162, "L_FA": 0.0162,
            "R_Hand": 0.0061, "L_Hand": 0.0061,
            "R_Thigh": 0.1416, "L_Thigh": 0.1416,
            "R_Shank": 0.0433, "L_Shank": 0.0433,
            "R_Foot": 0.0137, "L_Foot": 0.0137,
            "R_Toes": 0.0, "L_Toes": 0.0,
            }

  
rawDataPath = "Raw_Data/E1/roundhouse/"
rawAngPath = os.path.join(rawDataPath,"AngMoms_wrt_LAB.txt")
rawCogposPath = os.path.join(rawDataPath, "CoG_Position.txt")
rawCogvelPath = os.path.join(rawDataPath, "CoG_Velocity.txt")

loadedCogPosData = Dataloader.loadData(rawCogposPath)
loadedCogVelData = Dataloader.loadData(rawCogvelPath)
loadedAngMomData = Dataloader.loadData(rawAngPath)
newAngMomData = pandas.DataFrame()
body_parts_mapping = {
    "FullBody_AngMom": "FullBody_CoG",
    "L_Hand_AngMom_wrt_LAB": "L_Hand_CoG",
    "R_Hand_AngMom_wrt_LAB": "R_Hand_CoG",
    "L_FA_AngMom_wrt_LAB": "L_Forearm_CoG",
    "R_FA_AngMom_wrt_LAB": "R_Forearm_CoG",
    "L_UA_AngMom_wrt_LAB": "L_UpperArm_CoG",
    "R_UA_AngMom_wrt_LAB": "R_UpperArm_CoG",
    "Head_AngMom_wrt_LAB": "Head_CoG",
    "Trunk_AngMom_wrt_LAB": "Trunk_CoG",
    "Pelvis_AngMom_wrt_LAB": "Pelvis_CoG",
    "L_Thigh_AngMom_wrt_LAB": "L_Thigh_CoG",
    "R_Thigh_AngMom_wrt_LAB": "R_Thigh_CoG",
    "L_Shank_AngMom_wrt_LAB": "L_Shank_CoG",
    "R_Shank_AngMom_wrt_LAB": "R_Shank_CoG",
    "L_Foot_AngMom_wrt_LAB": "L_Foot_CoG",
    "R_Foot_AngMom_wrt_LAB": "R_Foot_CoG",
}
for bodypart in body_parts_mapping:
    
    if bodypart == "FullBody_AngMom":
        continue

    angmomSeries = loadedAngMomData[bodypart]
    cogposSeries = loadedCogPosData[body_parts_mapping[bodypart]+ "_pos"]
    cogvelSeries = loadedCogVelData[body_parts_mapping[bodypart]+ "_vel"]

    # Prepare to collect H_G values for each axis
    H_G_x = []
    H_G_y = []
    H_G_z = []
    for time in range(len(angmomSeries)):
        I_w = angmomSeries.iloc[time]
        r = cogposSeries.iloc[time] - loadedCogPosData["FullBody_CoG_pos"].iloc[time]
        m = weight * body_part_mass_ratios[bodypart.replace("_AngMom_wrt_LAB", "")]
        v = cogvelSeries.iloc[time]
        H_G = I_w + np.cross(r, v)
        H_G_x.append(H_G[0])
        H_G_y.append(H_G[1])
        H_G_z.append(H_G[2])

    # Add to newAngMomData with the same structure as loadedAngMomData
    for idx, axis in enumerate(['X', 'Y', 'Z']):
        col = (bodypart, axis)
        if col not in newAngMomData.columns:
            newAngMomData[col] = np.nan
        if axis == 'X':
            newAngMomData[col] = H_G_x
        elif axis == 'Y':
            newAngMomData[col] = H_G_y
        elif axis == 'Z':
            newAngMomData[col] = H_G_z

# Save with the same multi-header structure as the raw data
newAngMomData.to_csv("/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/outTest.csv", index=False, header=True)