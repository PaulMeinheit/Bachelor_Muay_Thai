"""Scaler helpers

This file contains utilities to resample and scale time-series data.
"""

from typing import Dict, List, Optional, Union
import os
import pandas as pd
import numpy as np
from scipy.interpolate import splrep, splev

def load_csvs_from_dir(directoryPath):
    dataframes = []
    for file in sorted(os.listdir(directoryPath)):
    
       data = pd.read_csv(os.path.join(directoryPath,file),header=[0, 1])
       dataframes.append(data)

    return dataframes
def calculateAverageSegementRatios(segments):
    ratios = []
    for segment in segments:
        total_frames = segment[4]
        phase_ratios = [
            segment[1] / total_frames,
            segment[2] / total_frames,
            segment[3] / total_frames,
            1
        ]
        ratios.append(phase_ratios)
    
    arr = np.array(ratios)

    # average each column (axis=0)
    avg = arr.mean(axis=0)

    print(avg)
        
    return avg



def spline_interpolate_df(df, new_length, k=3):
    """
    Spline-interpolates each column of a DataFrame to a new number of rows.
    
    df: original DataFrame
    new_length: number of rows desired in output
    k: spline order (default cubic = 3)
    """
    
    # original x-values (index positions 0 ... N-1)
    x = np.arange(len(df))
    
    # new x-values to interpolate onto
    x_new = np.linspace(0, len(df) - 1, new_length)
    
    out = {}
    for col in df.columns:
        y = df[col].values
        
        # build spline and evaluate it
        tck = splrep(x, y, k=k)
        out[col] = splev(x_new, tck)
    
    return pd.DataFrame(out, columns=df.columns)

def scaleToFourPhases(dict,segments,scaledResultPath,directory):
    framesToScaleTo = 500
    internalFolder = os.path.join(scaledResultPath,directory)
    os.makedirs(internalFolder)
    avgRatios = calculateAverageSegementRatios(segments)
    for i in range (len(dict)):
        dataToScale = dict[i]
        dataToScale = dataToScale.drop(dataToScale.columns[0], axis=1)
        #phase1_scaling
        phase1_end = int(avgRatios[0] * framesToScaleTo)
        phase1_data = dataToScale.iloc[0:segments[i][1]]
        phase1_scaled = spline_interpolate_df(phase1_data, phase1_end)
        #phase2_scaling
        phase2_end = int(avgRatios[1] * framesToScaleTo)
        phase2_data = dataToScale.iloc[segments[i][1]:segments[i][2]]
        phase2_scaled = spline_interpolate_df(phase2_data, phase2_end - phase1_end)
        #phase3_scaling
        phase3_end = int(avgRatios[2] * framesToScaleTo)
        phase3_data = dataToScale.iloc[segments[i][2]:segments[i][3]]
        phase3_scaled = spline_interpolate_df(phase3_data, phase3_end - phase2_end)
        #phase4_scaling
        phase4_end = framesToScaleTo
        phase4_data = dataToScale.iloc[segments[i][3]:segments[i][4]]
        phase4_scaled = spline_interpolate_df(phase4_data, phase4_end - phase3_end)
        #concat phases
        finalScaledData= pd.concat([phase1_scaled,phase2_scaled,phase3_scaled,phase4_scaled], ignore_index=True)
        tempname ="scaled" + str(i) + ".csv"
        open(os.path.join(internalFolder, tempname), "x")
        finalScaledData.to_csv(os.path.join(internalFolder, tempname), sep=",", index = False)