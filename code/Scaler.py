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

def scaleToFourPhases(dict, segments, scaledResultPath, directory):
    """
    Scale each phase independently to fixed length (0.25 * framesToScaleTo) using spline interpolation.
    
    """
    framesToScaleTo = 200  
    phase_frame_length = int(framesToScaleTo * 0.25)  # 120 frames per phase
    internalFolder = os.path.join(scaledResultPath, directory)
    os.makedirs(internalFolder)

    def _safe_spline(segment_df, out_len):
        """Safely interpolate a segment using spline."""
        if out_len <= 0:
            return pd.DataFrame(columns=segment_df.columns)
        n = len(segment_df)
        if n == 0:
            return pd.DataFrame(columns=segment_df.columns)
        if n == 1:
            row = segment_df.iloc[0].values
            arr = np.tile(row, (out_len, 1))
            return pd.DataFrame(arr, columns=segment_df.columns)
        # Use adaptive spline order
        k = min(3, n - 1)
        return spline_interpolate_df(segment_df, out_len, k=k)

    for i in range(len(dict)):
        # Get the raw data (excluding index column)
        raw_data = dict[i].drop(dict[i].columns[0], axis=1)
        
        # Get segment boundaries from the segments data
        seg = segments[i]  # [0, lift_off_frame, impact_frame, foot_down_frame, total_frames]
        
        # Extract each phase
        p1_df = raw_data.iloc[seg[0]:seg[1]]
        p2_df = raw_data.iloc[seg[1]:seg[2]]
        p3_df = raw_data.iloc[seg[2]:seg[3]]
        p4_df = raw_data.iloc[seg[3]:seg[4]]
        
        # Scale each phase to fixed length
        p1_scaled = _safe_spline(p1_df, phase_frame_length)
        p2_scaled = _safe_spline(p2_df, phase_frame_length)
        p3_scaled = _safe_spline(p3_df, phase_frame_length)
        p4_scaled = _safe_spline(p4_df, phase_frame_length)
        
        # Concatenate all phases
        finalScaledData = pd.concat([p1_scaled, p2_scaled, p3_scaled, p4_scaled], ignore_index=True)
        
        # Save the scaled data
        tempname = "scaled" + str(i) + ".csv"
        filepath = os.path.join(internalFolder, tempname)
        open(filepath, "x")
        finalScaledData.to_csv(filepath, sep=",", index=False)



