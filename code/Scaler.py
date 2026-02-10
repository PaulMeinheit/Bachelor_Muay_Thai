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
    framesToScaleTo = 480  # 4 phases of 120 frames each
    internalFolder = os.path.join(scaledResultPath,directory)
    os.makedirs(internalFolder)

    def _safe_spline(segment_df, out_len):
        if out_len <= 0:
            return pd.DataFrame(columns=segment_df.columns)
        n = len(segment_df)
        if n == 0:
            return pd.DataFrame(np.nan, index=range(out_len), columns=segment_df.columns)
        if n == 1:
            row = segment_df.iloc[0].values
            arr = np.tile(row, (out_len, 1))
            return pd.DataFrame(arr, columns=segment_df.columns)
        return spline_interpolate_df(segment_df, out_len)

    for i in range(len(dict)):
        dataToScale = dict[i].drop(dict[i].columns[0], axis=1)

        # fixed phase proportions: 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1
        base_len = int(framesToScaleTo * 0.25)
        segment_lengths = [base_len, base_len, base_len, base_len]
        # adjust last segment to ensure total equals framesToScaleTo
        segment_lengths[-1] += framesToScaleTo - sum(segment_lengths)

        # extract segments using provided indices in `segments`
        p1_df = dataToScale.iloc[0:segments[i][1]]
        p2_df = dataToScale.iloc[segments[i][1]:segments[i][2]]
        p3_df = dataToScale.iloc[segments[i][2]:segments[i][3]]
        p4_df = dataToScale.iloc[segments[i][3]:segments[i][4]]

        p1_scaled = _safe_spline(p1_df, segment_lengths[0])
        p2_scaled = _safe_spline(p2_df, segment_lengths[1])
        p3_scaled = _safe_spline(p3_df, segment_lengths[2])
        p4_scaled = _safe_spline(p4_df, segment_lengths[3])

        finalScaledData = pd.concat([p1_scaled, p2_scaled, p3_scaled, p4_scaled], ignore_index=True)
        tempname = "scaled" + str(i) + ".csv"
        open(os.path.join(internalFolder, tempname), "x")
        finalScaledData.to_csv(os.path.join(internalFolder, tempname), sep=",", index=False)
