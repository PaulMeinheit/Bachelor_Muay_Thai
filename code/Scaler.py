"""Scaler helpers

This file contains utilities to resample and scale time-series data.
"""

from typing import Dict, List, Optional, Union
import os
import pandas as pd


def load_csvs_from_dir(directoryPath):
    dataframes = []
    for file in sorted(os.listdir(directoryPath)):
    
       data = pd.read_csv(os.path.join(directoryPath,file),header=[0, 1])
       dataframes.append(data)

def scaleToFourPhases():
    # placeholder kept for API compatibility; see project scripts for
    # the desired scaling function implementation.
    return