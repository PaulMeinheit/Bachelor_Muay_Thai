import os
import pandas
import matplotlib
import matplotlib.pyplot as plt

def loadData(filepath):
    loadedData = pandas.read_csv(filepath_or_buffer = filepath, sep='\t', skiprows= lambda x: x in [0, 2, 3], header= [0,1])
    return loadedData

def loadDataNoSkip(filepath):
    loadedData = pandas.read_csv(filepath_or_buffer = filepath, sep=',', header= [0,1])
    return loadedData
    

def load_csvs_from_dir(input_dir):
    data_dict = {}
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_dir, filename)
            # Try comma-separated first (sliced files), fall back to tab-separated (raw files)
            try:
                data_dict[filename] = pandas.read_csv(filepath_or_buffer=filepath, sep=',', header=[0, 1])
            except Exception:
                try:
                    data_dict[filename] = loadData(filepath)
                except Exception as e:
                    print(f"Warning: could not load {filename}: {e}")
    return data_dict