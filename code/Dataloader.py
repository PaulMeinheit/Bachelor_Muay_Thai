import pandas
import matplotlib
import matplotlib.pyplot as plt

def loadData(filepath):
    loadedData = pandas.read_csv(filepath_or_buffer = filepath, sep='\t', skiprows= lambda x: x in [0, 2, 3], header= [0,1])
    return loadedData

def loadDataNoSkip(filepath):
    loadedData = pandas.read_csv(filepath_or_buffer = filepath, sep=',', header= [0,1])
    return loadedData