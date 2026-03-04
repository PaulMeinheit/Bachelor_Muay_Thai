# Method to plot 1D AMAC and AMOC scalar values for each bodypart

import pandas as pd
import matplotlib.pyplot as plt

def plot_scalar_amac_amoc(amac_scalar_path, amoc_scalar_path, bodyparts):
    df_amac = pd.read_csv(amac_scalar_path)
    df_amoc = pd.read_csv(amoc_scalar_path)
    for bp in bodyparts:
        amac_col = bp + "AMAC"
        amoc_col = bp + "AMOC"
        # Plot AMAC only
        if amac_col in df_amac.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df_amac[amac_col], label="AMAC", color='b')
            plt.ylabel(bp)
            plt.title(f"{bp} AMAC Scalar")
            plt.xlabel("Frame")
            plt.legend()
            plt.tight_layout()
            outname = f"AMAC_scalar_{bp}.png"
            plt.savefig(outname)
            plt.close()
            print(f"AMAC scalar plot saved as {outname}")
        # Plot AMOC only
        if amoc_col in df_amoc.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df_amoc[amoc_col], label="AMOC", color='r')
            plt.ylabel(bp)
            plt.title(f"{bp} AMOC Scalar")
            plt.xlabel("Frame")
            plt.legend()
            plt.tight_layout()
            outname = f"AMOC_scalar_{bp}.png"
            plt.savefig(outname)
            plt.close()
            print(f"AMOC scalar plot saved as {outname}")

import pandas as pd
import matplotlib.pyplot as plt

# Path to your data file
file_path = "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/processed_AngMomData/E3/roundhouse/scaled/AMOCVector/averaged.csv"

# User-selected body parts (base names)
bodyparts = ["L_Hand", "R_Hand", "L_FA", "R_FA", "L_UA", "R_UA", "Head", "Trunk", "Pelvis", "L_Thigh", "R_Thigh", "L_Shank", "R_Shank", "L_Foot", "R_Foot"]


# Create and save a separate plot for each bodypart
def plot_vector_components(df, bodyparts,file_path):
    df = pd.read_csv(file_path)
    for bp in bodyparts:
        vx = bp + "AMOCVX"
        vy = bp + "AMOCVY"
        vz = bp + "AMOCVZ"
        plt.figure(figsize=(10, 4))
        plt.plot(df[vx], label="VX")
        plt.plot(df[vy], label="VY")
        plt.plot(df[vz], label="VZ")
        plt.ylabel(bp)
        plt.title(f"{bp} AMOC Vector Components")
        plt.xlabel("Frame")
        plt.legend()
        plt.tight_layout()
        outname = f"AMOCVector_{bp}.png"
        plt.savefig(outname)
        plt.close()
        print(f"Plot saved as {outname}")


def compare_scalar_paths(pathbeginner, pathexpert, bodyparts, scalar_type="AMAC"):
    """
    Compare AMAC or AMOC scalar values from two different CSV files.

    Parameters:
        path1 (str): Path to first CSV file
        path2 (str): Path to second CSV file
        bodyparts (list): List of bodypart names (e.g. ["L_Hand", "R_Hand"])
        scalar_type (str): "AMAC" or "AMOC"
    """
    
    dfbeginner = pd.read_csv(pathbeginner)
    dfexpert = pd.read_csv(pathexpert)
    
    for bp in bodyparts:
        col_name = bp + scalar_type
        
        if col_name in dfbeginner.columns and col_name in dfexpert.columns:
            plt.figure(figsize=(10, 4))
            
            plt.plot(dfbeginner[col_name], label="Beginner")
            plt.plot(dfexpert[col_name], label="Expert")
            
            plt.ylabel(bp)
            plt.xlabel("Frame")
            plt.title(f"{bp} {scalar_type} Comparison")
            plt.legend()
            plt.tight_layout()
            
            outname = f"{scalar_type}_comparison_{bp}.png"
            plt.savefig(outname)
            plt.close()
            
            print(f"{scalar_type} comparison plot saved as {outname}")
        else:
            print(f"Column {col_name} not found in both files.")

compare_scalar_paths(
    pathbeginner="/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/scaled_Data/processed_AngMomData/N1/roundhouse/scaled/AMOCscalar/averaged.csv",
    pathexpert="/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/scaled_Data/processed_AngMomData/E3/roundhouse/scaled/AMOCscalar/averaged.csv",
    bodyparts=bodyparts,
    scalar_type="AMOC"
)