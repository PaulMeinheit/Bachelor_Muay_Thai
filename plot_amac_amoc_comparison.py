import pandas as pd
import matplotlib.pyplot as plt
import os

# Example file paths (update as needed)
amac_file = "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/processed_AngMomData/E3/roundhouse/scaled/AMACscalar/scaled2"
amoc_file = "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/processed_AngMomData/E3/roundhouse/scaled/AMOCscalar/scaled2"

# Read the CSV files
df_amac = pd.read_csv(amac_file)
df_amoc = pd.read_csv(amoc_file)

# User-selected columns to plot (update as needed)
selected_columns = [
    "L_FootAMAC", "R_FootAMAC", "L_HandAMAC", "R_HandAMAC", "HeadAMAC", "TrunkAMAC", "PelvisAMAC", "L_ThighAMAC", "R_ThighAMAC", "L_ShankAMAC", "R_ShankAMAC"
]

# Filter columns that exist in each dataframe
amac_cols = [col for col in selected_columns if col in df_amac.columns]
amoc_cols = [col.replace("AMAC", "AMOC") for col in amac_cols if col.replace("AMAC", "AMOC") in df_amoc.columns]

# Plot AMAC columns
fig1, axes1 = plt.subplots(len(amac_cols), 1, figsize=(10, 2*len(amac_cols)), sharex=True)
if len(amac_cols) == 1:
    axes1 = [axes1]
for i, col in enumerate(amac_cols):
    axes1[i].plot(df_amac[col])
    axes1[i].set_ylabel(col)
    axes1[i].set_title(f"{col} (AMAC)")
axes1[-1].set_xlabel("Frame")
plt.tight_layout()
plt.savefig("amac_selected.png")
plt.close(fig1)

# Plot AMOC columns
fig2, axes2 = plt.subplots(len(amoc_cols), 1, figsize=(10, 2*len(amoc_cols)), sharex=True)
if len(amoc_cols) == 1:
    axes2 = [axes2]
for i, col in enumerate(amoc_cols):
    axes2[i].plot(df_amoc[col])
    axes2[i].set_ylabel(col)
    axes2[i].set_title(f"{col} (AMOC)")
axes2[-1].set_xlabel("Frame")
plt.tight_layout()
plt.savefig("amoc_selected.png")
plt.close(fig2)

print(f"AMAC plot saved as amac_selected.png with columns: {amac_cols}")
print(f"AMOC plot saved as amoc_selected.png with columns: {amoc_cols}")
