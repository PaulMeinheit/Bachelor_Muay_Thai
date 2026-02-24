import pandas as pd
import matplotlib.pyplot as plt

# Load the right hand AMOC data
csv_path = "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/AMOCscalar/E2/uppercut.csv"  # Adjust path if running from a different directory

df = pd.read_csv(csv_path)

# The column for right hand AMOC
r_hand_col = "R_HandAMOC"

# 1. Line plot
plt.figure(figsize=(10, 4))
plt.plot(df[r_hand_col], label='R_HandAMOC', color='blue')
plt.title('Right Hand AMOC - Line Plot')
plt.xlabel('Frame')
plt.ylabel('AMOC Value')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Histogram
plt.figure(figsize=(8, 4))
plt.hist(df[r_hand_col].dropna(), bins=30, color='orange', edgecolor='black')
plt.title('Right Hand AMOC - Histogram')
plt.xlabel('AMOC Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3. Boxplot
plt.figure(figsize=(4, 6))
plt.boxplot(df[r_hand_col].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Right Hand AMOC - Boxplot')
plt.ylabel('AMOC Value')
plt.tight_layout()
plt.show()
