import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the right shank theta data (angle in radians)
csv_path = "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/calculatedAngMomStuff/Theta/E1/roundhouse.csv"  # Adjust path if running from a different directory

df = pd.read_csv(csv_path)

# The column for right shank theta
r_shank_col = "R_FootTheta"

# Convert radians to degrees for interpretability
r_shank_deg = np.rad2deg(df[r_shank_col].dropna())

plt.figure(figsize=(12, 5))
plt.plot(r_shank_deg, label='Right Shank Angle (deg)', color='purple')
plt.title('Right Shank Theta (Angle) Over Time')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Histogram to show angle distribution
plt.figure(figsize=(8, 4))
plt.hist(r_shank_deg, bins=40, color='skyblue', edgecolor='black')
plt.title('Distribution of Right Shank Theta (Degrees)')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
