import pandas as pd
import Dataloader
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import os

data = Dataloader.loadDataNoSkip("AveragedResults/CoG_Position.txt.csv")
dataToPlot = data["FullBody_CoG_pos"]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(dataToPlot['X'], dataToPlot['Y'], dataToPlot['Z'], c='red', marker='o')
ax.margins(x=1,z=2)

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

os.makedirs("plots", exist_ok=True)                    # create folder if missing
out_path = os.path.join("plots", "CoG_plot.png")       # file name + folder
fig.savefig(out_path, dpi=300, bbox_inches='tight')    # save high-res PNG
plt.show()


# load averaged data
df = Dataloader.loadDataNoSkip("AveragedResults/GRF_filtered12hz_resampled120hz.txt.csv")

# If your DataFrame has MultiIndex columns (level0=sensor, level1=axis) access with a tuple:
z = df[("FP2", "Z")]

# Simple plot
plt.figure(figsize=(10, 4))
plt.plot(z.index, z.values, label="FP2 Z")
plt.xlabel("Frame")
plt.ylabel("Z (force)")
plt.title("FP2 Z time-series")
plt.grid(True)
plt.legend()
plt.show()
# Save plot
plt.savefig(os.path.join("plots", "FP2_Z_plot.png"), dpi=300, bbox_inches='tight')


data = Dataloader.loadDataNoSkip("AveragedResults/JointPositions.txt.csv")
dataToPlot = data["L_ANKLE_POSITION"]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(dataToPlot['X'], dataToPlot['Y'], dataToPlot['Z'], c='blue', marker='o')
ax.margins(x=1,z=2)

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#save plot
os.makedirs("plots", exist_ok=True)                    # create folder if missing
out_path = os.path.join("plots", "L_ANKLE_POSITION_plot.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight')    # save high-res PNG
plt.show()

# load averaged data
df = Dataloader.loadDataNoSkip("AveragedResults/JointPositions.txt.csv")

# If your DataFrame has MultiIndex columns (level0=sensor, level1=axis) access with a tuple:
z = df[("L_HIP_POSITION", "X")]

# Simple plot
plt.figure(figsize=(10, 4))
plt.plot(z.index, z.values, label="X")
plt.title("X position of L_HIP over time")
plt.grid(True)
plt.legend()
plt.show()
# Save plot
# If your DataFrame has MultiIndex columns (level0=sensor, level1=axis) access with a tuple:
z = df[("L_HIP_POSITION", "Y")]

# Simple plot
plt.figure(figsize=(10, 4))
plt.plot(z.index, z.values, label="Y")
plt.title("Y position of L_HIP over time")
plt.grid(True)
plt.legend()
plt.show()
# Save plot
# If your DataFrame has MultiIndex columns (level0=sensor, level1=axis) access with a tuple:
z = df[("L_HIP_POSITION", "Z")]

# Simple plot
plt.figure(figsize=(10, 4))
plt.plot(z.index, z.values, label="Z")
plt.title("Z position of L_HIP over time")
plt.grid(True)
plt.legend()
plt.show()
# Save plot