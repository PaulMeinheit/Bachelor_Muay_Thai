import pandas as pd
import Dataloader
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import os
import numpy as np

data = Dataloader.loadDataNoSkip("processed_Data/E1/roundhouse/scaled/CoG_Position/scaled2")
dataToPlot = data["R_Foot_CoG_pos"]

# Extract X, Y, Z values
x_values = dataToPlot.iloc[:, 0]
y_values = dataToPlot.iloc[:, 1]
z_values = dataToPlot.iloc[:, 2]

# Create plot with 4 subplots
fig = plt.figure(figsize=(16, 12))

# X component over time
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x_values, label='X', color='red', linewidth=2)
ax1.set_xlabel('Frame')
ax1.set_ylabel('Position (m)')
ax1.set_title('FullBody CoG Position - X Direction')
ax1.grid(True)
ax1.legend()

# Y component over time
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(y_values, label='Y', color='green', linewidth=2)
ax2.set_xlabel('Frame')
ax2.set_ylabel('Position (m)')
ax2.set_title('FullBody CoG Position - Y Direction')
ax2.grid(True)
ax2.legend()

# Z component over time
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(z_values, label='Z', color='blue', linewidth=2)
ax3.set_xlabel('Frame')
ax3.set_ylabel('Position (m)')
ax3.set_title('FullBody CoG Position - Z Direction')
ax3.grid(True)
ax3.legend()

# 3D trajectory plot
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot(x_values, y_values, z_values, color='purple', linewidth=2)
ax4.scatter(x_values.iloc[0], y_values.iloc[0], z_values.iloc[0], color='green', s=100, label='Start')
ax4.scatter(x_values.iloc[-1], y_values.iloc[-1], z_values.iloc[-1], color='red', s=100, label='End')
ax4.set_xlabel('X Position (m)')
ax4.set_ylabel('Y Position (m)')
ax4.set_zlabel('Z Position (m)')
ax4.set_title('FullBody CoG Trajectory (3D)')
ax4.legend()
ax4.grid(True)

plt.tight_layout()

"""# Save the plot to the plots folder
plots_dir = "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/plots"
plot_filename = os.path.join(plots_dir, "rightfootN1s6scaled.png")
print(f"Attempting to save to: {os.path.abspath(plot_filename)}", flush=True)
print(f"Directory exists: {os.path.exists(plots_dir)}", flush=True)
try:
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}", flush=True)
    print(f"File exists after save: {os.path.exists(plot_filename)}", flush=True)
except Exception as e:
    print(f"Error saving plot: {e}", flush=True)
"""
plt.show()
plt.close()