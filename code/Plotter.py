import pandas as pd
import Dataloader
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import os

data = Dataloader.loadDataNoSkip("processed_Data/E1/roundhouse/scaled/AngMoms_wrt_LAB/scaled2.csv")
dataToPlot = data["FullBody_AngMom"]

# Extract X, Y, Z values
x_values = dataToPlot.iloc[:, 0]
y_values = dataToPlot.iloc[:, 1]
z_values = dataToPlot.iloc[:, 2]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot X component
axes[0].plot(x_values, label='X', color='red')
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Angular Momentum (X)')
axes[0].set_title('Angular Momentum - X Direction')
axes[0].grid(True)
axes[0].legend()

# Plot Y component
axes[1].plot(y_values, label='Y', color='green')
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Angular Momentum (Y)')
axes[1].set_title('Angular Momentum - Y Direction')
axes[1].grid(True)
axes[1].legend()

# Plot Z component
axes[2].plot(z_values, label='Z', color='blue')
axes[2].set_xlabel('Frame')
axes[2].set_ylabel('Angular Momentum (Z)')
axes[2].set_title('Angular Momentum - Z Direction')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()