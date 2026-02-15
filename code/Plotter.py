import pandas as pd
import Dataloader
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import os

data = Dataloader.loadDataNoSkip("processed_Data/N1/roundhouse/scaled/AngMoms_wrt_LAB/scaled5.csv")
dataToPlot = data["FullBody_AngMom"]

# Extract X, Y, Z values
x_values = dataToPlot.iloc[:, 0]
y_values = dataToPlot.iloc[:, 1]
z_values = dataToPlot.iloc[:, 2]

# Create plot of angular momentum in all 3 directions
fig = plt.figure(figsize=(15, 10))

# X component
ax1 = fig.add_subplot(221)
ax1.plot(x_values, label='X', color='red', linewidth=2)
ax1.set_xlabel('Frame')
ax1.set_ylabel('Angular Momentum (kg⋅m²/s)')
ax1.set_title('Angular Momentum - X Direction')
ax1.grid(True)
ax1.legend()

# Y component
ax2 = fig.add_subplot(222)
ax2.plot(y_values, label='Y', color='green', linewidth=2)
ax2.set_xlabel('Frame')
ax2.set_ylabel('Angular Momentum (kg⋅m²/s)')
ax2.set_title('Angular Momentum - Y Direction')
ax2.grid(True)
ax2.legend()

# Z component
ax3 = fig.add_subplot(223)
ax3.plot(z_values, label='Z', color='blue', linewidth=2)
ax3.set_xlabel('Frame')
ax3.set_ylabel('Angular Momentum (kg⋅m²/s)')
ax3.set_title('Angular Momentum - Z Direction')
ax3.grid(True)
ax3.legend()

# All components overlaid
ax4 = fig.add_subplot(224)
ax4.plot(x_values, label='X', color='red', linewidth=2, alpha=0.7)
ax4.plot(y_values, label='Y', color='green', linewidth=2, alpha=0.7)
ax4.plot(z_values, label='Z', color='blue', linewidth=2, alpha=0.7)
ax4.set_xlabel('Frame')
ax4.set_ylabel('Angular Momentum (kg⋅m²/s)')
ax4.set_title('All Components Over Time')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()