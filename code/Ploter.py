import pandas as pd
import Dataloader
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots


data = Dataloader.loadDataNoSkip("AveragedResults/JointPositions.txt.csv")
AnkleData = data["L_ANKLE_POSITION"]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(AnkleData['X'], AnkleData['Y'], AnkleData['Z'], c='blue', marker='o')

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()
