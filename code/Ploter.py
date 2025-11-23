import pandas as pd
import Dataloader
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots


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

# Show plot
plt.show()
