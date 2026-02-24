import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file
file_path = "test/AMACVector/E1/roundhouse.csv"
df = pd.read_csv(file_path)

# Plot the right foot vector as a 3D trajectory (X, Y, Z)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(
	df["R_FootAMACVX"],
	df["R_FootAMACVY"],
	df["R_FootAMACVZ"],
	label="Right Foot Trajectory",
	color="purple"
)
ax.set_title("Right Foot AMAC Vector Trajectory (E1 Roundhouse)")
ax.set_xlabel("AMAC VX")
ax.set_ylabel("AMAC VY")
ax.set_zlabel("AMAC VZ")
ax.legend(loc="best")
plt.tight_layout()
plt.show()