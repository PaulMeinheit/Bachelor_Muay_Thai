import pandas
import matplotlib
import matplotlib.pyplot as plt
loadedData = pandas.read_csv(filepath_or_buffer = "Sub01_Teep/JointPositions.txt", sep='\t', skiprows= lambda x: x in [0, 2, 3], header= [0,1])

JointToPlot = loadedData["R_ANKLE_POSITION"]


ax = plt.figure().add_subplot(projection='3d')


ax.plot(JointToPlot['X'], JointToPlot['Y'], JointToPlot['Z'])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()