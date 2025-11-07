import pandas
df = pandas.read_csv(filepath_or_buffer = "Sub01_Teep/CoG_Position.txt", sep='\t', skiprows= lambda x: x in [0, 2, 3], header= [0,1])

print(df.head())