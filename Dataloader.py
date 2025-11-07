import pandas
df = pandas.read_csv(filepath_or_buffer = "Sub01_Teep/CoG_Position.txt", sep='\t', header=[0,1,2,3])
df = df.drop(index=[0, 2])

print(df)