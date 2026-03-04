import os
import pandas as pd
import numpy as np


def average_scaled_files(directory, exclusions, output_file="averaged.csv"):
	# Find all scaled*.csv files
	files = [f for f in os.listdir(directory) if f.startswith("scaled") and f[-1] not in exclusions]
	if not files:
		raise ValueError("No scaled* files found in directory.")
	# Read all files into DataFrames
	dfs = [pd.read_csv(os.path.join(directory, f)) for f in sorted(files)]
	if not files:
		raise ValueError("No scaled*.csv files found in directory.")
	# Read all files into DataFrames
	dfs = [pd.read_csv(os.path.join(directory, f)) for f in sorted(files)]
	# Check all have same shape
	n_rows = dfs[0].shape[0]
	n_cols = dfs[0].shape[1]
	for df in dfs:
		if df.shape != (n_rows, n_cols):
			raise ValueError("All scaled files must have the same shape.")
	# Stack and average
	arr = np.stack([df.values for df in dfs], axis=0)
	avg_arr = np.mean(arr, axis=0)
	# Create averaged DataFrame with same columns
	avg_df = pd.DataFrame(avg_arr, columns=dfs[0].columns)
	# Write to output file in the same directory
	avg_df.to_csv(os.path.join(directory, output_file), index=False)
	print(f"Averaged file written to {os.path.join(directory, output_file)}")

