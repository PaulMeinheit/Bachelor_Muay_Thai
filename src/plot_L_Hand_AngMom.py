import pandas as pd
import matplotlib.pyplot as plt

# Load the output CSV
csv_path = "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/outTest.csv"  # Adjust path if running from a different directory

df = pd.read_csv(csv_path)

# The columns are tuples as strings, so we need to find the correct columns for L_Hand_AngMom
x_col = "('L_Hand_AngMom_wrt_LAB', 'X')"
y_col = "('L_Hand_AngMom_wrt_LAB', 'Y')"
z_col = "('L_Hand_AngMom_wrt_LAB', 'Z')"

plt.figure(figsize=(10,6))
plt.plot(df[x_col], label='X')
plt.plot(df[y_col], label='Y')
plt.plot(df[z_col], label='Z')
plt.title('L_Hand_AngMom (H_G) - All Axes')
plt.xlabel('Frame')
plt.ylabel('H_G')
plt.legend()
plt.tight_layout()
plt.show()
