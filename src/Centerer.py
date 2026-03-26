
import pandas as pd
import os
 
def center_cog(inputpath,outputpath):
    os.makedirs(outputpath, exist_ok=True)
    for file in os.listdir(inputpath):
        df = pd.read_csv(os.path.join(inputpath, file))
        centered = df.copy()
    
        xyz_cols = [c for c in df.columns if c.endswith(("_X", "_Y", "_Z"))]
    
        for col in xyz_cols:
            centered[col] = df[col] - df[col].iloc[0]
    
        centered.to_csv(os.path.join(outputpath, file), index=False)
        print(f"Centered file written to {outputpath}")