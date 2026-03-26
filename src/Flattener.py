import pandas as pd
import os

def flattenDirectory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for file in os.listdir(input_directory):
            input_path = os.path.join(input_directory, file)
            output_path = os.path.join(output_directory, file)
            flatten_csv(input_path, output_path)


def flatten_csv(input_path, output_path):   


    df = pd.read_csv(input_path, header=[0, 1])
 
    # Set ITEM as index (top-level col name may vary so match on level-1 == 'ITEM')
    item_col = [c for c in df.columns if c[1] == "ITEM"]
    if item_col:
        df = df.set_index(item_col[0])
        df.index.name = "ITEM"
 
    # Flatten MultiIndex columns: "SegmentName" + "X/Y/Z" -> "SegmentName_X"
    df.columns = [f"{lvl0}_{lvl1}" for lvl0, lvl1 in df.columns]
 
    if output_path:
        df.to_csv(output_path)
        print(f"Saved flattened CSV to: {output_path}")
 