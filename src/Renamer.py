import pandas as pd
import os
import ast

def renameDirectory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for file in os.listdir(input_directory):
            input_path = os.path.join(input_directory, file)
            output_path = os.path.join(output_directory, file)
            rename_csv(input_path, output_path)

def rename_csv(input_path, output_path):
     data = pd.read_csv(input_path, header=[0])
     newdf = rename_tuple_columns(data)
     newdf.to_csv(output_path)

def rename_tuple_columns(df):
    def parse_col(col):
        try:
            parsed = ast.literal_eval(col)
            if isinstance(parsed, tuple):
                return '_'.join(parsed)
        except:
            pass
        return col  # leave as-is if not a tuple string

    df.columns = [parse_col(c) for c in df.columns]
    return df
