import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_dataframe(path):
    # detect the row that contains the final header token 'ITEM'
    item_row = None
    try:
        with open(path, 'r', errors='ignore') as f:
            for idx, line in enumerate(f):
                tokens = line.strip().split()
                if len(tokens) and tokens[0] == 'ITEM':
                    item_row = idx
                    break
    except Exception:
        item_row = None

    if item_row is not None and item_row >= 1:
        header_rows = list(range(1, item_row + 1))
        try:
            df = pd.read_csv(path, header=header_rows, sep='\t', engine='python')
        except Exception:
            df = pd.read_csv(path, header=header_rows, delim_whitespace=True, engine='python')
    else:
        # try to let pandas infer separator, fallback to whitespace
        try:
            df = pd.read_csv(path, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(path, delim_whitespace=True, header=0)

    return df


def find_rfoot_columns(df):
    # Return x,y,z column names (or single column) for R_Foot_CoG_pos
    cols = []
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            if 'R_Foot_CoG_pos' in str(col[0]):
                cols.append(col)
    else:
        for col in df.columns:
            if 'R_Foot_CoG_pos' in str(col):
                cols.append(col)

    if len(cols) >= 3:
        return cols[:3]

    # try other heuristics: look for columns that contain R_Foot and CoG
    candidates = [c for c in df.columns if 'R_Foot' in str(c) and 'CoG' in str(c)]
    if len(candidates) >= 3:
        return candidates[:3]

    # try suffix based names like R_Foot_CoG_pos_X etc.
    x = [c for c in df.columns if 'R_Foot_CoG_pos' in str(c) and ('_X' in str(c) or ' X' in str(c) or '_x' in str(c))]
    y = [c for c in df.columns if 'R_Foot_CoG_pos' in str(c) and ('_Y' in str(c) or ' Y' in str(c) or '_y' in str(c))]
    z = [c for c in df.columns if 'R_Foot_CoG_pos' in str(c) and ('_Z' in str(c) or ' Z' in str(c) or '_z' in str(c))]
    if x and y and z:
        return [x[0], y[0], z[0]]

    # last resort: any numeric columns after locating R_Foot substring
    any_cols = [c for c in df.columns if 'R_Foot' in str(c)]
    if len(any_cols) >= 1:
        return [any_cols[0]]

    return []


def to_series(df, col):
    # extract a series given possible MultiIndex col or plain
    try:
        return pd.to_numeric(df[col], errors='coerce')
    except Exception:
        # try accessing as tuple
        if isinstance(col, tuple):
            return pd.to_numeric(df[col], errors='coerce')
        raise


def plot_for_subject(df, cols, outpath, subject):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig = plt.figure(figsize=(14, 10))

    if len(cols) >= 3:
        x = to_series(df, cols[0])
        y = to_series(df, cols[1])
        z = to_series(df, cols[2])

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(x, color='red', linewidth=1.5)
        ax1.set_title(f'{subject} R_Foot_CoG_pos - X')
        ax1.set_xlabel('Frame')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(y, color='green', linewidth=1.5)
        ax2.set_title(f'{subject} R_Foot_CoG_pos - Y')
        ax2.set_xlabel('Frame')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(z, color='blue', linewidth=1.5)
        ax3.set_title(f'{subject} R_Foot_CoG_pos - Z')
        ax3.set_xlabel('Frame')

        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot(x, y, z, color='purple', linewidth=1.2)
        ax4.set_title(f'{subject} R_Foot_CoG_pos Trajectory')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')

    elif len(cols) == 1:
        s = to_series(df, cols[0])
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(s, color='tab:blue')
        ax.set_title(f'{subject} {cols[0]}')
        ax.set_xlabel('Frame')

    else:
        raise RuntimeError('No R_Foot_CoG_pos columns found')

    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    subjects = ['E1', 'E3', 'N1', 'N2', 'N3', 'N4']
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    outputs = []

    for subj in subjects:
        path = os.path.join(repo_root, 'Raw_Data', subj, 'roundhouse', 'CoG_Position.txt')
        if not os.path.exists(path):
            print(f'Missing file for {subj}: {path}')
            continue
        print(f'Reading {path}')
        df = read_dataframe(path)
        cols = find_rfoot_columns(df)
        if not cols:
            print(f'Could not find R_Foot_CoG_pos columns in {path}')
            continue
        outdir = os.path.join(repo_root, 'plots')
        outfile = os.path.join(outdir, f'R_Foot_CoG_pos_{subj}.png')
        try:
            plot_for_subject(df, cols, outfile, subj)
            print(f'Wrote plot: {outfile}')
            outputs.append(outfile)
        except Exception as e:
            print(f'Error plotting {subj}: {e}')

    if outputs:
        print('Generated files:')
        for p in outputs:
            print(p)
    else:
        print('No plots generated.')


if __name__ == '__main__':
    main()
