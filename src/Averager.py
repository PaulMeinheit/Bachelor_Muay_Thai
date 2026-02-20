"""Averager

Read scaled CSV files under `scaledResults/` and produce one averaged
CSV per subdirectory. The averaged files are written to `AveragedResults/`.

Behavior:
- For each subdirectory in `scaledResults/` (non-recursive), collect all
  `.csv` files in directory order.
- Read each CSV with `pandas.read_csv(header=[0,1])` where possible (many
  files in this project have two header rows). If that fails, fall back
  to `header=0`.
- Detect if the first column is an index column (e.g. named 'ITEM' or
  'Unnamed...' or monotonically increasing). If so, align on that index
  (intersection across files) and do element-wise averaging for numeric
  columns. Non-numeric columns are taken from the first file.
- If no index column is detected, truncate all files to the minimum
  number of rows and average numeric columns row-wise.

This script does not try to guess advanced mismatches; it will raise
exceptions on irreconcilable differences so you can inspect the inputs.
"""

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import os


def _read_csv_try_multi(fp: Path) -> pd.DataFrame:
    """Try reading CSV with a 2-row header, fall back to single-row."""
    try:
        df = pd.read_csv(fp, header=[0, 1])
        return df
    except Exception:
        # fallback to a single header row
        return pd.read_csv(fp, header=0)


def _detect_index_column(df: pd.DataFrame) -> bool:
    """Heuristic: decide whether first column should be treated as index.

    Returns True when the first column name looks like an index header
    (contains 'unnamed' or 'item') or the column values are numeric and
    monotonic non-decreasing.
    """
    first_col = df.columns[0]
    # check header text
    if isinstance(first_col, tuple):
        names = [str(x).lower() for x in first_col]
        if any('unnamed' in n for n in names) or any('item' in n for n in names):
            return True
    else:
        s = str(first_col).lower()
        if 'unnamed' in s or 'item' in s:
            return True

    # check values
    col = df.iloc[:, 0]
    # coerce to numeric to test monotonicity
    num = pd.to_numeric(col, errors='coerce')
    if num.isna().any():
        return False
    diffs = num.diff().iloc[1:]
    if (diffs >= 0).all():
        return True
    return False


def average_csvs_in_subdir(subdir: Path, out_dir: Path) -> None:
    """Average all CSV files in `subdir` and write single output to `out_dir`.

    The output filename is `<subdir.name>.csv` inside `out_dir`.
    """
    csv_files = [p for p in sorted(subdir.iterdir()) if p.suffix.lower() == '.csv']
    if not csv_files:
        return

    dfs: List[pd.DataFrame] = []
    for fp in csv_files:
        df = _read_csv_try_multi(fp)
        dfs.append(df)

    # Decide whether to use first column as index
    use_index = _detect_index_column(dfs[0])
    if use_index:
        # set index for all frames, compute common intersection
        for i in range(len(dfs)):
            dfs[i] = dfs[i].set_index(dfs[i].columns[0])

        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)

        if len(common_index) == 0:
            raise ValueError(f"No common index across files in {subdir}")

        # reindex all dfs to the common index
        for i in range(len(dfs)):
            dfs[i] = dfs[i].loc[common_index]

    else:
        # truncate to minimum number of rows
        min_len = min(len(df) for df in dfs)
        dfs = [df.iloc[:min_len].copy() for df in dfs]

    # Numeric columns to average (based on first df)
    numeric_cols = dfs[0].select_dtypes(include=[np.number]).columns

    # stack numeric values into 3D array and compute mean
    stacked = np.stack([df[numeric_cols].to_numpy(dtype=float) for df in dfs], axis=2)
    mean_vals = np.nanmean(stacked, axis=2)

    # build averaged DataFrame starting from first df to preserve structure
    avg = dfs[0].copy()
    avg[numeric_cols] = mean_vals

    # If we had set an index earlier, reset it to include the index column
    if use_index:
        # reset index so it becomes the first column again
        avg = avg.reset_index()

    # ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subdir.name}.csv"
    # write using pandas which will preserve MultiIndex headers
    avg.to_csv(out_path, index=False)


def run(scaled_results_dir: str = 'scaledResults', output_dir: str = 'AveragedResults') -> None:
    base = Path(scaled_results_dir)
    out_base = Path(output_dir)
    if not base.exists():
        raise FileNotFoundError(f"Scaled results directory not found: {base}")

    for entry in sorted(base.iterdir()):
        if entry.is_dir():
            print(f"Averaging CSVs in {entry}")
            average_csvs_in_subdir(entry, out_base)
