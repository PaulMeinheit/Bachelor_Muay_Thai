# AI Coding Guidelines for Bachelor_Muay_Thai Project

## Architecture Overview
This project processes biomechanical data for Muay Thai kick analysis through a modular pipeline:
- **Dataloader**: Loads multi-header CSV files (e.g., tab-separated with skipped rows)
- **Slicer**: Segments raw trial data into individual kick attempts
- **SegmentFinder**: Identifies kick phases (lift-off, impact, foot-down) from ground reaction force (GRF) data
- **Scaler**: Scales each trial to 500 frames across 4 phases using cubic spline interpolation
- **Averager**: Averages multiple scaled trials per data type, handling index alignment and numeric averaging
- **Main**: Orchestrates the pipeline from raw data to averaged results

Data flows: `Sub01_Teep/` → `slicedResults/Sub01_Teep/` → `scaledResults/` → `AveragedResults/`

## Key Patterns
- **Multi-header CSVs**: Use `pd.read_csv(header=[0,1])` for files with two header rows (e.g., `('FP1', 'Z')` columns)
- **File naming**: Data files end in `.txt` but are comma/tab-separated CSVs; output CSVs use `.csv`
- **Directory structure**: Results organized by data type subdirs (e.g., `scaledResults/JointAngles.txt/`) containing `scaled0.csv`, `scaled1.csv`, etc.
- **Spline scaling**: Use `scipy.interpolate.splrep/splev` for resampling time-series to fixed lengths (see `Scaler.spline_interpolate_df`)
- **Segment detection**: Threshold-based detection on GRF Z-axis for kick phases (lift-off <8N, foot-down >15N)
- **Averaging logic**: Detect index columns by name ('Unnamed', 'ITEM') or monotonic values; average numeric columns only

## Dependencies
- pandas (data manipulation, multi-header support)
- numpy (numeric operations)
- scipy (spline interpolation)
- matplotlib (plotting, though not used in core pipeline)

## Workflow
Run the full pipeline: `python code/Main.py` (processes hardcoded `Sub01_Teep` data)

Individual modules can be imported and run separately for debugging/modification.

## Conventions
- Import modules relatively within `code/` directory
- Use `os.path.join` for cross-platform paths
- Handle file creation with `open(path, "x")` followed by `to_csv` to avoid overwrites
- Print progress for long-running operations (e.g., averaging subdirs)</content>
<parameter name="filePath">/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/.github/copilot-instructions.md