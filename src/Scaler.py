import Dataloader
import os
import pandas as pd
import numpy as np

def scaleDirectoryToFourPhases(input_dir, segments, output_dir, output_subdir):
    data = Dataloader.load_csvs_from_dir(input_dir)
    scaleToFourPhases(data, segments, output_dir, output_subdir)
    
def scaleToFourPhases(data, segments, output_dir, output_subdir):
    i = 0
    for filename, df in data.items():
        # Each file corresponds to one segment at index i
        scaled_df = scaleDataFrameToFourPhases(df, segments[i])
        output_path = os.path.join(output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        # Preserve MultiIndex header structure when writing CSV
        scaled_df.to_csv(os.path.join(output_path, "scaled" + str(i)), index=False, header=True)
        i += 1

def scaleDataFrameToFourPhases(df, segment):
    # segment is [phase0_start, phase0_end, phase1_end, phase2_end, phase3_end]
    # We resample each phase to a fixed number of frames (25 each -> 100 total)
    num_frames_per_phase = 25
    
    phase0 = df.iloc[segment[0]:segment[1]]
    phase1 = df.iloc[segment[1]:segment[2]]
    phase2 = df.iloc[segment[2]:segment[3]]
    phase3 = df.iloc[segment[3]:segment[4]]
    
    scaled_phase0 = resamplePhase(phase0, num_frames_per_phase)
    scaled_phase1 = resamplePhase(phase1, num_frames_per_phase)
    scaled_phase2 = resamplePhase(phase2, num_frames_per_phase)
    scaled_phase3 = resamplePhase(phase3, num_frames_per_phase)
    
    scaled_phases = [scaled_phase0, scaled_phase1, scaled_phase2, scaled_phase3]
    
    return pd.concat(scaled_phases, ignore_index=True)

def resamplePhase(phase_df, num_frames):
    # Resample the phase dataframe to num_frames using interpolation
    if len(phase_df) == 0:
        print("empty phase encountered during resampling")
        return pd.DataFrame()  # return empty if no data in phase

    # Preserve original column MultiIndex structure
    cols = phase_df.columns
    is_multiindex = isinstance(phase_df.columns, pd.MultiIndex)
    
    orig_n = len(phase_df)

    # positions for original and target
    x_orig = np.linspace(0, orig_n - 1, orig_n)
    x_new = np.linspace(0, orig_n - 1, num_frames)

    out_dict = {}
    for c in cols:
        y = phase_df[c].to_numpy(dtype=float)
        y_new = np.interp(x_new, x_orig, y)
        out_dict[c] = y_new

    out_df = pd.DataFrame(out_dict, columns=cols)
    
    # Restore MultiIndex if input had it
    if is_multiindex:
        out_df.columns = pd.MultiIndex.from_tuples(out_df.columns)
    
    return out_df.reset_index(drop=True)