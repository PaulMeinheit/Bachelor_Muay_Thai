import Dataloader
import pandas as pd
import os

def test_segment_xyz_loading():
    # Use the same paths as in CoGChanger.py
    rawDataPath = "Raw_Data/N1/teep/"
    angmom_path = os.path.join(rawDataPath, "AngMoms_wrt_LAB.txt")
    cog_path = os.path.join(rawDataPath, "CoG_Position.txt")

    angmom = Dataloader.loadData(angmom_path)
    angmom = angmom.drop(labels=[0,1])
    angmom = angmom.drop(angmom.index[-1])
    angmom = angmom.drop(columns=angmom.columns[0])

    cog = Dataloader.loadData(cog_path)
    cog = cog.drop(labels=[0,1])
    cog = cog.drop(cog.index[-1])
    cog = cog.drop(columns=cog.columns[0])

    # Check that for each segment, X/Y/Z columns exist in both files
    angmom_segments = set([col[0] for col in angmom.columns])
    cog_segments = set([col[0] for col in cog.columns])
    common_segments = angmom_segments & cog_segments
    assert len(common_segments) > 0, "No common segments found between AngMom and CoG files!"

    for seg in common_segments:
        for axis in ['X', 'Y', 'Z']:
            assert (seg, axis) in angmom.columns, f"Missing {(seg, axis)} in AngMom file"
            assert (seg, axis) in cog.columns, f"Missing {(seg, axis)} in CoG file"
        # Check that the data is numeric and has the same length
        angmom_xyz = [angmom[(seg, axis)] for axis in ['X', 'Y', 'Z']]
        cog_xyz = [cog[(seg, axis)] for axis in ['X', 'Y', 'Z']]
        for a, c in zip(angmom_xyz, cog_xyz):
            assert pd.api.types.is_numeric_dtype(a), f"AngMom {seg} axis not numeric"
            assert pd.api.types.is_numeric_dtype(c), f"CoG {seg} axis not numeric"
            assert len(a) == len(c), f"Length mismatch for segment {seg}"
    print("Test passed: All segments and axes loaded correctly and are numeric.")

if __name__ == "__main__":
    test_segment_xyz_loading()
