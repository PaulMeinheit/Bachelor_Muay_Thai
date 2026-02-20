import Dataloader
import pandas
import os
rawDataPath = "Raw_Data/E1/roundhouse/"
rawAngPath = os.path.join(rawDataPath,"AngMoms_wrt_LAB.txt")
rawCoGPath = os.path.join(rawDataPath,"CoG_Position.txt")


loadedAngMomData = Dataloader.loadData(rawAngPath)
loadedAngMomData = loadedAngMomData.drop(labels=[0,1])  
loadedAngMomData = loadedAngMomData.drop(loadedAngMomData.index[-1])
loadedAngMomData = loadedAngMomData.drop(columns=loadedAngMomData.columns[0])  # Drop the first column


loadedCogData = Dataloader.loadData(rawCoGPath)
loadedCogData = loadedCogData.drop(labels=[0,1])
loadedCogData = loadedCogData.drop(loadedCogData.index[-1])
loadedCogData = loadedCogData.drop(columns=loadedCogData.columns[0])  # Drop the first column
first_level_keys = loadedAngMomData.keys()



empty_df = loadedCogData.iloc[0:0]
counter = 0
for key in first_level_keys:
    counter += 1
    if counter != 2:
        continue
    counter = 0
    multipurposeKey = key[0]


# Loop through all segments and load xyz values for each segment from AngMoms_wrt_LAB.txt and CoG_Position.txt
segment_xyz_angmom = {}
segment_xyz_cog = {}


# Match segments by common prefix, handling different suffixes
angmom_suffix = '_AngMom_wrt_LAB'
cog_suffix = '_CoG_pos'

angmom_segments = [col for col in loadedAngMomData.columns.levels[0] if col.endswith(angmom_suffix)]
cog_segments = [col for col in loadedCogData.columns.levels[0] if col.endswith(cog_suffix)]

# Build a mapping from prefix to full segment name for both files
def get_prefix(segname, suffix):
    if segname.endswith(suffix):
        return segname[: -len(suffix)]
    return segname

angmom_prefix_map = {get_prefix(seg, angmom_suffix): seg for seg in angmom_segments}
cog_prefix_map = {get_prefix(seg, cog_suffix): seg for seg in cog_segments}

common_prefixes = set(angmom_prefix_map.keys()) & set(cog_prefix_map.keys())

for prefix in common_prefixes:
    angmom_seg = angmom_prefix_map[prefix]
    cog_seg = cog_prefix_map[prefix]
    # Get xyz values for this segment from AngMoms_wrt_LAB.txt
    if all((angmom_seg, axis) in loadedAngMomData.columns for axis in ['X', 'Y', 'Z']):
        angmom_xyz = (
            loadedAngMomData[(angmom_seg, 'X')],
            loadedAngMomData[(angmom_seg, 'Y')],
            loadedAngMomData[(angmom_seg, 'Z')]
        )
        segment_xyz_angmom[prefix] = angmom_xyz
        print(segment_xyz_angmom[prefix])
    # Get xyz values for this segment from CoG_Position.txt
    if all((cog_seg, axis) in loadedCogData.columns for axis in ['X', 'Y', 'Z']):
        cog_xyz = (
            loadedCogData[(cog_seg, 'X')],
            loadedCogData[(cog_seg, 'Y')],
            loadedCogData[(cog_seg, 'Z')]
        )
        segment_xyz_cog[prefix] = cog_xyz
        
    
# Now segment_xyz_angmom and segment_xyz_cog contain the xyz values for each segment, ready for calculations


