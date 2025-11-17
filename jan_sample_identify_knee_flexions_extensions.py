# * coding: utf8 *
"""
Created on Sat Dec 14

@author: janlau
"""

'''
Jan Lau's PhD Project 1: Longitudinal Right Backleg Roundhouse Kicks
Plots of interest: Model output data

This code:
     Identifies when the knee extensions occur for each of the ten kicks per session
     Identifies when the knee flexions (BEFORE extension) occur for each of the ten kicks per session

This is done because:
	 This will be incorporated into a bigger code that segments each kick into four phases (Preparation, Chambering, Extension, Retraction)

Scaled time: Yes
Kick segmentation: No

Notes:
	 Kicking leg = RIGHT leg
	 Supporting leg = LEFT leg
	 s1, s2, s3: FP1 is supporting leg, FP2 is kicking leg
	 s4, s5, s6, s7: FP4 is supporting leg, FP5 is kicking leg
	 s8: FP5 is supporting leg, FP4 is kicking leg
     expert: FP4 is supporting leg, FP5 is kicking leg
'''

import matplotlib.pyplot as plt
import setup_for_plot_kinematic_data as kinematic_data # Load the data and other setups etc.
import load_biorbd_reconstructed_data as biorbd_recon # keeping it consistent with angular momentum
import scipy # for find_peaks
import numpy as np
import roundhousekicks_frame_segmentation as kick_frames
import math

def obtainFlexionPeaksAndIndices(scaled_time_array, df_trajectory, peak_height, max_width):
    peak_indices = scipy.signal.find_peaks(df_trajectory, height = peak_height, width = max_width)
    # print('peak indices: ', peak_indices[0])
    # print('scaled_time_array: ', scaled_time_array[peak_indices[0]])
    peak_indices_for_scaled_time = scaled_time_array[peak_indices[0]]
    peaks = df_trajectory[peak_indices[0]]
    return([peak_indices[0], peak_indices_for_scaled_time, peaks])

def cleanupFlexionPeaksAndIndicesIfNeeded(indices_og, indices_falsepeaks, kicks_start, kicks_end, scaled_time_array, df_trajectory):
    indices_new = np.setdiff1d(indices_og, indices_falsepeaks)
    indices_new = np.delete(indices_new, np.where(indices_new > kicks_end))
    indices_new = np.delete(indices_new, np.where(indices_new < kicks_start))
    new_indices_for_scaledtime = scaled_time_array[indices_new]
    new_peaks = df_trajectory[indices_new]
    return([indices_new, new_indices_for_scaledtime, new_peaks])

def obtainExtensionPeaksAndIndices(scaled_time_array, df_trajectory, peak_height, max_width):
    peak_indices = scipy.signal.find_peaks(-df_trajectory, height = peak_height, width = max_width)
    # print('peak indices: ', peak_indices[0])
    # print('scaled_time_array: ', scaled_time_array[peak_indices[0]])
    peak_indices_for_scaled_time = scaled_time_array[peak_indices[0]]
    peaks = df_trajectory[peak_indices[0]]
    return([peak_indices[0], peak_indices_for_scaled_time, peaks])

def cleanupExtensionPeaksAndIndicesIfNeeded(indices_og, indices_falsepeaks, kicks_start, kicks_end, scaled_time_array, df_trajectory):
    indices_new = np.setdiff1d(indices_og, indices_falsepeaks)
    indices_new = np.delete(indices_new, np.where(indices_new > kicks_end))
    indices_new = np.delete(indices_new, np.where(indices_new < kicks_start))
    new_indices_for_scaledtime = scaled_time_array[indices_new]
    new_peaks = df_trajectory[indices_new]
    return([indices_new, new_indices_for_scaledtime, new_peaks])

'''Locate MAX KNEE FLEXIONS'''
# Session 1 / Month 1
[knee_flexions_s1_indices, knee_flexions_s1_indices_for_scaledtime, knee_flexions_s1] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s1, biorbd_recon.roundhousekicks_biorbd_q_filtered_s1.loc[:, 'R_Shank_RotY'], 1.5, max_width = 20) # 6
[knee_flexions_s1_indices_falsepeaks, knee_flexions_s1_indices_for_scaledtime_falsepeaks, knee_flexions_s1_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s1, biorbd_recon.roundhousekicks_biorbd_q_filtered_s1.loc[:, 'R_Shank_RotY'], 2.1, max_width = 50) # 6
[knee_flexions_s1_indices, knee_flexions_s1_indices_for_scaledtime, knee_flexions_s1] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_s1_indices, knee_flexions_s1_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s1_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s1_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s1, biorbd_recon.roundhousekicks_biorbd_q_filtered_s1.loc[:, 'R_Shank_RotY'])
# manually_remove_flexions_s1 = [1, 3, 4, 6, 7, 9, 12, 14, 16, 19, 20, 21, 23, 24, 25] # after splitting s1k5 into s1k5a and s1k5b 
manually_remove_flexions_s1 = [1, 3, 4, 6, 7, 9, 10, 12, 14, 16, 19, 20, 21, 23, 24, 25] # before splitting s1k5 into s1k5a and s1k5b
knee_flexions_s1_indices = np.delete(knee_flexions_s1_indices, manually_remove_flexions_s1)
knee_flexions_s1_indices_for_scaledtime = np.delete(knee_flexions_s1_indices_for_scaledtime, manually_remove_flexions_s1)
knee_flexions_s1 = np.delete(knee_flexions_s1, manually_remove_flexions_s1)
# assert(len(knee_flexions_s1) == 11) # after splitting s1k5 into s1k5a and s1k5b
assert(len(knee_flexions_s1) == 10)

# Session 2 / Month 2
[knee_flexions_s2_indices, knee_flexions_s2_indices_for_scaledtime, knee_flexions_s2] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s2, biorbd_recon.roundhousekicks_biorbd_q_filtered_s2.loc[:, 'R_Shank_RotY'], 100*math.pi/180, max_width = 50)
[knee_flexions_s2_indices_falsepeaks, knee_flexions_s2_indices_for_scaledtime_falsepeaks, knee_flexions_s2_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s2, biorbd_recon.roundhousekicks_biorbd_q_filtered_s2.loc[:, 'R_Shank_RotY'], 130*math.pi/180, max_width = 50)
[knee_flexions_s2_indices, knee_flexions_s2_indices_for_scaledtime, knee_flexions_s2] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_s2_indices, knee_flexions_s2_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s2_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s2_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s2, biorbd_recon.roundhousekicks_biorbd_q_filtered_s2.loc[:, 'R_Shank_RotY'])
manually_remove_flexions_s2 = [2, 4, 10, 12, 14]
knee_flexions_s2_indices = np.delete(knee_flexions_s2_indices, manually_remove_flexions_s2)
knee_flexions_s2_indices_for_scaledtime = np.delete(knee_flexions_s2_indices_for_scaledtime, manually_remove_flexions_s2)
knee_flexions_s2 = np.delete(knee_flexions_s2, manually_remove_flexions_s2)
assert(len(knee_flexions_s2) == 10)

# Session 3 / Month 4
[knee_flexions_s3_indices, knee_flexions_s3_indices_for_scaledtime, knee_flexions_s3] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s3, biorbd_recon.roundhousekicks_biorbd_q_filtered_s3.loc[:, 'R_Shank_RotY'], 90*math.pi/180, max_width = 20)
[knee_flexions_s3_indices_falsepeaks, knee_flexions_s3_indices_for_scaledtime_falsepeaks, knee_flexions_s3_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s3, biorbd_recon.roundhousekicks_biorbd_q_filtered_s3.loc[:, 'R_Shank_RotY'], 125*math.pi/180, max_width = 50)
[knee_flexions_s3_indices, knee_flexions_s3_indices_for_scaledtime, knee_flexions_s3] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_s3_indices, knee_flexions_s3_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s3_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s3_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s3, biorbd_recon.roundhousekicks_biorbd_q_filtered_s3.loc[:, 'R_Shank_RotY'])
manually_remove_flexions_s3 = [1, 3, 5, 8]
knee_flexions_s3_indices = np.delete(knee_flexions_s3_indices, manually_remove_flexions_s3)
knee_flexions_s3_indices_for_scaledtime = np.delete(knee_flexions_s3_indices_for_scaledtime, manually_remove_flexions_s3)
knee_flexions_s3 = np.delete(knee_flexions_s3, manually_remove_flexions_s3)
assert(len(knee_flexions_s3) == 10)

# Session 4 / Month 6
[knee_flexions_s4_indices, knee_flexions_s4_indices_for_scaledtime, knee_flexions_s4] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s4, biorbd_recon.roundhousekicks_biorbd_q_filtered_s4.loc[:, 'R_Shank_RotY'], 1.8, max_width = 20)
[knee_flexions_s4_indices_falsepeaks, knee_flexions_s4_indices_for_scaledtime_falsepeaks, knee_flexions_s4_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s4, biorbd_recon.roundhousekicks_biorbd_q_filtered_s4.loc[:, 'R_Shank_RotY'], 128*math.pi/180, max_width = 50)
[knee_flexions_s4_indices, knee_flexions_s4_indices_for_scaledtime, knee_flexions_s4] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_s4_indices, knee_flexions_s4_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s4_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s4_mocap[9,1]),
     biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s4, biorbd_recon.roundhousekicks_biorbd_q_filtered_s4.loc[:, 'R_Shank_RotY'])
manually_remove_flexions_s4 = [2, 3, 5, 7, 9, 11, 12, 15, 16, 18]
knee_flexions_s4_indices = np.delete(knee_flexions_s4_indices, manually_remove_flexions_s4)
knee_flexions_s4_indices_for_scaledtime = np.delete(knee_flexions_s4_indices_for_scaledtime, manually_remove_flexions_s4)
knee_flexions_s4 = np.delete(knee_flexions_s4, manually_remove_flexions_s4)
assert(len(knee_flexions_s4) == 10)

# Session 5 / Month 8
[knee_flexions_s5_indices, knee_flexions_s5_indices_for_scaledtime, knee_flexions_s5] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s5, biorbd_recon.roundhousekicks_biorbd_q_filtered_s5.loc[:, 'R_Shank_RotY'], 1.9, max_width = 20)
[knee_flexions_s5_indices_falsepeaks, knee_flexions_s5_indices_for_scaledtime_falsepeaks, knee_flexions_s5_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s5, biorbd_recon.roundhousekicks_biorbd_q_filtered_s5.loc[:, 'R_Shank_RotY'], 2.1, max_width = 50)
[knee_flexions_s5_indices, knee_flexions_s5_indices_for_scaledtime, knee_flexions_s5] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_s5_indices, knee_flexions_s5_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s5_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s5_mocap[9,1]),
     biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s5, biorbd_recon.roundhousekicks_biorbd_q_filtered_s5.loc[:, 'R_Shank_RotY'])
manually_remove_flexions_s5 = [3, 10]
knee_flexions_s5_indices = np.delete(knee_flexions_s5_indices, manually_remove_flexions_s5)
knee_flexions_s5_indices_for_scaledtime = np.delete(knee_flexions_s5_indices_for_scaledtime, manually_remove_flexions_s5)
knee_flexions_s5 = np.delete(knee_flexions_s5, manually_remove_flexions_s5)
assert(len(knee_flexions_s5) == 10)

# Session 6 / Month 9
[knee_flexions_s6_indices, knee_flexions_s6_indices_for_scaledtime, knee_flexions_s6] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s6, biorbd_recon.roundhousekicks_biorbd_q_filtered_s6.loc[:, 'R_Shank_RotY'], 105*math.pi/180, max_width = 20)
[knee_flexions_s6_indices_falsepeaks, knee_flexions_s6_indices_for_scaledtime_falsepeaks, knee_flexions_s6_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s6, biorbd_recon.roundhousekicks_biorbd_q_filtered_s6.loc[:, 'R_Shank_RotY'], 120*math.pi/180, max_width = 50)
[knee_flexions_s6_indices, knee_flexions_s6_indices_for_scaledtime, knee_flexions_s6] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_s6_indices, knee_flexions_s6_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s6_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s6_mocap[9,1]),
     biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s6, biorbd_recon.roundhousekicks_biorbd_q_filtered_s6.loc[:, 'R_Shank_RotY'])
manually_remove_flexions_s6 = 8
knee_flexions_s6_indices = np.delete(knee_flexions_s6_indices, manually_remove_flexions_s6)
knee_flexions_s6_indices_for_scaledtime = np.delete(knee_flexions_s6_indices_for_scaledtime, manually_remove_flexions_s6)
knee_flexions_s6 = np.delete(knee_flexions_s6, manually_remove_flexions_s6)
assert(len(knee_flexions_s6) == 10)

# Session 7 / Month 10
[knee_flexions_s7_indices, knee_flexions_s7_indices_for_scaledtime, knee_flexions_s7] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s7, biorbd_recon.roundhousekicks_biorbd_q_filtered_s7.loc[:, 'R_Shank_RotY'], 110*math.pi/180, max_width = 30)
[knee_flexions_s7_indices_falsepeaks, knee_flexions_s7_indices_for_scaledtime_falsepeaks, knee_flexions_s7_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s7, biorbd_recon.roundhousekicks_biorbd_q_filtered_s7.loc[:, 'R_Shank_RotY'], 120*math.pi/180, max_width = 50)
[knee_flexions_s7_indices, knee_flexions_s7_indices_for_scaledtime, knee_flexions_s7] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_s7_indices, knee_flexions_s7_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s7_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s7_mocap[9,1]),
     biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s7, biorbd_recon.roundhousekicks_biorbd_q_filtered_s7.loc[:, 'R_Shank_RotY'])
manually_remove_flexions_s7 = 3
knee_flexions_s7_indices = np.delete(knee_flexions_s7_indices, manually_remove_flexions_s7)
knee_flexions_s7_indices_for_scaledtime = np.delete(knee_flexions_s7_indices_for_scaledtime, manually_remove_flexions_s7)
knee_flexions_s7 = np.delete(knee_flexions_s7, manually_remove_flexions_s7)
assert(len(knee_flexions_s7) == 10)

# Session 8 / Month 12
[knee_flexions_s8_indices, knee_flexions_s8_indices_for_scaledtime, knee_flexions_s8] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s8, biorbd_recon.roundhousekicks_biorbd_q_filtered_s8.loc[:, 'R_Shank_RotY'], 1.9, max_width = 30)
[knee_flexions_s8_indices_falsepeaks, knee_flexions_s8_indices_for_scaledtime_falsepeaks, knee_flexions_s8_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s8, biorbd_recon.roundhousekicks_biorbd_q_filtered_s8.loc[:, 'R_Shank_RotY'], 2.15, max_width = 50)
[knee_flexions_s8_indices, knee_flexions_s8_indices_for_scaledtime, knee_flexions_s8] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_s8_indices, knee_flexions_s8_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s8_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s8_mocap[9,1]),
     biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s8, biorbd_recon.roundhousekicks_biorbd_q_filtered_s8.loc[:, 'R_Shank_RotY'])
assert(len(knee_flexions_s8) == 10)

# Expert
[knee_flexions_expert_indices, knee_flexions_expert_indices_for_scaledtime, knee_flexions_expert] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_expert, biorbd_recon.roundhousekicks_biorbd_q_filtered_expert.loc[:, 'R_Shank_RotY'], 125*math.pi/180, max_width = 5) # 125
[knee_flexions_expert_indices_falsepeaks, knee_flexions_expert_indices_for_scaledtime_falsepeaks, knee_flexions_expert_falsepeaks] = obtainFlexionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_expert, biorbd_recon.roundhousekicks_biorbd_q_filtered_expert.loc[:, 'R_Shank_RotY'], 134*math.pi/180, max_width = 40) # 134
[knee_flexions_expert_indices, knee_flexions_expert_indices_for_scaledtime, knee_flexions_expert] = cleanupFlexionPeaksAndIndicesIfNeeded(
    knee_flexions_expert_indices, knee_flexions_expert_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_expert_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_expert_mocap[9,1]),
     biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_expert, biorbd_recon.roundhousekicks_biorbd_q_filtered_expert.loc[:, 'R_Shank_RotY'])
manually_remove_flexions_expert = [5, 7, 9, 11, 13]
knee_flexions_expert_indices = np.delete(knee_flexions_expert_indices, manually_remove_flexions_expert)
knee_flexions_expert_indices_for_scaledtime = np.delete(knee_flexions_expert_indices_for_scaledtime, manually_remove_flexions_expert)
knee_flexions_expert = np.delete(knee_flexions_expert, manually_remove_flexions_expert)
assert(len(knee_flexions_expert) == 10)

''' Locate KNEE EXTENSIONS '''
# Session 1 / Month 1
[knee_extensions_s1_indices, knee_extensions_s1_indices_for_scaledtime, knee_extensions_s1] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s1, biorbd_recon.roundhousekicks_biorbd_q_filtered_s1.loc[:, 'R_Shank_RotY'], -0.25, max_width = 6) # 6 deg for modeloutputs
[knee_extensions_s1_indices_falsepeaks, knee_extensions_s1_indices_for_scaledtime_falsepeaks, knee_extensions_s1_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s1, biorbd_recon.roundhousekicks_biorbd_q_filtered_s1.loc[:, 'R_Shank_RotY'], -0.1, max_width = 40)
[knee_extensions_s1_indices, knee_extensions_s1_indices_for_scaledtime, knee_extensions_s1] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_s1_indices, knee_extensions_s1_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s1_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s1_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s1, biorbd_recon.roundhousekicks_biorbd_q_filtered_s1.loc[:, 'R_Shank_RotY'])
manually_remove_extensions_s1 = [4, 5, 10]
knee_extensions_s1_indices = np.delete(knee_extensions_s1_indices, manually_remove_extensions_s1)
knee_extensions_s1_indices_for_scaledtime = np.delete(knee_extensions_s1_indices_for_scaledtime, manually_remove_extensions_s1)
knee_extensions_s1 = np.delete(knee_extensions_s1, manually_remove_extensions_s1)
assert(len(knee_extensions_s1) == 10)

# Session 2 / Month 2
[knee_extensions_s2_indices, knee_extensions_s2_indices_for_scaledtime, knee_extensions_s2] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s2, biorbd_recon.roundhousekicks_biorbd_q_filtered_s2.loc[:, 'R_Shank_RotY'], -0.4, max_width = 6)
[knee_extensions_s2_indices_falsepeaks, knee_extensions_s2_indices_for_scaledtime_falsepeaks, knee_extensions_s2_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s2, biorbd_recon.roundhousekicks_biorbd_q_filtered_s2.loc[:, 'R_Shank_RotY'], -0.2, max_width = 20)
[knee_extensions_s2_indices, knee_extensions_s2_indices_for_scaledtime, knee_extensions_s2] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_s2_indices, knee_extensions_s2_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s2_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s2_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s2, biorbd_recon.roundhousekicks_biorbd_q_filtered_s2.loc[:, 'R_Shank_RotY'])
manually_remove_extensions_s2 = [2, 5]
knee_extensions_s2_indices = np.delete(knee_extensions_s2_indices, manually_remove_extensions_s2)
knee_extensions_s2_indices_for_scaledtime = np.delete(knee_extensions_s2_indices_for_scaledtime, manually_remove_extensions_s2)
knee_extensions_s2 = np.delete(knee_extensions_s2, manually_remove_extensions_s2)
assert(len(knee_extensions_s2) == 10)

# Session 3 / Month 4
[knee_extensions_s3_indices, knee_extensions_s3_indices_for_scaledtime, knee_extensions_s3] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s3, biorbd_recon.roundhousekicks_biorbd_q_filtered_s3.loc[:, 'R_Shank_RotY'], -0.3, max_width = 6)
[knee_extensions_s3_indices_falsepeaks, knee_extensions_s3_indices_for_scaledtime_falsepeaks, knee_extensions_s3_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s3, biorbd_recon.roundhousekicks_biorbd_q_filtered_s3.loc[:, 'R_Shank_RotY'], -0.1, max_width = 40)
[knee_extensions_s3_indices, knee_extensions_s3_indices_for_scaledtime, knee_extensions_s3] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_s3_indices, knee_extensions_s3_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s3_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s3_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s3, biorbd_recon.roundhousekicks_biorbd_q_filtered_s3.loc[:, 'R_Shank_RotY'])
manually_remove_extensions_s3 = 1
knee_extensions_s3_indices = np.delete(knee_extensions_s3_indices, manually_remove_extensions_s3)
knee_extensions_s3_indices_for_scaledtime = np.delete(knee_extensions_s3_indices_for_scaledtime, manually_remove_extensions_s3)
knee_extensions_s3 = np.delete(knee_extensions_s3, manually_remove_extensions_s3)
assert(len(knee_extensions_s3) == 10)

# Session 4 / Month 6
[knee_extensions_s4_indices, knee_extensions_s4_indices_for_scaledtime, knee_extensions_s4] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s4, biorbd_recon.roundhousekicks_biorbd_q_filtered_s4.loc[:, 'R_Shank_RotY'], -0.5, max_width = 6)
[knee_extensions_s4_indices_falsepeaks, knee_extensions_s4_indices_for_scaledtime_falsepeaks, knee_extensions_s4_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s4, biorbd_recon.roundhousekicks_biorbd_q_filtered_s4.loc[:, 'R_Shank_RotY'], 0.3, max_width = 50)
[knee_extensions_s4_indices, knee_extensions_s4_indices_for_scaledtime, knee_extensions_s4] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_s4_indices, knee_extensions_s4_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s4_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s4_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s4, biorbd_recon.roundhousekicks_biorbd_q_filtered_s4.loc[:, 'R_Shank_RotY'])
manually_remove_extensions_s4 = [0, 2, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 21]
knee_extensions_s4_indices = np.delete(knee_extensions_s4_indices, manually_remove_extensions_s4)
knee_extensions_s4_indices_for_scaledtime = np.delete(knee_extensions_s4_indices_for_scaledtime, manually_remove_extensions_s4)
knee_extensions_s4 = np.delete(knee_extensions_s4, manually_remove_extensions_s4)
assert(len(knee_extensions_s4) == 10)

# Session 5 / Month 8
[knee_extensions_s5_indices, knee_extensions_s5_indices_for_scaledtime, knee_extensions_s5] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s5, biorbd_recon.roundhousekicks_biorbd_q_filtered_s5.loc[:, 'R_Shank_RotY'], -0.5, max_width = 6)
[knee_extensions_s5_indices_falsepeaks, knee_extensions_s5_indices_for_scaledtime_falsepeaks, knee_extensions_s5_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s5, biorbd_recon.roundhousekicks_biorbd_q_filtered_s5.loc[:, 'R_Shank_RotY'], -0.3, max_width = 50)
[knee_extensions_s5_indices, knee_extensions_s5_indices_for_scaledtime, knee_extensions_s5] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_s5_indices, knee_extensions_s5_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s5_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s5_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s5, biorbd_recon.roundhousekicks_biorbd_q_filtered_s5.loc[:, 'R_Shank_RotY'])
manually_remove_extensions_s5 = [0, 1, 3, 4, 5, 7, 8, 11, 13, 15, 17, 18, 20, 22, 23, 24]
knee_extensions_s5_indices = np.delete(knee_extensions_s5_indices, manually_remove_extensions_s5)
knee_extensions_s5_indices_for_scaledtime = np.delete(knee_extensions_s5_indices_for_scaledtime, manually_remove_extensions_s5)
knee_extensions_s5 = np.delete(knee_extensions_s5, manually_remove_extensions_s5)
assert(len(knee_extensions_s5) == 10)

# Session 6 / Month 9
[knee_extensions_s6_indices, knee_extensions_s6_indices_for_scaledtime, knee_extensions_s6] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s6, biorbd_recon.roundhousekicks_biorbd_q_filtered_s6.loc[:, 'R_Shank_RotY'], -0.5, max_width = 6)
[knee_extensions_s6_indices_falsepeaks, knee_extensions_s6_indices_for_scaledtime_falsepeaks, knee_extensions_s6_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s6, biorbd_recon.roundhousekicks_biorbd_q_filtered_s6.loc[:, 'R_Shank_RotY'], -0.2, max_width = 50)
[knee_extensions_s6_indices, knee_extensions_s6_indices_for_scaledtime, knee_extensions_s6] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_s6_indices, knee_extensions_s6_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s6_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s6_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s6, biorbd_recon.roundhousekicks_biorbd_q_filtered_s6.loc[:, 'R_Shank_RotY'])
manually_remove_extensions_s6 = [0, 2, 4, 5, 7, 8, 10, 11, 13, 15, 17, 18, 20, 22, 23, 25]
knee_extensions_s6_indices = np.delete(knee_extensions_s6_indices, manually_remove_extensions_s6)
knee_extensions_s6_indices_for_scaledtime = np.delete(knee_extensions_s6_indices_for_scaledtime, manually_remove_extensions_s6)
knee_extensions_s6 = np.delete(knee_extensions_s6, manually_remove_extensions_s6)
assert(len(knee_extensions_s6) == 10)

# Session 7 / Month 10
[knee_extensions_s7_indices, knee_extensions_s7_indices_for_scaledtime, knee_extensions_s7] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s7, biorbd_recon.roundhousekicks_biorbd_q_filtered_s7.loc[:, 'R_Shank_RotY'], -0.6, max_width = 6)
[knee_extensions_s7_indices_falsepeaks, knee_extensions_s7_indices_for_scaledtime_falsepeaks, knee_extensions_s7_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s7, biorbd_recon.roundhousekicks_biorbd_q_filtered_s7.loc[:, 'R_Shank_RotY'], -0.1, max_width = 50)
[knee_extensions_s7_indices, knee_extensions_s7_indices_for_scaledtime, knee_extensions_s7] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_s7_indices, knee_extensions_s7_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s7_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s7_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s7, biorbd_recon.roundhousekicks_biorbd_q_filtered_s7.loc[:, 'R_Shank_RotY'])
manually_remove_extensions_s7 = [0, 2, 3, 5, 7, 9, 11, 14, 15, 17, 19, 20, 21]
knee_extensions_s7_indices = np.delete(knee_extensions_s7_indices, manually_remove_extensions_s7)
knee_extensions_s7_indices_for_scaledtime = np.delete(knee_extensions_s7_indices_for_scaledtime, manually_remove_extensions_s7)
knee_extensions_s7 = np.delete(knee_extensions_s7, manually_remove_extensions_s7)
assert(len(knee_extensions_s7) == 10)

# Session 8 / Month 12
[knee_extensions_s8_indices, knee_extensions_s8_indices_for_scaledtime, knee_extensions_s8] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s8, biorbd_recon.roundhousekicks_biorbd_q_filtered_s8.loc[:, 'R_Shank_RotY'], -0.5, max_width = 6)
[knee_extensions_s8_indices_falsepeaks, knee_extensions_s8_indices_for_scaledtime_falsepeaks, knee_extensions_s8_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s8, biorbd_recon.roundhousekicks_biorbd_q_filtered_s8.loc[:, 'R_Shank_RotY'], -0.4, max_width = 50)
[knee_extensions_s8_indices, knee_extensions_s8_indices_for_scaledtime, knee_extensions_s8] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_s8_indices, knee_extensions_s8_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_s8_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_s8_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s8, biorbd_recon.roundhousekicks_biorbd_q_filtered_s8.loc[:, 'R_Shank_RotY'])
# manually_remove_extensions_s8 = [3, 6, 9, 12, 13] # before fixing the weird upperbody origins
manually_remove_extensions_s8 = [0, 4, 5, 7, 9, 12, 15, 16] # after fixing the weird upperbody origins, for comz[0] = 0.93217
knee_extensions_s8_indices = np.delete(knee_extensions_s8_indices, manually_remove_extensions_s8)
knee_extensions_s8_indices_for_scaledtime = np.delete(knee_extensions_s8_indices_for_scaledtime, manually_remove_extensions_s8)
knee_extensions_s8 = np.delete(knee_extensions_s8, manually_remove_extensions_s8)
assert(len(knee_extensions_s8) == 10)

# Expert
[knee_extensions_expert_indices, knee_extensions_expert_indices_for_scaledtime, knee_extensions_expert] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_expert, biorbd_recon.roundhousekicks_biorbd_q_filtered_expert.loc[:, 'R_Shank_RotY'], -0.4, max_width = 6)
[knee_extensions_expert_indices_falsepeaks, knee_extensions_expert_indices_for_scaledtime_falsepeaks, knee_extensions_expert_falsepeaks] = obtainExtensionPeaksAndIndices(
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_expert, biorbd_recon.roundhousekicks_biorbd_q_filtered_expert.loc[:, 'R_Shank_RotY'], -0.05, max_width = 50)
[knee_extensions_expert_indices, knee_extensions_expert_indices_for_scaledtime, knee_extensions_expert] = cleanupExtensionPeaksAndIndicesIfNeeded(
    knee_extensions_expert_indices, knee_extensions_expert_indices_falsepeaks,
    int(kick_frames.roundhousekicks_start_end_frames_expert_mocap[0,0]), int(kick_frames.roundhousekicks_start_end_frames_expert_mocap[9,1]),
    biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_expert, biorbd_recon.roundhousekicks_biorbd_q_filtered_expert.loc[:, 'R_Shank_RotY'])
manually_remove_extensions_expert = [0, 2, 4, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 22, 23, 25, 26, 28]
knee_extensions_expert_indices = np.delete(knee_extensions_expert_indices, manually_remove_extensions_expert)
knee_extensions_expert_indices_for_scaledtime = np.delete(knee_extensions_expert_indices_for_scaledtime, manually_remove_extensions_expert)
knee_extensions_expert = np.delete(knee_extensions_expert, manually_remove_extensions_expert)
assert(len(knee_extensions_expert) == 10)

'''Plot Kicking Leg Flexions and Extensions'''
asis_vs_qknee, ((asis_vs_qknee_s1, asis_vs_qknee_s2, asis_vs_qknee_s3),
                (asis_vs_qknee_s4, asis_vs_qknee_s5, asis_vs_qknee_s6),
                (asis_vs_qknee_s7, asis_vs_qknee_s8, asis_vs_qknee_expert)) = plt.subplots(3,3)
asis_vs_qknee.set_figheight(10)
asis_vs_qknee.set_figwidth(20)

asis_vs_qknee_s1.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s1, biorbd_recon.roundhousekicks_biorbd_q_filtered_s1.loc[:, "R_Shank_RotY"], color = biorbd_recon.s12k5_color)
asis_vs_qknee_s1.plot(knee_extensions_s1_indices_for_scaledtime, knee_extensions_s1, 'x')
asis_vs_qknee_s1.plot(knee_flexions_s1_indices_for_scaledtime, knee_flexions_s1, 'o')
asis_vs_qknee_s1.set_title('Month 1, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_s1.loc[100, "ASIS_Width_Scalar"])

asis_vs_qknee_s2.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s2, biorbd_recon.roundhousekicks_biorbd_q_filtered_s2.loc[:, "R_Shank_RotY"], color = biorbd_recon.s12k5_color)
asis_vs_qknee_s2.plot(knee_extensions_s2_indices_for_scaledtime, knee_extensions_s2, 'x')
asis_vs_qknee_s2.plot(knee_flexions_s2_indices_for_scaledtime, knee_flexions_s2, 'o')
asis_vs_qknee_s2.set_title('Month 2, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_s2.loc[100, "ASIS_Width_Scalar"])

asis_vs_qknee_s3.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s3, biorbd_recon.roundhousekicks_biorbd_q_filtered_s3.loc[:, "R_Shank_RotY"], color = biorbd_recon.s34k5_color)
asis_vs_qknee_s3.plot(knee_extensions_s3_indices_for_scaledtime, knee_extensions_s3, 'x')
asis_vs_qknee_s3.plot(knee_flexions_s3_indices_for_scaledtime, knee_flexions_s3, 'o')
asis_vs_qknee_s3.set_title('Month 4, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_s3.loc[100, "ASIS_Width_Scalar"])

asis_vs_qknee_s4.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s4, biorbd_recon.roundhousekicks_biorbd_q_filtered_s4.loc[:, "R_Shank_RotY"], color = biorbd_recon.s34k5_color)
asis_vs_qknee_s4.plot(knee_extensions_s4_indices_for_scaledtime, knee_extensions_s4, 'x')
asis_vs_qknee_s4.plot(knee_flexions_s4_indices_for_scaledtime, knee_flexions_s4, 'o')
asis_vs_qknee_s4.set_title('Month 6, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_s4.loc[80, "ASIS_Width_Scalar"])

asis_vs_qknee_s5.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s5, biorbd_recon.roundhousekicks_biorbd_q_filtered_s5.loc[:, "R_Shank_RotY"], color = biorbd_recon.s567k5_color)
asis_vs_qknee_s5.plot(knee_extensions_s5_indices_for_scaledtime, knee_extensions_s5, 'x')
asis_vs_qknee_s5.plot(knee_flexions_s5_indices_for_scaledtime, knee_flexions_s5, 'o')
asis_vs_qknee_s5.set_title('Month 8, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_s5.loc[100, "ASIS_Width_Scalar"])

asis_vs_qknee_s6.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s6, biorbd_recon.roundhousekicks_biorbd_q_filtered_s6.loc[:, "R_Shank_RotY"], color = biorbd_recon.s567k5_color)
asis_vs_qknee_s6.plot(knee_extensions_s6_indices_for_scaledtime, knee_extensions_s6, 'x')
asis_vs_qknee_s6.plot(knee_flexions_s6_indices_for_scaledtime, knee_flexions_s6, 'o')
asis_vs_qknee_s6.set_title('Month 9, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_s6.loc[100, "ASIS_Width_Scalar"])

asis_vs_qknee_s7.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s7, biorbd_recon.roundhousekicks_biorbd_q_filtered_s7.loc[:, "R_Shank_RotY"], color = biorbd_recon.s567k5_color)
asis_vs_qknee_s7.plot(knee_extensions_s7_indices_for_scaledtime, knee_extensions_s7, 'x')
asis_vs_qknee_s7.plot(knee_flexions_s7_indices_for_scaledtime, knee_flexions_s7, 'o')
asis_vs_qknee_s7.set_title('Month 10, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_s7.loc[80, "ASIS_Width_Scalar"])

asis_vs_qknee_s8.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_s8, biorbd_recon.roundhousekicks_biorbd_q_filtered_s8.loc[:, "R_Shank_RotY"], color = biorbd_recon.s8k5_color)
asis_vs_qknee_s8.plot(knee_extensions_s8_indices_for_scaledtime, knee_extensions_s8, 'x')
asis_vs_qknee_s8.plot(knee_flexions_s8_indices_for_scaledtime, knee_flexions_s8, 'o')
asis_vs_qknee_s8.set_title('Month 12, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_s8.loc[50, "ASIS_Width_Scalar"])

asis_vs_qknee_expert.plot(biorbd_recon.scaled_t_array_roundhousekicks_kinematic_raw_expert, biorbd_recon.roundhousekicks_biorbd_q_filtered_expert.loc[:, "R_Shank_RotY"], color = biorbd_recon.expertk10_color)
asis_vs_qknee_expert.plot(knee_extensions_expert_indices_for_scaledtime, knee_extensions_expert, 'x')
asis_vs_qknee_expert.plot(knee_flexions_expert_indices_for_scaledtime, knee_flexions_expert, 'o')
# asis_vs_qknee_expert.set_title('Expert for Coomparison, ASIS Width = %imm' %kinematic_data.roundhousekicks_modeloutputs_filtered_expert.loc[100, "ASIS_Width_Scalar"])

asis_vs_qknee.suptitle('Identification of Knee Flexions and Extensions in All Sessions', fontsize = biorbd_recon.standaloneplot_title_font_size) #, y = 0.92)
asis_vs_qknee.supxlabel('Scaled Time', fontsize = biorbd_recon.standaloneplot_axes_font_size) #, y = 0.075)
asis_vs_qknee.supylabel('Angle (deg)', fontsize = biorbd_recon.standaloneplot_axes_font_size) #, x = 0.075)
asis_vs_qknee.savefig('modeloutputs_plots/knee_flexions_extensions_filtered.jpg')
print("Figure saved in modeloutputs_plots folder as knee_flexions_extensions_filtered.jpg")
plt.close(asis_vs_qknee)