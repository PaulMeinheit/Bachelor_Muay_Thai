# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:40:32 2024

@author: janlau
"""

'''
Jan Lau's PhD Project 1: Longitudinal Right Back-leg Roundhouse Kicks
Plots of interest: RESAMPLED force plate data on when KICKING foot is on or off the force plate

This code:
	- Identifies when the kicking foot LEAVES the FP to enter the Chambering phase
    - Identifies when the kicking foot MAKES CONTACT with the FP after the strike
    
Something to ponder for the future: Should I really be taking foot off and foot on FP moments as the actual start/end indices of each kick?

Scaled time: Yes
Kick segmentation: No

Notes:
	- s1_kick5, s2_kick10, and s3_kick4 for sure cannot be considered as good/stable kicks (there might be more, but TBD)
	- Kicking leg = RIGHT leg
	- Supporting leg = LEFT leg
	- s1, s2, s3: FP1 is supporting leg, FP2 is kicking leg
	- s4, s5, s6, s7: FP4 is supporting leg, FP5 is kicking leg
	- s8: FP5 is supporting leg, FP4 is kicking leg
	- expert: FP4 is supporting leg, FP5 is kicking leg
'''

import matplotlib.pyplot as plt
import setup_for_plot_fp_data as fp_data # Load the data and other setups etc.
import roundhousekicks_frame_segmentation as kick_frames
import numpy as np
import pandas as pd
import IPython

''' Identify kicking foot OFF FP'''
def find_foot_on_off_FP_idx(session, start_idx, end_idx, fp_df):
	kick_window = fp_df.loc[start_idx:end_idx, :]

	if session == "s1" or session == "s2" or session == "s3":
		# Find foot off FP instances
		footoff_idxs = kick_window.index[kick_window.loc[start_idx:end_idx, "FP2_Fz"] >= -4]
		footoff_Fzvals = kick_window.loc[footoff_idxs, "FP2_Fz"]

		# Find foot on FP instance
		footon_idx = footoff_idxs[-1] + 1
		footon_Fzval = kick_window.loc[footon_idx, "FP2_Fz"]

	elif session == "s4" or session == "s5" or session == "s6" or session == "s7" or session == "expert":
		# Find foot off FP instances
		footoff_idxs = kick_window.index[kick_window.loc[start_idx:end_idx, "FP5_Fz"] >= -4]
		footoff_Fzvals = kick_window.loc[footoff_idxs, "FP5_Fz"]

		# Find foot on FP instance
		footon_idx = footoff_idxs[-1] + 1
		footon_Fzval = kick_window.loc[footon_idx, "FP5_Fz"]

	else:
		# Find foot off FP instances
		footoff_idxs = kick_window.index[kick_window.loc[start_idx:end_idx, "FP4_Fz"] >= -4]
		footoff_Fzvals = kick_window.loc[footoff_idxs, "FP4_Fz"]

		# Find foot on FP instance
		footon_idx = footoff_idxs[-1] + 1
		footon_Fzval = kick_window.loc[footon_idx, "FP4_Fz"]

	# Re-express indices as scaled time
	footoff_scaledidxs = (footoff_idxs - start_idx) / (end_idx - start_idx)
	footon_scaledidx = (footon_idx - start_idx) / (end_idx - start_idx)

	### Determine the specific instances for foot off and foot on ###
	# # First, we handle the specific cases # #
	''' For s1_kick5, there are TWO foot-off-foot-on instances '''
	if session == "s1" and start_idx == kick_frames.roundhousekicks_start_end_frames_s1_mocap[4,0]: # s1_kick5 has two instances
		# Identify where the gap occurs between the two instances of foot off
		gaps = np.ndarray([2])
		i_gap = 0
		curr_idx_val = footoff_idxs[0]
		for next_idx_val in footoff_idxs[1:]:
			i = 0
			if next_idx_val - curr_idx_val > 5:
				gaps[i_gap] = curr_idx_val
				gaps[i_gap + 1] = next_idx_val
				i_gap = i_gap + 1
				i = i + 1
				curr_idx_val = next_idx_val
			curr_idx_val = next_idx_val
			i = i + 1
		
		# First instance foot off FP
		footoff_idx1 = footoff_idxs[0]
		footoff_scaledidx1 = (footoff_idx1 - start_idx) / (end_idx - start_idx)
		footoff_Fzval1 = kick_window.loc[footoff_idx1, "FP2_Fz"]
		print("\nFirst instance of foot off FP in s1k5: ", footoff_idx1, " (actual idx) / ", footoff_scaledidx1, " (scaled idx)")

		# First instance foot on FP
		footon_idx1 = gaps[0] + 1
		footon_scaledidx1 = (footon_idx1 - start_idx) / (end_idx - start_idx)
		footon_Fzval1 = kick_window.loc[footon_idx1, "FP2_Fz"]
		print("\nFirst instance of foot on FP in s1k5: ", footon_idx1, " (actual idx) / ", footon_scaledidx1, " (scaled idx)")

		# Second instance foot off FP
		footoff_idx2 = gaps[1]
		footoff_scaledidx2 = (footoff_idx2 - start_idx) / (end_idx - start_idx)
		footoff_Fzval2 = kick_window.loc[footoff_idx2, "FP2_Fz"]
		print("\nSecond instance of foot off FP in s1k5: ", footoff_idx2, " (actual idx) / ", footoff_scaledidx2, " (scaled idx)")

		# Second instance foot on FP
		footon_idx2 = footoff_idxs[-1] + 1
		footon_scaledidx2 = (footon_idx2 - start_idx) / (end_idx - start_idx)
		footon_Fzval2 = kick_window.loc[footon_idx2, "FP2_Fz"]
		print("\nSecond instance of foot on FP in s1k5: ", footon_idx2, " (actual idx) / ", footon_scaledidx2, " (scaled idx)")

		# Return the two sets of values
		return([footoff_idx1, footoff_Fzval1, footoff_scaledidx1, footon_idx1, footon_Fzval1, footon_scaledidx1,
		  footoff_idx2, footoff_Fzval2, footoff_scaledidx2, footon_idx2, footon_Fzval2, footon_scaledidx2, kick_window])

	# # For general cases, we take the first foot off instance to denote the start of Chambering phase
	footoff_idx = footoff_idxs[0]
	footoff_scaledidx = footoff_scaledidxs[0]
	footoff_Fzval = footoff_Fzvals[:1]

	return [footoff_idx, footoff_Fzval, footoff_scaledidx, footon_idx, footon_Fzval, footon_scaledidx, kick_window]

# Session 1
[s1_kick1_footoff_idx, s1_kick1_footoff_Fzval, s1_kick1_footoff_scaledidx, s1_kick1_footon_idx, s1_kick1_footon_Fzval, s1_kick1_footon_scaledidx, s1_kick1_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick2_footoff_idx, s1_kick2_footoff_Fzval, s1_kick2_footoff_scaledidx, s1_kick2_footon_idx, s1_kick2_footon_Fzval, s1_kick2_footon_scaledidx, s1_kick2_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick3_footoff_idx, s1_kick3_footoff_Fzval, s1_kick3_footoff_scaledidx, s1_kick3_footon_idx, s1_kick3_footon_Fzval, s1_kick3_footon_scaledidx, s1_kick3_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick4_footoff_idx, s1_kick4_footoff_Fzval, s1_kick4_footoff_scaledidx, s1_kick4_footon_idx, s1_kick4_footon_Fzval, s1_kick4_footon_scaledidx, s1_kick4_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick5_footoff_idx1, s1_kick5_footoff_Fzval1, s1_kick5_footoff_scaledidx1, s1_kick5_footon_idx1, s1_kick5_footon_Fzval1, s1_kick5_footon_scaledidx1, s1_kick5_footoff_idx2, 
 s1_kick5_footoff_Fzval2, s1_kick5_footoff_scaledidx2, s1_kick5_footon_idx2, s1_kick5_footon_Fzval2, s1_kick5_footon_scaledidx2, s1_kick5_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick6_footoff_idx, s1_kick6_footoff_Fzval, s1_kick6_footoff_scaledidx, s1_kick6_footon_idx, s1_kick6_footon_Fzval, s1_kick6_footon_scaledidx, s1_kick6_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick7_footoff_idx, s1_kick7_footoff_Fzval, s1_kick7_footoff_scaledidx, s1_kick7_footon_idx, s1_kick7_footon_Fzval, s1_kick7_footon_scaledidx, s1_kick7_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick8_footoff_idx, s1_kick8_footoff_Fzval, s1_kick8_footoff_scaledidx, s1_kick8_footon_idx, s1_kick8_footon_Fzval, s1_kick8_footon_scaledidx, s1_kick8_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick9_footoff_idx, s1_kick9_footoff_Fzval, s1_kick9_footoff_scaledidx, s1_kick9_footon_idx, s1_kick9_footon_Fzval, s1_kick9_footon_scaledidx, s1_kick9_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_s1)

[s1_kick10_footoff_idx, s1_kick10_footoff_Fzval, s1_kick10_footoff_scaledidx, s1_kick10_footon_idx, s1_kick10_footon_Fzval, s1_kick10_footon_scaledidx, s1_kick10_window_fp] = find_foot_on_off_FP_idx(
	"s1", kick_frames.roundhousekicks_start_end_frames_s1_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_s1_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_s1)

# Session 2
[s2_kick1_footoff_idx, s2_kick1_footoff_Fzval, s2_kick1_footoff_scaledidx, s2_kick1_footon_idx, s2_kick1_footon_Fzval, s2_kick1_footon_scaledidx, s2_kick1_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_s2)

[s2_kick2_footoff_idx, s2_kick2_footoff_Fzval, s2_kick2_footoff_scaledidx, s2_kick2_footon_idx, s2_kick2_footon_Fzval, s2_kick2_footon_scaledidx, s2_kick2_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_s2)

[s2_kick3_footoff_idx, s2_kick3_footoff_Fzval, s2_kick3_footoff_scaledidx, s2_kick3_footon_idx, s2_kick3_footon_Fzval, s2_kick3_footon_scaledidx, s2_kick3_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_s2)

[s2_kick4_footoff_idx, s2_kick4_footoff_Fzval, s2_kick4_footoff_scaledidx, s2_kick4_footon_idx, s2_kick4_footon_Fzval, s2_kick4_footon_scaledidx, s2_kick4_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_s2)

[s2_kick5_footoff_idx, s2_kick5_footoff_Fzval, s2_kick5_footoff_scaledidx, s2_kick5_footon_idx, s2_kick5_footon_Fzval, s2_kick5_footon_scaledidx, s2_kick5_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_s2)

[s2_kick6_footoff_idx, s2_kick6_footoff_Fzval, s2_kick6_footoff_scaledidx, s2_kick6_footon_idx, s2_kick6_footon_Fzval, s2_kick6_footon_scaledidx, s2_kick6_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_s2)

[s2_kick7_footoff_idx, s2_kick7_footoff_Fzval, s2_kick7_footoff_scaledidx, s2_kick7_footon_idx, s2_kick7_footon_Fzval, s2_kick7_footon_scaledidx, s2_kick7_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_s2)

[s2_kick8_footoff_idx, s2_kick8_footoff_Fzval, s2_kick8_footoff_scaledidx, s2_kick8_footon_idx, s2_kick8_footon_Fzval, s2_kick8_footon_scaledidx, s2_kick8_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_s2)

[s2_kick9_footoff_idx, s2_kick9_footoff_Fzval, s2_kick9_footoff_scaledidx, s2_kick9_footon_idx, s2_kick9_footon_Fzval, s2_kick9_footon_scaledidx, s2_kick9_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_s2)

# Exclude s2_kick10 from the good/stable kick analysis
[s2_kick10_footoff_idx, s2_kick10_footoff_Fzval, s2_kick10_footoff_scaledidx, s2_kick10_footon_idx, s2_kick10_footon_Fzval, s2_kick10_footon_scaledidx, s2_kick10_window_fp] = find_foot_on_off_FP_idx(
	"s2", kick_frames.roundhousekicks_start_end_frames_s2_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_s2_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_s2)

# Session 3
[s3_kick1_footoff_idx, s3_kick1_footoff_Fzval, s3_kick1_footoff_scaledidx, s3_kick1_footon_idx, s3_kick1_footon_Fzval, s3_kick1_footon_scaledidx, s3_kick1_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_s3)

[s3_kick2_footoff_idx, s3_kick2_footoff_Fzval, s3_kick2_footoff_scaledidx, s3_kick2_footon_idx, s3_kick2_footon_Fzval, s3_kick2_footon_scaledidx, s3_kick2_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_s3)

[s3_kick3_footoff_idx, s3_kick3_footoff_Fzval, s3_kick3_footoff_scaledidx, s3_kick3_footon_idx, s3_kick3_footon_Fzval, s3_kick3_footon_scaledidx, s3_kick3_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_s3)

# Exclude s3_kick4 from the good/stable kick analysis
[s3_kick4_footoff_idx, s3_kick4_footoff_Fzval, s3_kick4_footoff_scaledidx, s3_kick4_footon_idx, s3_kick4_footon_Fzval, s3_kick4_footon_scaledidx, s3_kick4_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_s3)

[s3_kick5_footoff_idx, s3_kick5_footoff_Fzval, s3_kick5_footoff_scaledidx, s3_kick5_footon_idx, s3_kick5_footon_Fzval, s3_kick5_footon_scaledidx, s3_kick5_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_s3)

[s3_kick6_footoff_idx, s3_kick6_footoff_Fzval, s3_kick6_footoff_scaledidx, s3_kick6_footon_idx, s3_kick6_footon_Fzval, s3_kick6_footon_scaledidx, s3_kick6_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_s3)

[s3_kick7_footoff_idx, s3_kick7_footoff_Fzval, s3_kick7_footoff_scaledidx, s3_kick7_footon_idx, s3_kick7_footon_Fzval, s3_kick7_footon_scaledidx, s3_kick7_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_s3)

[s3_kick8_footoff_idx, s3_kick8_footoff_Fzval, s3_kick8_footoff_scaledidx, s3_kick8_footon_idx, s3_kick8_footon_Fzval, s3_kick8_footon_scaledidx, s3_kick8_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_s3)

[s3_kick9_footoff_idx, s3_kick9_footoff_Fzval, s3_kick9_footoff_scaledidx, s3_kick9_footon_idx, s3_kick9_footon_Fzval, s3_kick9_footon_scaledidx, s3_kick9_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_s3)

[s3_kick10_footoff_idx, s3_kick10_footoff_Fzval, s3_kick10_footoff_scaledidx, s3_kick10_footon_idx, s3_kick10_footon_Fzval, s3_kick10_footon_scaledidx, s3_kick10_window_fp] = find_foot_on_off_FP_idx(
	"s3", kick_frames.roundhousekicks_start_end_frames_s3_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_s3_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_s3)

# Session 4
[s4_kick1_footoff_idx, s4_kick1_footoff_Fzval, s4_kick1_footoff_scaledidx, s4_kick1_footon_idx, s4_kick1_footon_Fzval, s4_kick1_footon_scaledidx, s4_kick1_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick2_footoff_idx, s4_kick2_footoff_Fzval, s4_kick2_footoff_scaledidx, s4_kick2_footon_idx, s4_kick2_footon_Fzval, s4_kick2_footon_scaledidx, s4_kick2_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick3_footoff_idx, s4_kick3_footoff_Fzval, s4_kick3_footoff_scaledidx, s4_kick3_footon_idx, s4_kick3_footon_Fzval, s4_kick3_footon_scaledidx, s4_kick3_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick4_footoff_idx, s4_kick4_footoff_Fzval, s4_kick4_footoff_scaledidx, s4_kick4_footon_idx, s4_kick4_footon_Fzval, s4_kick4_footon_scaledidx, s4_kick4_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick5_footoff_idx, s4_kick5_footoff_Fzval, s4_kick5_footoff_scaledidx, s4_kick5_footon_idx, s4_kick5_footon_Fzval, s4_kick5_footon_scaledidx, s4_kick5_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick6_footoff_idx, s4_kick6_footoff_Fzval, s4_kick6_footoff_scaledidx, s4_kick6_footon_idx, s4_kick6_footon_Fzval, s4_kick6_footon_scaledidx, s4_kick6_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick7_footoff_idx, s4_kick7_footoff_Fzval, s4_kick7_footoff_scaledidx, s4_kick7_footon_idx, s4_kick7_footon_Fzval, s4_kick7_footon_scaledidx, s4_kick7_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick8_footoff_idx, s4_kick8_footoff_Fzval, s4_kick8_footoff_scaledidx, s4_kick8_footon_idx, s4_kick8_footon_Fzval, s4_kick8_footon_scaledidx, s4_kick8_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick9_footoff_idx, s4_kick9_footoff_Fzval, s4_kick9_footoff_scaledidx, s4_kick9_footon_idx, s4_kick9_footon_Fzval, s4_kick9_footon_scaledidx, s4_kick9_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_s4)

[s4_kick10_footoff_idx, s4_kick10_footoff_Fzval, s4_kick10_footoff_scaledidx, s4_kick10_footon_idx, s4_kick10_footon_Fzval, s4_kick10_footon_scaledidx, s4_kick10_window_fp] = find_foot_on_off_FP_idx(
	"s4", kick_frames.roundhousekicks_start_end_frames_s4_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_s4_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_s4)

# Session 5
[s5_kick1_footoff_idx, s5_kick1_footoff_Fzval, s5_kick1_footoff_scaledidx, s5_kick1_footon_idx, s5_kick1_footon_Fzval, s5_kick1_footon_scaledidx, s5_kick1_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick2_footoff_idx, s5_kick2_footoff_Fzval, s5_kick2_footoff_scaledidx, s5_kick2_footon_idx, s5_kick2_footon_Fzval, s5_kick2_footon_scaledidx, s5_kick2_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick3_footoff_idx, s5_kick3_footoff_Fzval, s5_kick3_footoff_scaledidx, s5_kick3_footon_idx, s5_kick3_footon_Fzval, s5_kick3_footon_scaledidx, s5_kick3_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick4_footoff_idx, s5_kick4_footoff_Fzval, s5_kick4_footoff_scaledidx, s5_kick4_footon_idx, s5_kick4_footon_Fzval, s5_kick4_footon_scaledidx, s5_kick4_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick5_footoff_idx, s5_kick5_footoff_Fzval, s5_kick5_footoff_scaledidx, s5_kick5_footon_idx, s5_kick5_footon_Fzval, s5_kick5_footon_scaledidx, s5_kick5_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick6_footoff_idx, s5_kick6_footoff_Fzval, s5_kick6_footoff_scaledidx, s5_kick6_footon_idx, s5_kick6_footon_Fzval, s5_kick6_footon_scaledidx, s5_kick6_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick7_footoff_idx, s5_kick7_footoff_Fzval, s5_kick7_footoff_scaledidx, s5_kick7_footon_idx, s5_kick7_footon_Fzval, s5_kick7_footon_scaledidx, s5_kick7_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick8_footoff_idx, s5_kick8_footoff_Fzval, s5_kick8_footoff_scaledidx, s5_kick8_footon_idx, s5_kick8_footon_Fzval, s5_kick8_footon_scaledidx, s5_kick8_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick9_footoff_idx, s5_kick9_footoff_Fzval, s5_kick9_footoff_scaledidx, s5_kick9_footon_idx, s5_kick9_footon_Fzval, s5_kick9_footon_scaledidx, s5_kick9_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_s5)

[s5_kick10_footoff_idx, s5_kick10_footoff_Fzval, s5_kick10_footoff_scaledidx, s5_kick10_footon_idx, s5_kick10_footon_Fzval, s5_kick10_footon_scaledidx, s5_kick10_window_fp] = find_foot_on_off_FP_idx(
	"s5", kick_frames.roundhousekicks_start_end_frames_s5_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_s5_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_s5)

# Session 6
[s6_kick1_footoff_idx, s6_kick1_footoff_Fzval, s6_kick1_footoff_scaledidx, s6_kick1_footon_idx, s6_kick1_footon_Fzval, s6_kick1_footon_scaledidx, s6_kick1_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick2_footoff_idx, s6_kick2_footoff_Fzval, s6_kick2_footoff_scaledidx, s6_kick2_footon_idx, s6_kick2_footon_Fzval, s6_kick2_footon_scaledidx, s6_kick2_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick3_footoff_idx, s6_kick3_footoff_Fzval, s6_kick3_footoff_scaledidx, s6_kick3_footon_idx, s6_kick3_footon_Fzval, s6_kick3_footon_scaledidx, s6_kick3_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick4_footoff_idx, s6_kick4_footoff_Fzval, s6_kick4_footoff_scaledidx, s6_kick4_footon_idx, s6_kick4_footon_Fzval, s6_kick4_footon_scaledidx, s6_kick4_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick5_footoff_idx, s6_kick5_footoff_Fzval, s6_kick5_footoff_scaledidx, s6_kick5_footon_idx, s6_kick5_footon_Fzval, s6_kick5_footon_scaledidx, s6_kick5_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick6_footoff_idx, s6_kick6_footoff_Fzval, s6_kick6_footoff_scaledidx, s6_kick6_footon_idx, s6_kick6_footon_Fzval, s6_kick6_footon_scaledidx, s6_kick6_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick7_footoff_idx, s6_kick7_footoff_Fzval, s6_kick7_footoff_scaledidx, s6_kick7_footon_idx, s6_kick7_footon_Fzval, s6_kick7_footon_scaledidx, s6_kick7_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick8_footoff_idx, s6_kick8_footoff_Fzval, s6_kick8_footoff_scaledidx, s6_kick8_footon_idx, s6_kick8_footon_Fzval, s6_kick8_footon_scaledidx, s6_kick8_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick9_footoff_idx, s6_kick9_footoff_Fzval, s6_kick9_footoff_scaledidx, s6_kick9_footon_idx, s6_kick9_footon_Fzval, s6_kick9_footon_scaledidx, s6_kick9_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_s6)

[s6_kick10_footoff_idx, s6_kick10_footoff_Fzval, s6_kick10_footoff_scaledidx, s6_kick10_footon_idx, s6_kick10_footon_Fzval, s6_kick10_footon_scaledidx, s6_kick10_window_fp] = find_foot_on_off_FP_idx(
	"s6", kick_frames.roundhousekicks_start_end_frames_s6_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_s6_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_s6)

# Session 7
[s7_kick1_footoff_idx, s7_kick1_footoff_Fzval, s7_kick1_footoff_scaledidx, s7_kick1_footon_idx, s7_kick1_footon_Fzval, s7_kick1_footon_scaledidx, s7_kick1_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick2_footoff_idx, s7_kick2_footoff_Fzval, s7_kick2_footoff_scaledidx, s7_kick2_footon_idx, s7_kick2_footon_Fzval, s7_kick2_footon_scaledidx, s7_kick2_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick3_footoff_idx, s7_kick3_footoff_Fzval, s7_kick3_footoff_scaledidx, s7_kick3_footon_idx, s7_kick3_footon_Fzval, s7_kick3_footon_scaledidx, s7_kick3_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick4_footoff_idx, s7_kick4_footoff_Fzval, s7_kick4_footoff_scaledidx, s7_kick4_footon_idx, s7_kick4_footon_Fzval, s7_kick4_footon_scaledidx, s7_kick4_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick5_footoff_idx, s7_kick5_footoff_Fzval, s7_kick5_footoff_scaledidx, s7_kick5_footon_idx, s7_kick5_footon_Fzval, s7_kick5_footon_scaledidx, s7_kick5_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick6_footoff_idx, s7_kick6_footoff_Fzval, s7_kick6_footoff_scaledidx, s7_kick6_footon_idx, s7_kick6_footon_Fzval, s7_kick6_footon_scaledidx, s7_kick6_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick7_footoff_idx, s7_kick7_footoff_Fzval, s7_kick7_footoff_scaledidx, s7_kick7_footon_idx, s7_kick7_footon_Fzval, s7_kick7_footon_scaledidx, s7_kick7_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick8_footoff_idx, s7_kick8_footoff_Fzval, s7_kick8_footoff_scaledidx, s7_kick8_footon_idx, s7_kick8_footon_Fzval, s7_kick8_footon_scaledidx, s7_kick8_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick9_footoff_idx, s7_kick9_footoff_Fzval, s7_kick9_footoff_scaledidx, s7_kick9_footon_idx, s7_kick9_footon_Fzval, s7_kick9_footon_scaledidx, s7_kick9_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_s7)

[s7_kick10_footoff_idx, s7_kick10_footoff_Fzval, s7_kick10_footoff_scaledidx, s7_kick10_footon_idx, s7_kick10_footon_Fzval, s7_kick10_footon_scaledidx, s7_kick10_window_fp] = find_foot_on_off_FP_idx(
	"s7", kick_frames.roundhousekicks_start_end_frames_s7_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_s7_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_s7)

# Session 8
[s8_kick1_footoff_idx, s8_kick1_footoff_Fzval, s8_kick1_footoff_scaledidx, s8_kick1_footon_idx, s8_kick1_footon_Fzval, s8_kick1_footon_scaledidx, s8_kick1_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick2_footoff_idx, s8_kick2_footoff_Fzval, s8_kick2_footoff_scaledidx, s8_kick2_footon_idx, s8_kick2_footon_Fzval, s8_kick2_footon_scaledidx, s8_kick2_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick3_footoff_idx, s8_kick3_footoff_Fzval, s8_kick3_footoff_scaledidx, s8_kick3_footon_idx, s8_kick3_footon_Fzval, s8_kick3_footon_scaledidx, s8_kick3_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick4_footoff_idx, s8_kick4_footoff_Fzval, s8_kick4_footoff_scaledidx, s8_kick4_footon_idx, s8_kick4_footon_Fzval, s8_kick4_footon_scaledidx, s8_kick4_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick5_footoff_idx, s8_kick5_footoff_Fzval, s8_kick5_footoff_scaledidx, s8_kick5_footon_idx, s8_kick5_footon_Fzval, s8_kick5_footon_scaledidx, s8_kick5_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick6_footoff_idx, s8_kick6_footoff_Fzval, s8_kick6_footoff_scaledidx, s8_kick6_footon_idx, s8_kick6_footon_Fzval, s8_kick6_footon_scaledidx, s8_kick6_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick7_footoff_idx, s8_kick7_footoff_Fzval, s8_kick7_footoff_scaledidx, s8_kick7_footon_idx, s8_kick7_footon_Fzval, s8_kick7_footon_scaledidx, s8_kick7_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick8_footoff_idx, s8_kick8_footoff_Fzval, s8_kick8_footoff_scaledidx, s8_kick8_footon_idx, s8_kick8_footon_Fzval, s8_kick8_footon_scaledidx, s8_kick8_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick9_footoff_idx, s8_kick9_footoff_Fzval, s8_kick9_footoff_scaledidx, s8_kick9_footon_idx, s8_kick9_footon_Fzval, s8_kick9_footon_scaledidx, s8_kick9_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_s8)

[s8_kick10_footoff_idx, s8_kick10_footoff_Fzval, s8_kick10_footoff_scaledidx, s8_kick10_footon_idx, s8_kick10_footon_Fzval, s8_kick10_footon_scaledidx, s8_kick10_window_fp] = find_foot_on_off_FP_idx(
	"s8", kick_frames.roundhousekicks_start_end_frames_s8_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_s8_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_s8)

# Expert
[expert_kick1_footoff_idx, expert_kick1_footoff_Fzval, expert_kick1_footoff_scaledidx, expert_kick1_footon_idx, expert_kick1_footon_Fzval, expert_kick1_footon_scaledidx, expert_kick1_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[0,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[0,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick2_footoff_idx, expert_kick2_footoff_Fzval, expert_kick2_footoff_scaledidx, expert_kick2_footon_idx, expert_kick2_footon_Fzval, expert_kick2_footon_scaledidx, expert_kick2_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[1,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[1,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick3_footoff_idx, expert_kick3_footoff_Fzval, expert_kick3_footoff_scaledidx, expert_kick3_footon_idx, expert_kick3_footon_Fzval, expert_kick3_footon_scaledidx, expert_kick3_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[2,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[2,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick4_footoff_idx, expert_kick4_footoff_Fzval, expert_kick4_footoff_scaledidx, expert_kick4_footon_idx, expert_kick4_footon_Fzval, expert_kick4_footon_scaledidx, expert_kick4_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[3,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[3,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick5_footoff_idx, expert_kick5_footoff_Fzval, expert_kick5_footoff_scaledidx, expert_kick5_footon_idx, expert_kick5_footon_Fzval, expert_kick5_footon_scaledidx, expert_kick5_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[4,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[4,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick6_footoff_idx, expert_kick6_footoff_Fzval, expert_kick6_footoff_scaledidx, expert_kick6_footon_idx, expert_kick6_footon_Fzval, expert_kick6_footon_scaledidx, expert_kick6_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[5,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[5,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick7_footoff_idx, expert_kick7_footoff_Fzval, expert_kick7_footoff_scaledidx, expert_kick7_footon_idx, expert_kick7_footon_Fzval, expert_kick7_footon_scaledidx, expert_kick7_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[6,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[6,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick8_footoff_idx, expert_kick8_footoff_Fzval, expert_kick8_footoff_scaledidx, expert_kick8_footon_idx, expert_kick8_footon_Fzval, expert_kick8_footon_scaledidx, expert_kick8_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[7,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[7,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick9_footoff_idx, expert_kick9_footoff_Fzval, expert_kick9_footoff_scaledidx, expert_kick9_footon_idx, expert_kick9_footon_Fzval, expert_kick9_footon_scaledidx, expert_kick9_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[8,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[8,1], fp_data.roundhousekicks_filtered_resampled_expert)

[expert_kick10_footoff_idx, expert_kick10_footoff_Fzval, expert_kick10_footoff_scaledidx, expert_kick10_footon_idx, expert_kick10_footon_Fzval, expert_kick10_footon_scaledidx, expert_kick10_window_fp] = find_foot_on_off_FP_idx(
	"expert", kick_frames.roundhousekicks_start_end_frames_expert_mocap[9,0], kick_frames.roundhousekicks_start_end_frames_expert_mocap[9,1], fp_data.roundhousekicks_filtered_resampled_expert)

# ''' Plot for VERIFICATION PURPOSES: Are the footoff and footon instances correct?'''
# # Session 1
# Fz_s1_figs, ((Fz_s1_kick1, Fz_s1_kick2, Fz_s1_kick3, Fz_s1_kick4, Fz_s1_kick5),
# 			 (Fz_s1_kick6, Fz_s1_kick7, Fz_s1_kick8, Fz_s1_kick9, Fz_s1_kick10)) = plt.subplots(2,5)
# Fz_s1_figs.set_figheight(10)
# Fz_s1_figs.set_figwidth(25)

# Fz_s1_kick1.plot(kick_frames.scaled_time_kick1_s1_mocap, s1_kick1_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick2.plot(kick_frames.scaled_time_kick2_s1_mocap, s1_kick2_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick3.plot(kick_frames.scaled_time_kick3_s1_mocap, s1_kick3_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick4.plot(kick_frames.scaled_time_kick4_s1_mocap, s1_kick4_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick5.plot(kick_frames.scaled_time_kick5_s1_mocap, s1_kick5_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick6.plot(kick_frames.scaled_time_kick6_s1_mocap, s1_kick6_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick7.plot(kick_frames.scaled_time_kick7_s1_mocap, s1_kick7_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick8.plot(kick_frames.scaled_time_kick8_s1_mocap, s1_kick8_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick9.plot(kick_frames.scaled_time_kick9_s1_mocap, s1_kick9_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick10.plot(kick_frames.scaled_time_kick10_s1_mocap, s1_kick10_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s1_kick1.plot(s1_kick1_footoff_scaledidx, s1_kick1_footoff_Fzval, 'o')
# Fz_s1_kick2.plot(s1_kick2_footoff_scaledidx, s1_kick2_footoff_Fzval, 'o')
# Fz_s1_kick3.plot(s1_kick3_footoff_scaledidx, s1_kick3_footoff_Fzval, 'o')
# Fz_s1_kick4.plot(s1_kick4_footoff_scaledidx, s1_kick4_footoff_Fzval, 'o')
# Fz_s1_kick5.plot(s1_kick5_footoff_scaledidx1, s1_kick5_footoff_Fzval1, 'o')
# Fz_s1_kick5.plot(s1_kick5_footoff_scaledidx2, s1_kick5_footoff_Fzval2, 'o')
# Fz_s1_kick6.plot(s1_kick6_footoff_scaledidx, s1_kick6_footoff_Fzval, 'o')
# Fz_s1_kick7.plot(s1_kick7_footoff_scaledidx, s1_kick7_footoff_Fzval, 'o')
# Fz_s1_kick8.plot(s1_kick8_footoff_scaledidx, s1_kick8_footoff_Fzval, 'o')
# Fz_s1_kick9.plot(s1_kick9_footoff_scaledidx, s1_kick9_footoff_Fzval, 'o')
# Fz_s1_kick10.plot(s1_kick10_footoff_scaledidx, s1_kick10_footoff_Fzval, 'o')
# Fz_s1_kick1.plot(s1_kick1_footon_scaledidx, s1_kick1_footon_Fzval, 'x')
# Fz_s1_kick2.plot(s1_kick2_footon_scaledidx, s1_kick2_footon_Fzval, 'x')
# Fz_s1_kick3.plot(s1_kick3_footon_scaledidx, s1_kick3_footon_Fzval, 'x')
# Fz_s1_kick4.plot(s1_kick4_footon_scaledidx, s1_kick4_footon_Fzval, 'x')
# Fz_s1_kick5.plot(s1_kick5_footon_scaledidx1, s1_kick5_footon_Fzval1, 'x')
# Fz_s1_kick5.plot(s1_kick5_footon_scaledidx2, s1_kick5_footon_Fzval2, 'x')
# Fz_s1_kick6.plot(s1_kick6_footon_scaledidx, s1_kick6_footon_Fzval, 'x')
# Fz_s1_kick7.plot(s1_kick7_footon_scaledidx, s1_kick7_footon_Fzval, 'x')
# Fz_s1_kick8.plot(s1_kick8_footon_scaledidx, s1_kick8_footon_Fzval, 'x')
# Fz_s1_kick9.plot(s1_kick9_footon_scaledidx, s1_kick9_footon_Fzval, 'x')
# Fz_s1_kick10.plot(s1_kick10_footon_scaledidx, s1_kick10_footon_Fzval, 'x')

# Fz_s1_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_s1_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_s1_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_s1_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_s1_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_s1_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_s1_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_s1_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_s1_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_s1_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_s1_figs.suptitle('Session 1 Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_s1_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_s1_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_s1_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/s1_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as s1_kickfoot_on_off_FP.jpg")

# # Session 2
# Fz_s2_figs, ((Fz_s2_kick1, Fz_s2_kick2, Fz_s2_kick3, Fz_s2_kick4, Fz_s2_kick5),
# 			 (Fz_s2_kick6, Fz_s2_kick7, Fz_s2_kick8, Fz_s2_kick9, Fz_s2_kick10)) = plt.subplots(2,5)
# Fz_s2_figs.set_figheight(10)
# Fz_s2_figs.set_figwidth(25)

# Fz_s2_kick1.plot(kick_frames.scaled_time_kick1_s2_mocap, s2_kick1_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick2.plot(kick_frames.scaled_time_kick2_s2_mocap, s2_kick2_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick3.plot(kick_frames.scaled_time_kick3_s2_mocap, s2_kick3_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick4.plot(kick_frames.scaled_time_kick4_s2_mocap, s2_kick4_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick5.plot(kick_frames.scaled_time_kick5_s2_mocap, s2_kick5_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick6.plot(kick_frames.scaled_time_kick6_s2_mocap, s2_kick6_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick7.plot(kick_frames.scaled_time_kick7_s2_mocap, s2_kick7_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick8.plot(kick_frames.scaled_time_kick8_s2_mocap, s2_kick8_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick9.plot(kick_frames.scaled_time_kick9_s2_mocap, s2_kick9_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick10.plot(kick_frames.scaled_time_kick10_s2_mocap, s2_kick10_window_fp.loc[:, "FP2_Fz"], color = fp_data.s12k5_color)
# Fz_s2_kick1.plot(s2_kick1_footoff_scaledidx, s2_kick1_footoff_Fzval, 'o')
# Fz_s2_kick2.plot(s2_kick2_footoff_scaledidx, s2_kick2_footoff_Fzval, 'o')
# Fz_s2_kick3.plot(s2_kick3_footoff_scaledidx, s2_kick3_footoff_Fzval, 'o')
# Fz_s2_kick4.plot(s2_kick4_footoff_scaledidx, s2_kick4_footoff_Fzval, 'o')
# Fz_s2_kick5.plot(s2_kick5_footoff_scaledidx, s2_kick5_footoff_Fzval, 'o')
# Fz_s2_kick6.plot(s2_kick6_footoff_scaledidx, s2_kick6_footoff_Fzval, 'o')
# Fz_s2_kick7.plot(s2_kick7_footoff_scaledidx, s2_kick7_footoff_Fzval, 'o')
# Fz_s2_kick8.plot(s2_kick8_footoff_scaledidx, s2_kick8_footoff_Fzval, 'o')
# Fz_s2_kick9.plot(s2_kick9_footoff_scaledidx, s2_kick9_footoff_Fzval, 'o')
# Fz_s2_kick10.plot(s2_kick10_footoff_scaledidx, s2_kick10_footoff_Fzval, 'o')
# Fz_s2_kick1.plot(s2_kick1_footon_scaledidx, s2_kick1_footon_Fzval, 'x')
# Fz_s2_kick2.plot(s2_kick2_footon_scaledidx, s2_kick2_footon_Fzval, 'x')
# Fz_s2_kick3.plot(s2_kick3_footon_scaledidx, s2_kick3_footon_Fzval, 'x')
# Fz_s2_kick4.plot(s2_kick4_footon_scaledidx, s2_kick4_footon_Fzval, 'x')
# Fz_s2_kick5.plot(s2_kick5_footon_scaledidx, s2_kick5_footon_Fzval, 'x')
# Fz_s2_kick6.plot(s2_kick6_footon_scaledidx, s2_kick6_footon_Fzval, 'x')
# Fz_s2_kick7.plot(s2_kick7_footon_scaledidx, s2_kick7_footon_Fzval, 'x')
# Fz_s2_kick8.plot(s2_kick8_footon_scaledidx, s2_kick8_footon_Fzval, 'x')
# Fz_s2_kick9.plot(s2_kick9_footon_scaledidx, s2_kick9_footon_Fzval, 'x')
# Fz_s2_kick10.plot(s2_kick10_footon_scaledidx, s2_kick10_footon_Fzval, 'x')

# Fz_s2_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_s2_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_s2_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_s2_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_s2_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_s2_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_s2_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_s2_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_s2_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_s2_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_s2_figs.suptitle('Session 2 Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_s2_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_s2_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_s2_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/s2_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as s2_kickfoot_on_off_FP.jpg")

# # Session 3
# Fz_s3_figs, ((Fz_s3_kick1, Fz_s3_kick2, Fz_s3_kick3, Fz_s3_kick4, Fz_s3_kick5),
# 			 (Fz_s3_kick6, Fz_s3_kick7, Fz_s3_kick8, Fz_s3_kick9, Fz_s3_kick10)) = plt.subplots(2,5)
# Fz_s3_figs.set_figheight(10)
# Fz_s3_figs.set_figwidth(25)

# Fz_s3_kick1.plot(kick_frames.scaled_time_kick1_s3_mocap, s3_kick1_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick2.plot(kick_frames.scaled_time_kick2_s3_mocap, s3_kick2_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick3.plot(kick_frames.scaled_time_kick3_s3_mocap, s3_kick3_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick4.plot(kick_frames.scaled_time_kick4_s3_mocap, s3_kick4_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick5.plot(kick_frames.scaled_time_kick5_s3_mocap, s3_kick5_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick6.plot(kick_frames.scaled_time_kick6_s3_mocap, s3_kick6_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick7.plot(kick_frames.scaled_time_kick7_s3_mocap, s3_kick7_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick8.plot(kick_frames.scaled_time_kick8_s3_mocap, s3_kick8_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick9.plot(kick_frames.scaled_time_kick9_s3_mocap, s3_kick9_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick10.plot(kick_frames.scaled_time_kick10_s3_mocap, s3_kick10_window_fp.loc[:, "FP2_Fz"], color = fp_data.s34k5_color)
# Fz_s3_kick1.plot(s3_kick1_footoff_scaledidx, s3_kick1_footoff_Fzval, 'o')
# Fz_s3_kick2.plot(s3_kick2_footoff_scaledidx, s3_kick2_footoff_Fzval, 'o')
# Fz_s3_kick3.plot(s3_kick3_footoff_scaledidx, s3_kick3_footoff_Fzval, 'o')
# Fz_s3_kick4.plot(s3_kick4_footoff_scaledidx, s3_kick4_footoff_Fzval, 'o')
# Fz_s3_kick5.plot(s3_kick5_footoff_scaledidx, s3_kick5_footoff_Fzval, 'o')
# Fz_s3_kick6.plot(s3_kick6_footoff_scaledidx, s3_kick6_footoff_Fzval, 'o')
# Fz_s3_kick7.plot(s3_kick7_footoff_scaledidx, s3_kick7_footoff_Fzval, 'o')
# Fz_s3_kick8.plot(s3_kick8_footoff_scaledidx, s3_kick8_footoff_Fzval, 'o')
# Fz_s3_kick9.plot(s3_kick9_footoff_scaledidx, s3_kick9_footoff_Fzval, 'o')
# Fz_s3_kick10.plot(s3_kick10_footoff_scaledidx, s3_kick10_footoff_Fzval, 'o')
# Fz_s3_kick1.plot(s3_kick1_footon_scaledidx, s3_kick1_footon_Fzval, 'x')
# Fz_s3_kick2.plot(s3_kick2_footon_scaledidx, s3_kick2_footon_Fzval, 'x')
# Fz_s3_kick3.plot(s3_kick3_footon_scaledidx, s3_kick3_footon_Fzval, 'x')
# Fz_s3_kick4.plot(s3_kick4_footon_scaledidx, s3_kick4_footon_Fzval, 'x')
# Fz_s3_kick5.plot(s3_kick5_footon_scaledidx, s3_kick5_footon_Fzval, 'x')
# Fz_s3_kick6.plot(s3_kick6_footon_scaledidx, s3_kick6_footon_Fzval, 'x')
# Fz_s3_kick7.plot(s3_kick7_footon_scaledidx, s3_kick7_footon_Fzval, 'x')
# Fz_s3_kick8.plot(s3_kick8_footon_scaledidx, s3_kick8_footon_Fzval, 'x')
# Fz_s3_kick9.plot(s3_kick9_footon_scaledidx, s3_kick9_footon_Fzval, 'x')
# Fz_s3_kick10.plot(s3_kick10_footon_scaledidx, s3_kick10_footon_Fzval, 'x')

# Fz_s3_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_s3_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_s3_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_s3_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_s3_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_s3_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_s3_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_s3_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_s3_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_s3_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_s3_figs.suptitle('Session 3 Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_s3_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_s3_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_s3_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/s3_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as s3_kickfoot_on_off_FP.jpg")

# # Session 4
# Fz_s4_figs, ((Fz_s4_kick1, Fz_s4_kick2, Fz_s4_kick3, Fz_s4_kick4, Fz_s4_kick5),
# 			 (Fz_s4_kick6, Fz_s4_kick7, Fz_s4_kick8, Fz_s4_kick9, Fz_s4_kick10)) = plt.subplots(2,5)
# Fz_s4_figs.set_figheight(10)
# Fz_s4_figs.set_figwidth(25)

# Fz_s4_kick1.plot(kick_frames.scaled_time_kick1_s4_mocap, s4_kick1_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick2.plot(kick_frames.scaled_time_kick2_s4_mocap, s4_kick2_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick3.plot(kick_frames.scaled_time_kick3_s4_mocap, s4_kick3_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick4.plot(kick_frames.scaled_time_kick4_s4_mocap, s4_kick4_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick5.plot(kick_frames.scaled_time_kick5_s4_mocap, s4_kick5_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick6.plot(kick_frames.scaled_time_kick6_s4_mocap, s4_kick6_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick7.plot(kick_frames.scaled_time_kick7_s4_mocap, s4_kick7_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick8.plot(kick_frames.scaled_time_kick8_s4_mocap, s4_kick8_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick9.plot(kick_frames.scaled_time_kick9_s4_mocap, s4_kick9_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick10.plot(kick_frames.scaled_time_kick10_s4_mocap, s4_kick10_window_fp.loc[:, "FP5_Fz"], color = fp_data.s34k5_color)
# Fz_s4_kick1.plot(s4_kick1_footoff_scaledidx, s4_kick1_footoff_Fzval, 'o')
# Fz_s4_kick2.plot(s4_kick2_footoff_scaledidx, s4_kick2_footoff_Fzval, 'o')
# Fz_s4_kick3.plot(s4_kick3_footoff_scaledidx, s4_kick3_footoff_Fzval, 'o')
# Fz_s4_kick4.plot(s4_kick4_footoff_scaledidx, s4_kick4_footoff_Fzval, 'o')
# Fz_s4_kick5.plot(s4_kick5_footoff_scaledidx, s4_kick5_footoff_Fzval, 'o')
# Fz_s4_kick6.plot(s4_kick6_footoff_scaledidx, s4_kick6_footoff_Fzval, 'o')
# Fz_s4_kick7.plot(s4_kick7_footoff_scaledidx, s4_kick7_footoff_Fzval, 'o')
# Fz_s4_kick8.plot(s4_kick8_footoff_scaledidx, s4_kick8_footoff_Fzval, 'o')
# Fz_s4_kick9.plot(s4_kick9_footoff_scaledidx, s4_kick9_footoff_Fzval, 'o')
# Fz_s4_kick10.plot(s4_kick10_footoff_scaledidx, s4_kick10_footoff_Fzval, 'o')
# Fz_s4_kick1.plot(s4_kick1_footon_scaledidx, s4_kick1_footon_Fzval, 'x')
# Fz_s4_kick2.plot(s4_kick2_footon_scaledidx, s4_kick2_footon_Fzval, 'x')
# Fz_s4_kick3.plot(s4_kick3_footon_scaledidx, s4_kick3_footon_Fzval, 'x')
# Fz_s4_kick4.plot(s4_kick4_footon_scaledidx, s4_kick4_footon_Fzval, 'x')
# Fz_s4_kick5.plot(s4_kick5_footon_scaledidx, s4_kick5_footon_Fzval, 'x')
# Fz_s4_kick6.plot(s4_kick6_footon_scaledidx, s4_kick6_footon_Fzval, 'x')
# Fz_s4_kick7.plot(s4_kick7_footon_scaledidx, s4_kick7_footon_Fzval, 'x')
# Fz_s4_kick8.plot(s4_kick8_footon_scaledidx, s4_kick8_footon_Fzval, 'x')
# Fz_s4_kick9.plot(s4_kick9_footon_scaledidx, s4_kick9_footon_Fzval, 'x')
# Fz_s4_kick10.plot(s4_kick10_footon_scaledidx, s4_kick10_footon_Fzval, 'x')

# Fz_s4_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_s4_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_s4_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_s4_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_s4_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_s4_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_s4_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_s4_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_s4_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_s4_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_s4_figs.suptitle('Session 4 Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_s4_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_s4_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_s4_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/s4_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as s4_kickfoot_on_off_FP.jpg")

# # Session 5
# Fz_s5_figs, ((Fz_s5_kick1, Fz_s5_kick2, Fz_s5_kick3, Fz_s5_kick4, Fz_s5_kick5),
# 			 (Fz_s5_kick6, Fz_s5_kick7, Fz_s5_kick8, Fz_s5_kick9, Fz_s5_kick10)) = plt.subplots(2,5)
# Fz_s5_figs.set_figheight(10)
# Fz_s5_figs.set_figwidth(25)

# Fz_s5_kick1.plot(kick_frames.scaled_time_kick1_s5_mocap, s5_kick1_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick2.plot(kick_frames.scaled_time_kick2_s5_mocap, s5_kick2_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick3.plot(kick_frames.scaled_time_kick3_s5_mocap, s5_kick3_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick4.plot(kick_frames.scaled_time_kick4_s5_mocap, s5_kick4_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick5.plot(kick_frames.scaled_time_kick5_s5_mocap, s5_kick5_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick6.plot(kick_frames.scaled_time_kick6_s5_mocap, s5_kick6_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick7.plot(kick_frames.scaled_time_kick7_s5_mocap, s5_kick7_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick8.plot(kick_frames.scaled_time_kick8_s5_mocap, s5_kick8_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick9.plot(kick_frames.scaled_time_kick9_s5_mocap, s5_kick9_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick10.plot(kick_frames.scaled_time_kick10_s5_mocap, s5_kick10_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s5_kick1.plot(s5_kick1_footoff_scaledidx, s5_kick1_footoff_Fzval, 'o')
# Fz_s5_kick2.plot(s5_kick2_footoff_scaledidx, s5_kick2_footoff_Fzval, 'o')
# Fz_s5_kick3.plot(s5_kick3_footoff_scaledidx, s5_kick3_footoff_Fzval, 'o')
# Fz_s5_kick4.plot(s5_kick4_footoff_scaledidx, s5_kick4_footoff_Fzval, 'o')
# Fz_s5_kick5.plot(s5_kick5_footoff_scaledidx, s5_kick5_footoff_Fzval, 'o')
# Fz_s5_kick6.plot(s5_kick6_footoff_scaledidx, s5_kick6_footoff_Fzval, 'o')
# Fz_s5_kick7.plot(s5_kick7_footoff_scaledidx, s5_kick7_footoff_Fzval, 'o')
# Fz_s5_kick8.plot(s5_kick8_footoff_scaledidx, s5_kick8_footoff_Fzval, 'o')
# Fz_s5_kick9.plot(s5_kick9_footoff_scaledidx, s5_kick9_footoff_Fzval, 'o')
# Fz_s5_kick10.plot(s5_kick10_footoff_scaledidx, s5_kick10_footoff_Fzval, 'o')
# Fz_s5_kick1.plot(s5_kick1_footon_scaledidx, s5_kick1_footon_Fzval, 'x')
# Fz_s5_kick2.plot(s5_kick2_footon_scaledidx, s5_kick2_footon_Fzval, 'x')
# Fz_s5_kick3.plot(s5_kick3_footon_scaledidx, s5_kick3_footon_Fzval, 'x')
# Fz_s5_kick4.plot(s5_kick4_footon_scaledidx, s5_kick4_footon_Fzval, 'x')
# Fz_s5_kick5.plot(s5_kick5_footon_scaledidx, s5_kick5_footon_Fzval, 'x')
# Fz_s5_kick6.plot(s5_kick6_footon_scaledidx, s5_kick6_footon_Fzval, 'x')
# Fz_s5_kick7.plot(s5_kick7_footon_scaledidx, s5_kick7_footon_Fzval, 'x')
# Fz_s5_kick8.plot(s5_kick8_footon_scaledidx, s5_kick8_footon_Fzval, 'x')
# Fz_s5_kick9.plot(s5_kick9_footon_scaledidx, s5_kick9_footon_Fzval, 'x')
# Fz_s5_kick10.plot(s5_kick10_footon_scaledidx, s5_kick10_footon_Fzval, 'x')

# Fz_s5_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_s5_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_s5_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_s5_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_s5_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_s5_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_s5_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_s5_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_s5_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_s5_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_s5_figs.suptitle('Session 5 Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_s5_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_s5_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_s5_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/s5_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as s5_kickfoot_on_off_FP.jpg")

# # Session 6
# Fz_s6_figs, ((Fz_s6_kick1, Fz_s6_kick2, Fz_s6_kick3, Fz_s6_kick4, Fz_s6_kick5),
# 			 (Fz_s6_kick6, Fz_s6_kick7, Fz_s6_kick8, Fz_s6_kick9, Fz_s6_kick10)) = plt.subplots(2,5)
# Fz_s6_figs.set_figheight(10)
# Fz_s6_figs.set_figwidth(25)

# Fz_s6_kick1.plot(kick_frames.scaled_time_kick1_s6_mocap, s6_kick1_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick2.plot(kick_frames.scaled_time_kick2_s6_mocap, s6_kick2_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick3.plot(kick_frames.scaled_time_kick3_s6_mocap, s6_kick3_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick4.plot(kick_frames.scaled_time_kick4_s6_mocap, s6_kick4_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick5.plot(kick_frames.scaled_time_kick5_s6_mocap, s6_kick5_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick6.plot(kick_frames.scaled_time_kick6_s6_mocap, s6_kick6_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick7.plot(kick_frames.scaled_time_kick7_s6_mocap, s6_kick7_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick8.plot(kick_frames.scaled_time_kick8_s6_mocap, s6_kick8_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick9.plot(kick_frames.scaled_time_kick9_s6_mocap, s6_kick9_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick10.plot(kick_frames.scaled_time_kick10_s6_mocap, s6_kick10_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s6_kick1.plot(s6_kick1_footoff_scaledidx, s6_kick1_footoff_Fzval, 'o')
# Fz_s6_kick2.plot(s6_kick2_footoff_scaledidx, s6_kick2_footoff_Fzval, 'o')
# Fz_s6_kick3.plot(s6_kick3_footoff_scaledidx, s6_kick3_footoff_Fzval, 'o')
# Fz_s6_kick4.plot(s6_kick4_footoff_scaledidx, s6_kick4_footoff_Fzval, 'o')
# Fz_s6_kick5.plot(s6_kick5_footoff_scaledidx, s6_kick5_footoff_Fzval, 'o')
# Fz_s6_kick6.plot(s6_kick6_footoff_scaledidx, s6_kick6_footoff_Fzval, 'o')
# Fz_s6_kick7.plot(s6_kick7_footoff_scaledidx, s6_kick7_footoff_Fzval, 'o')
# Fz_s6_kick8.plot(s6_kick8_footoff_scaledidx, s6_kick8_footoff_Fzval, 'o')
# Fz_s6_kick9.plot(s6_kick9_footoff_scaledidx, s6_kick9_footoff_Fzval, 'o')
# Fz_s6_kick10.plot(s6_kick10_footoff_scaledidx, s6_kick10_footoff_Fzval, 'o')
# Fz_s6_kick1.plot(s6_kick1_footon_scaledidx, s6_kick1_footon_Fzval, 'x')
# Fz_s6_kick2.plot(s6_kick2_footon_scaledidx, s6_kick2_footon_Fzval, 'x')
# Fz_s6_kick3.plot(s6_kick3_footon_scaledidx, s6_kick3_footon_Fzval, 'x')
# Fz_s6_kick4.plot(s6_kick4_footon_scaledidx, s6_kick4_footon_Fzval, 'x')
# Fz_s6_kick5.plot(s6_kick5_footon_scaledidx, s6_kick5_footon_Fzval, 'x')
# Fz_s6_kick6.plot(s6_kick6_footon_scaledidx, s6_kick6_footon_Fzval, 'x')
# Fz_s6_kick7.plot(s6_kick7_footon_scaledidx, s6_kick7_footon_Fzval, 'x')
# Fz_s6_kick8.plot(s6_kick8_footon_scaledidx, s6_kick8_footon_Fzval, 'x')
# Fz_s6_kick9.plot(s6_kick9_footon_scaledidx, s6_kick9_footon_Fzval, 'x')
# Fz_s6_kick10.plot(s6_kick10_footon_scaledidx, s6_kick10_footon_Fzval, 'x')

# Fz_s6_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_s6_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_s6_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_s6_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_s6_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_s6_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_s6_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_s6_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_s6_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_s6_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_s6_figs.suptitle('Session 6 Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_s6_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_s6_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_s6_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/s6_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as s6_kickfoot_on_off_FP.jpg")

# # Session 7
# Fz_s7_figs, ((Fz_s7_kick1, Fz_s7_kick2, Fz_s7_kick3, Fz_s7_kick4, Fz_s7_kick5),
# 			 (Fz_s7_kick6, Fz_s7_kick7, Fz_s7_kick8, Fz_s7_kick9, Fz_s7_kick10)) = plt.subplots(2,5)
# Fz_s7_figs.set_figheight(10)
# Fz_s7_figs.set_figwidth(25)

# Fz_s7_kick1.plot(kick_frames.scaled_time_kick1_s7_mocap, s7_kick1_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick2.plot(kick_frames.scaled_time_kick2_s7_mocap, s7_kick2_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick3.plot(kick_frames.scaled_time_kick3_s7_mocap, s7_kick3_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick4.plot(kick_frames.scaled_time_kick4_s7_mocap, s7_kick4_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick5.plot(kick_frames.scaled_time_kick5_s7_mocap, s7_kick5_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick6.plot(kick_frames.scaled_time_kick6_s7_mocap, s7_kick6_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick7.plot(kick_frames.scaled_time_kick7_s7_mocap, s7_kick7_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick8.plot(kick_frames.scaled_time_kick8_s7_mocap, s7_kick8_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick9.plot(kick_frames.scaled_time_kick9_s7_mocap, s7_kick9_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick10.plot(kick_frames.scaled_time_kick10_s7_mocap, s7_kick10_window_fp.loc[:, "FP5_Fz"], color = fp_data.s567k5_color)
# Fz_s7_kick1.plot(s7_kick1_footoff_scaledidx, s7_kick1_footoff_Fzval, 'o')
# Fz_s7_kick2.plot(s7_kick2_footoff_scaledidx, s7_kick2_footoff_Fzval, 'o')
# Fz_s7_kick3.plot(s7_kick3_footoff_scaledidx, s7_kick3_footoff_Fzval, 'o')
# Fz_s7_kick4.plot(s7_kick4_footoff_scaledidx, s7_kick4_footoff_Fzval, 'o')
# Fz_s7_kick5.plot(s7_kick5_footoff_scaledidx, s7_kick5_footoff_Fzval, 'o')
# Fz_s7_kick6.plot(s7_kick6_footoff_scaledidx, s7_kick6_footoff_Fzval, 'o')
# Fz_s7_kick7.plot(s7_kick7_footoff_scaledidx, s7_kick7_footoff_Fzval, 'o')
# Fz_s7_kick8.plot(s7_kick8_footoff_scaledidx, s7_kick8_footoff_Fzval, 'o')
# Fz_s7_kick9.plot(s7_kick9_footoff_scaledidx, s7_kick9_footoff_Fzval, 'o')
# Fz_s7_kick10.plot(s7_kick10_footoff_scaledidx, s7_kick10_footoff_Fzval, 'o')
# Fz_s7_kick1.plot(s7_kick1_footon_scaledidx, s7_kick1_footon_Fzval, 'x')
# Fz_s7_kick2.plot(s7_kick2_footon_scaledidx, s7_kick2_footon_Fzval, 'x')
# Fz_s7_kick3.plot(s7_kick3_footon_scaledidx, s7_kick3_footon_Fzval, 'x')
# Fz_s7_kick4.plot(s7_kick4_footon_scaledidx, s7_kick4_footon_Fzval, 'x')
# Fz_s7_kick5.plot(s7_kick5_footon_scaledidx, s7_kick5_footon_Fzval, 'x')
# Fz_s7_kick6.plot(s7_kick6_footon_scaledidx, s7_kick6_footon_Fzval, 'x')
# Fz_s7_kick7.plot(s7_kick7_footon_scaledidx, s7_kick7_footon_Fzval, 'x')
# Fz_s7_kick8.plot(s7_kick8_footon_scaledidx, s7_kick8_footon_Fzval, 'x')
# Fz_s7_kick9.plot(s7_kick9_footon_scaledidx, s7_kick9_footon_Fzval, 'x')
# Fz_s7_kick10.plot(s7_kick10_footon_scaledidx, s7_kick10_footon_Fzval, 'x')

# Fz_s7_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_s7_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_s7_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_s7_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_s7_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_s7_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_s7_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_s7_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_s7_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_s7_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_s7_figs.suptitle('Session 7 Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_s7_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_s7_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_s7_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/s7_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as s7_kickfoot_on_off_FP.jpg")

# # Session 8
# Fz_s8_figs, ((Fz_s8_kick1, Fz_s8_kick2, Fz_s8_kick3, Fz_s8_kick4, Fz_s8_kick5),
# 			 (Fz_s8_kick6, Fz_s8_kick7, Fz_s8_kick8, Fz_s8_kick9, Fz_s8_kick10)) = plt.subplots(2,5)
# Fz_s8_figs.set_figheight(10)
# Fz_s8_figs.set_figwidth(25)

# Fz_s8_kick1.plot(kick_frames.scaled_time_kick1_s8_mocap, s8_kick1_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick2.plot(kick_frames.scaled_time_kick2_s8_mocap, s8_kick2_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick3.plot(kick_frames.scaled_time_kick3_s8_mocap, s8_kick3_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick4.plot(kick_frames.scaled_time_kick4_s8_mocap, s8_kick4_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick5.plot(kick_frames.scaled_time_kick5_s8_mocap, s8_kick5_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick6.plot(kick_frames.scaled_time_kick6_s8_mocap, s8_kick6_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick7.plot(kick_frames.scaled_time_kick7_s8_mocap, s8_kick7_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick8.plot(kick_frames.scaled_time_kick8_s8_mocap, s8_kick8_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick9.plot(kick_frames.scaled_time_kick9_s8_mocap, s8_kick9_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick10.plot(kick_frames.scaled_time_kick10_s8_mocap, s8_kick10_window_fp.loc[:, "FP4_Fz"], color = fp_data.s8k5_color)
# Fz_s8_kick1.plot(s8_kick1_footoff_scaledidx, s8_kick1_footoff_Fzval, 'o')
# Fz_s8_kick2.plot(s8_kick2_footoff_scaledidx, s8_kick2_footoff_Fzval, 'o')
# Fz_s8_kick3.plot(s8_kick3_footoff_scaledidx, s8_kick3_footoff_Fzval, 'o')
# Fz_s8_kick4.plot(s8_kick4_footoff_scaledidx, s8_kick4_footoff_Fzval, 'o')
# Fz_s8_kick5.plot(s8_kick5_footoff_scaledidx, s8_kick5_footoff_Fzval, 'o')
# Fz_s8_kick6.plot(s8_kick6_footoff_scaledidx, s8_kick6_footoff_Fzval, 'o')
# Fz_s8_kick7.plot(s8_kick7_footoff_scaledidx, s8_kick7_footoff_Fzval, 'o')
# Fz_s8_kick8.plot(s8_kick8_footoff_scaledidx, s8_kick8_footoff_Fzval, 'o')
# Fz_s8_kick9.plot(s8_kick9_footoff_scaledidx, s8_kick9_footoff_Fzval, 'o')
# Fz_s8_kick10.plot(s8_kick10_footoff_scaledidx, s8_kick10_footoff_Fzval, 'o')
# Fz_s8_kick1.plot(s8_kick1_footon_scaledidx, s8_kick1_footon_Fzval, 'x')
# Fz_s8_kick2.plot(s8_kick2_footon_scaledidx, s8_kick2_footon_Fzval, 'x')
# Fz_s8_kick3.plot(s8_kick3_footon_scaledidx, s8_kick3_footon_Fzval, 'x')
# Fz_s8_kick4.plot(s8_kick4_footon_scaledidx, s8_kick4_footon_Fzval, 'x')
# Fz_s8_kick5.plot(s8_kick5_footon_scaledidx, s8_kick5_footon_Fzval, 'x')
# Fz_s8_kick6.plot(s8_kick6_footon_scaledidx, s8_kick6_footon_Fzval, 'x')
# Fz_s8_kick7.plot(s8_kick7_footon_scaledidx, s8_kick7_footon_Fzval, 'x')
# Fz_s8_kick8.plot(s8_kick8_footon_scaledidx, s8_kick8_footon_Fzval, 'x')
# Fz_s8_kick9.plot(s8_kick9_footon_scaledidx, s8_kick9_footon_Fzval, 'x')
# Fz_s8_kick10.plot(s8_kick10_footon_scaledidx, s8_kick10_footon_Fzval, 'x')

# Fz_s8_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_s8_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_s8_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_s8_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_s8_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_s8_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_s8_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_s8_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_s8_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_s8_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_s8_figs.suptitle('Session 8 Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_s8_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_s8_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_s8_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/s8_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as s8_kickfoot_on_off_FP.jpg")

# # Expert for Comparison
# Fz_expert_figs, ((Fz_expert_kick1, Fz_expert_kick2, Fz_expert_kick3, Fz_expert_kick4, Fz_expert_kick5),
# 			 (Fz_expert_kick6, Fz_expert_kick7, Fz_expert_kick8, Fz_expert_kick9, Fz_expert_kick10)) = plt.subplots(2,5)
# Fz_expert_figs.set_figheight(10)
# Fz_expert_figs.set_figwidth(25)

# Fz_expert_kick1.plot(kick_frames.scaled_time_kick1_expert_mocap, expert_kick1_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick2.plot(kick_frames.scaled_time_kick2_expert_mocap, expert_kick2_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick3.plot(kick_frames.scaled_time_kick3_expert_mocap, expert_kick3_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick4.plot(kick_frames.scaled_time_kick4_expert_mocap, expert_kick4_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick5.plot(kick_frames.scaled_time_kick5_expert_mocap, expert_kick5_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick6.plot(kick_frames.scaled_time_kick6_expert_mocap, expert_kick6_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick7.plot(kick_frames.scaled_time_kick7_expert_mocap, expert_kick7_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick8.plot(kick_frames.scaled_time_kick8_expert_mocap, expert_kick8_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick9.plot(kick_frames.scaled_time_kick9_expert_mocap, expert_kick9_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick10.plot(kick_frames.scaled_time_kick10_expert_mocap, expert_kick10_window_fp.loc[:, "FP5_Fz"], color = fp_data.expertk10_color)
# Fz_expert_kick1.plot(expert_kick1_footoff_scaledidx, expert_kick1_footoff_Fzval, 'o')
# Fz_expert_kick2.plot(expert_kick2_footoff_scaledidx, expert_kick2_footoff_Fzval, 'o')
# Fz_expert_kick3.plot(expert_kick3_footoff_scaledidx, expert_kick3_footoff_Fzval, 'o')
# Fz_expert_kick4.plot(expert_kick4_footoff_scaledidx, expert_kick4_footoff_Fzval, 'o')
# Fz_expert_kick5.plot(expert_kick5_footoff_scaledidx, expert_kick5_footoff_Fzval, 'o')
# Fz_expert_kick6.plot(expert_kick6_footoff_scaledidx, expert_kick6_footoff_Fzval, 'o')
# Fz_expert_kick7.plot(expert_kick7_footoff_scaledidx, expert_kick7_footoff_Fzval, 'o')
# Fz_expert_kick8.plot(expert_kick8_footoff_scaledidx, expert_kick8_footoff_Fzval, 'o')
# Fz_expert_kick9.plot(expert_kick9_footoff_scaledidx, expert_kick9_footoff_Fzval, 'o')
# Fz_expert_kick10.plot(expert_kick10_footoff_scaledidx, expert_kick10_footoff_Fzval, 'o')
# Fz_expert_kick1.plot(expert_kick1_footon_scaledidx, expert_kick1_footon_Fzval, 'x')
# Fz_expert_kick2.plot(expert_kick2_footon_scaledidx, expert_kick2_footon_Fzval, 'x')
# Fz_expert_kick3.plot(expert_kick3_footon_scaledidx, expert_kick3_footon_Fzval, 'x')
# Fz_expert_kick4.plot(expert_kick4_footon_scaledidx, expert_kick4_footon_Fzval, 'x')
# Fz_expert_kick5.plot(expert_kick5_footon_scaledidx, expert_kick5_footon_Fzval, 'x')
# Fz_expert_kick6.plot(expert_kick6_footon_scaledidx, expert_kick6_footon_Fzval, 'x')
# Fz_expert_kick7.plot(expert_kick7_footon_scaledidx, expert_kick7_footon_Fzval, 'x')
# Fz_expert_kick8.plot(expert_kick8_footon_scaledidx, expert_kick8_footon_Fzval, 'x')
# Fz_expert_kick9.plot(expert_kick9_footon_scaledidx, expert_kick9_footon_Fzval, 'x')
# Fz_expert_kick10.plot(expert_kick10_footon_scaledidx, expert_kick10_footon_Fzval, 'x')

# Fz_expert_kick1.set_title("Kick 1", fontsize = fp_data.title_font_size)
# Fz_expert_kick2.set_title("Kick 2", fontsize = fp_data.title_font_size)
# Fz_expert_kick3.set_title("Kick 3", fontsize = fp_data.title_font_size)
# Fz_expert_kick4.set_title("Kick 4", fontsize = fp_data.title_font_size)
# Fz_expert_kick5.set_title("Kick 5", fontsize = fp_data.title_font_size)
# Fz_expert_kick6.set_title("Kick 6", fontsize = fp_data.title_font_size)
# Fz_expert_kick7.set_title("Kick 7", fontsize = fp_data.title_font_size)
# Fz_expert_kick8.set_title("Kick 8", fontsize = fp_data.title_font_size)
# Fz_expert_kick9.set_title("Kick 9", fontsize = fp_data.title_font_size)
# Fz_expert_kick10.set_title("Kick 10", fontsize = fp_data.title_font_size)

# Fz_expert_figs.suptitle('Expert for Comparison Kicking Foot Vertical Forces during Back-leg Roundhouse Kick', fontsize = 24) #, y = 0.92)
# Fz_expert_figs.supxlabel('Scaled Time', fontsize = 18) #, y = 0.075)
# Fz_expert_figs.supylabel('Force (N)', fontsize = 18) #, x = 0.075)
# Fz_expert_figs.savefig("FP_plots_fulltraj/verify_foot_onoff_FP/resampled/expert_kickfoot_on_off_FP.jpg")
# print("Figure saved in FP_plots_fulltraj/verify_foot_onoff_FP/resampled folder as expert_kickfoot_on_off_FP.jpg")