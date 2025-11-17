# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 2024

@author: janlau
"""

'''
Jan Lau's PhD Project 1: Longitudinal Right Back-leg Roundhouse Kicks
Plots of interest: Whatever plot it is but with each phase scaled

Recall the five phases and points of segmentation:

|       Preparation             |         Chambering               |            Extension              |                Retraction                    |            Termination               |
0 [stand still] ------> 1 [lift kick foot] 0 ------> 1 [max kick knee flexion] 0 ------> 1 [strike / max knee extension] 0 ----------------> 1 [kick foot on gnd] 0 -------------> [Kick ends]

This code is intended to identify all frames of each of each kick for all sessions.
    1. INTERPOLATE DATA (numpy.interp)
    2. Read into kick_frames to narrow down the start and end frames of each kick to consider for detecting foot off and foot on FP
    3. Detect lifting of kicking (R) foot by finding when Fz reaches -4 AND save frame number
    4. Detect maximum knee flexion of kicking foot (see identify_knee_flexions.py for inspo) AND save frame number
    5. Detect maximum knee extension of kicking foot (see identify_knee_extensions.py for inspo) AND save frame number
    6. Detect ground contact of kicking foot by finding when Fz < -4 (i.t.o. raw data where -ve means contact with ground)

This code has time and scaled time arrays.

Scaled time: Yes
Kick segmentation: Soon

Notes:
	- Kicking leg = RIGHT leg
	- Supporting leg = LEFT leg
	- s1, s2, s3: FP1 is supporting leg, FP2 is kicking leg
	- s4, s5, s6, s7: FP4 is supporting leg, FP5 is kicking leg
	- s8: FP5 is supporting leg, FP4 is kicking leg
    - expert: FP4 is supporting leg, FP5 is kicking leg
'''

import matplotlib.pyplot as plt
import identify_knee_flexions_extensions as knee_flx_exts # To obtain max knee flexion and extension for phase changes
import identify_kickfoot_onoff_resampled100Hz_FP as fp_on_off # To obtain foot off and on FP instances for phase changes
import scipy # for find_peaks
import numpy as np
import roundhousekicks_frame_segmentation as kick_frames
import setup_for_plot_kinematic_data as kinematic_data # Load the data and other setups etc.


def obtain_phase_time_array_variations(start_idx, end_idx):
    t_array_idx = np.arange(start_idx - start_idx, end_idx - start_idx + 1, 1) # final point included because it will be the first point of the next phase when concatenated
    t_array = t_array_idx / kick_frames.vicon_frequency
    t_array_scaled = t_array_idx / (end_idx - start_idx)
    return[t_array, t_array_scaled]

def concatenate_time_array_variations(phase1_t_array, phase1_t_array_scaled, phase2_t_array, phase2_t_array_scaled,
                                      phase3_t_array, phase3_t_array_scaled, phase4_t_array, phase4_t_array_scaled,
                                      phase5_t_array, phase5_t_array_scaled):

    # print("p1 len: ", len(phase1_t_array_scaled))
    # print("p2 len: ", len(phase2_t_array_scaled))
    # print("p3 len: ", len(phase3_t_array_scaled))
    # print("p4 len: ", len(phase4_t_array_scaled))
    # print("p5 len: ", len(phase5_t_array_scaled))
    
    allphases_t_array = np.concatenate((phase1_t_array, phase2_t_array[1:] + phase1_t_array[-1],
                                        phase3_t_array[1:] + phase2_t_array[-1] + phase1_t_array[-1],
                                        phase4_t_array[1:] + phase3_t_array[-1] + phase2_t_array[-1] + phase1_t_array[-1],
                                        phase5_t_array[1:] + phase4_t_array[-1] + phase3_t_array[-1] + phase2_t_array[-1] + phase1_t_array[-1]))
    allphases_t_array_scaled = np.concatenate((phase1_t_array_scaled, phase2_t_array_scaled[1:] + phase1_t_array_scaled[-1],
                                               phase3_t_array_scaled[1:] + phase2_t_array_scaled[-1] + 1,
                                               phase4_t_array_scaled[1:] + phase3_t_array_scaled[-1] + 2,
                                               phase5_t_array_scaled[1:] + phase4_t_array_scaled[-1] + 3))
    
    # print("total len: ", allphases_t_array_scaled)

    return[allphases_t_array, allphases_t_array_scaled]

''' PHASE 1: From kick_frames.roundhousekicks_start_end_frames_s#_mocap[0] (0) to fp_on_off.s#_kick#_footoff_idx (1) '''
# -- Calculate time arrays (scaled and non-scaled)
# ---- Session 1
[s1_kick1_phase1_t_array, s1_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[0,0], fp_on_off.s1_kick1_footoff_idx)
[s1_kick2_phase1_t_array, s1_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[1,0], fp_on_off.s1_kick2_footoff_idx)
[s1_kick3_phase1_t_array, s1_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[2,0], fp_on_off.s1_kick3_footoff_idx)
[s1_kick4_phase1_t_array, s1_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[3,0], fp_on_off.s1_kick4_footoff_idx) # Unstable kick
[s1_kick5_phase1_t_array, s1_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[4,0], fp_on_off.s1_kick5_footoff_idx2) # Unstable kick
[s1_kick6_phase1_t_array, s1_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[5,0], fp_on_off.s1_kick6_footoff_idx) # Unstable kick
[s1_kick7_phase1_t_array, s1_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[6,0], fp_on_off.s1_kick7_footoff_idx)
[s1_kick8_phase1_t_array, s1_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[7,0], fp_on_off.s1_kick8_footoff_idx)
[s1_kick9_phase1_t_array, s1_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[8,0], fp_on_off.s1_kick9_footoff_idx)
[s1_kick10_phase1_t_array, s1_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s1_mocap[9,0], fp_on_off.s1_kick10_footoff_idx) # Unstable kick

# ---- Session 2
[s2_kick1_phase1_t_array, s2_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[0,0], fp_on_off.s2_kick1_footoff_idx)
[s2_kick2_phase1_t_array, s2_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[1,0], fp_on_off.s2_kick2_footoff_idx)
[s2_kick3_phase1_t_array, s2_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[2,0], fp_on_off.s2_kick3_footoff_idx)
[s2_kick4_phase1_t_array, s2_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[3,0], fp_on_off.s2_kick4_footoff_idx)
[s2_kick5_phase1_t_array, s2_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[4,0], fp_on_off.s2_kick5_footoff_idx)
[s2_kick6_phase1_t_array, s2_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[5,0], fp_on_off.s2_kick6_footoff_idx)
[s2_kick7_phase1_t_array, s2_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[6,0], fp_on_off.s2_kick7_footoff_idx)
[s2_kick8_phase1_t_array, s2_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[7,0], fp_on_off.s2_kick8_footoff_idx)
[s2_kick9_phase1_t_array, s2_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[8,0], fp_on_off.s2_kick9_footoff_idx)
[s2_kick10_phase1_t_array, s2_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s2_mocap[9,0], fp_on_off.s2_kick10_footoff_idx)

# ---- Session 3
[s3_kick1_phase1_t_array, s3_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[0,0], fp_on_off.s3_kick1_footoff_idx)
[s3_kick2_phase1_t_array, s3_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[1,0], fp_on_off.s3_kick2_footoff_idx)
[s3_kick3_phase1_t_array, s3_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[2,0], fp_on_off.s3_kick3_footoff_idx)
[s3_kick4_phase1_t_array, s3_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[3,0], fp_on_off.s3_kick4_footoff_idx)
[s3_kick5_phase1_t_array, s3_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[4,0], fp_on_off.s3_kick5_footoff_idx)
[s3_kick6_phase1_t_array, s3_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[5,0], fp_on_off.s3_kick6_footoff_idx)
[s3_kick7_phase1_t_array, s3_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[6,0], fp_on_off.s3_kick7_footoff_idx)
[s3_kick8_phase1_t_array, s3_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[7,0], fp_on_off.s3_kick8_footoff_idx)
[s3_kick9_phase1_t_array, s3_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[8,0], fp_on_off.s3_kick9_footoff_idx)
[s3_kick10_phase1_t_array, s3_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s3_mocap[9,0], fp_on_off.s3_kick10_footoff_idx)

# ---- Session 4
[s4_kick1_phase1_t_array, s4_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[0,0], fp_on_off.s4_kick1_footoff_idx)
[s4_kick2_phase1_t_array, s4_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[1,0], fp_on_off.s4_kick2_footoff_idx)
[s4_kick3_phase1_t_array, s4_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[2,0], fp_on_off.s4_kick3_footoff_idx)
[s4_kick4_phase1_t_array, s4_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[3,0], fp_on_off.s4_kick4_footoff_idx)
[s4_kick5_phase1_t_array, s4_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[4,0], fp_on_off.s4_kick5_footoff_idx)
[s4_kick6_phase1_t_array, s4_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[5,0], fp_on_off.s4_kick6_footoff_idx)
[s4_kick7_phase1_t_array, s4_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[6,0], fp_on_off.s4_kick7_footoff_idx)
[s4_kick8_phase1_t_array, s4_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[7,0], fp_on_off.s4_kick8_footoff_idx)
[s4_kick9_phase1_t_array, s4_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[8,0], fp_on_off.s4_kick9_footoff_idx)
[s4_kick10_phase1_t_array, s4_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s4_mocap[9,0], fp_on_off.s4_kick10_footoff_idx)

# ---- Session 5
[s5_kick1_phase1_t_array, s5_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[0,0], fp_on_off.s5_kick1_footoff_idx)
[s5_kick2_phase1_t_array, s5_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[1,0], fp_on_off.s5_kick2_footoff_idx)
[s5_kick3_phase1_t_array, s5_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[2,0], fp_on_off.s5_kick3_footoff_idx)
[s5_kick4_phase1_t_array, s5_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[3,0], fp_on_off.s5_kick4_footoff_idx)
[s5_kick5_phase1_t_array, s5_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[4,0], fp_on_off.s5_kick5_footoff_idx)
[s5_kick6_phase1_t_array, s5_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[5,0], fp_on_off.s5_kick6_footoff_idx)
[s5_kick7_phase1_t_array, s5_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[6,0], fp_on_off.s5_kick7_footoff_idx)
[s5_kick8_phase1_t_array, s5_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[7,0], fp_on_off.s5_kick8_footoff_idx)
[s5_kick9_phase1_t_array, s5_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[8,0], fp_on_off.s5_kick9_footoff_idx)
[s5_kick10_phase1_t_array, s5_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s5_mocap[9,0], fp_on_off.s5_kick10_footoff_idx)

# ---- Session 6
[s6_kick1_phase1_t_array, s6_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[0,0], fp_on_off.s6_kick1_footoff_idx)
[s6_kick2_phase1_t_array, s6_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[1,0], fp_on_off.s6_kick2_footoff_idx)
[s6_kick3_phase1_t_array, s6_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[2,0], fp_on_off.s6_kick3_footoff_idx)
[s6_kick4_phase1_t_array, s6_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[3,0], fp_on_off.s6_kick4_footoff_idx)
[s6_kick5_phase1_t_array, s6_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[4,0], fp_on_off.s6_kick5_footoff_idx)
[s6_kick6_phase1_t_array, s6_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[5,0], fp_on_off.s6_kick6_footoff_idx)
[s6_kick7_phase1_t_array, s6_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[6,0], fp_on_off.s6_kick7_footoff_idx)
[s6_kick8_phase1_t_array, s6_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[7,0], fp_on_off.s6_kick8_footoff_idx)
[s6_kick9_phase1_t_array, s6_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[8,0], fp_on_off.s6_kick9_footoff_idx)
[s6_kick10_phase1_t_array, s6_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s6_mocap[9,0], fp_on_off.s6_kick10_footoff_idx)

# ---- Session 7
[s7_kick1_phase1_t_array, s7_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[0,0], fp_on_off.s7_kick1_footoff_idx)
[s7_kick2_phase1_t_array, s7_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[1,0], fp_on_off.s7_kick2_footoff_idx)
[s7_kick3_phase1_t_array, s7_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[2,0], fp_on_off.s7_kick3_footoff_idx)
[s7_kick4_phase1_t_array, s7_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[3,0], fp_on_off.s7_kick4_footoff_idx)
[s7_kick5_phase1_t_array, s7_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[4,0], fp_on_off.s7_kick5_footoff_idx)
[s7_kick6_phase1_t_array, s7_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[5,0], fp_on_off.s7_kick6_footoff_idx)
[s7_kick7_phase1_t_array, s7_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[6,0], fp_on_off.s7_kick7_footoff_idx)
[s7_kick8_phase1_t_array, s7_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[7,0], fp_on_off.s7_kick8_footoff_idx)
[s7_kick9_phase1_t_array, s7_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[8,0], fp_on_off.s7_kick9_footoff_idx)
[s7_kick10_phase1_t_array, s7_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s7_mocap[9,0], fp_on_off.s7_kick10_footoff_idx)

# ---- Session 8
[s8_kick1_phase1_t_array, s8_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[0,0], fp_on_off.s8_kick1_footoff_idx)
[s8_kick2_phase1_t_array, s8_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[1,0], fp_on_off.s8_kick2_footoff_idx)
[s8_kick3_phase1_t_array, s8_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[2,0], fp_on_off.s8_kick3_footoff_idx)
[s8_kick4_phase1_t_array, s8_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[3,0], fp_on_off.s8_kick4_footoff_idx)
[s8_kick5_phase1_t_array, s8_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[4,0], fp_on_off.s8_kick5_footoff_idx)
[s8_kick6_phase1_t_array, s8_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[5,0], fp_on_off.s8_kick6_footoff_idx)
[s8_kick7_phase1_t_array, s8_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[6,0], fp_on_off.s8_kick7_footoff_idx)
[s8_kick8_phase1_t_array, s8_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[7,0], fp_on_off.s8_kick8_footoff_idx)
[s8_kick9_phase1_t_array, s8_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[8,0], fp_on_off.s8_kick9_footoff_idx)
[s8_kick10_phase1_t_array, s8_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_s8_mocap[9,0], fp_on_off.s8_kick10_footoff_idx)

# ---- Expert
[expert_kick1_phase1_t_array, expert_kick1_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[0,0], fp_on_off.expert_kick1_footoff_idx)
[expert_kick2_phase1_t_array, expert_kick2_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[1,0], fp_on_off.expert_kick2_footoff_idx)
[expert_kick3_phase1_t_array, expert_kick3_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[2,0], fp_on_off.expert_kick3_footoff_idx)
[expert_kick4_phase1_t_array, expert_kick4_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[3,0], fp_on_off.expert_kick4_footoff_idx)
[expert_kick5_phase1_t_array, expert_kick5_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[4,0], fp_on_off.expert_kick5_footoff_idx)
[expert_kick6_phase1_t_array, expert_kick6_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[5,0], fp_on_off.expert_kick6_footoff_idx)
[expert_kick7_phase1_t_array, expert_kick7_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[6,0], fp_on_off.expert_kick7_footoff_idx)
[expert_kick8_phase1_t_array, expert_kick8_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[7,0], fp_on_off.expert_kick8_footoff_idx)
[expert_kick9_phase1_t_array, expert_kick9_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[8,0], fp_on_off.expert_kick9_footoff_idx)
[expert_kick10_phase1_t_array, expert_kick10_phase1_t_array_scaled] = obtain_phase_time_array_variations(kick_frames.roundhousekicks_start_end_frames_expert_mocap[9,0], fp_on_off.expert_kick10_footoff_idx) # UNSTABLE


''' PHASE 2: From fp_on_off.s#_kick#_footoff_idx (0) to max knee flexion (Chambering) '''
# -- Calculate time arrays (scaled and non-scaled)
# ---- Session 1
[s1_kick1_phase2_t_array, s1_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick1_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[0])
[s1_kick2_phase2_t_array, s1_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick2_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[1])
[s1_kick3_phase2_t_array, s1_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick3_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[2])
[s1_kick4_phase2_t_array, s1_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick4_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[3])
[s1_kick5_phase2_t_array, s1_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick5_footoff_idx2, knee_flx_exts.knee_flexions_s1_indices[4])
[s1_kick6_phase2_t_array, s1_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick6_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[5])
[s1_kick7_phase2_t_array, s1_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick7_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[6])
[s1_kick8_phase2_t_array, s1_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick8_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[7])
[s1_kick9_phase2_t_array, s1_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick9_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[8])
[s1_kick10_phase2_t_array, s1_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick10_footoff_idx, knee_flx_exts.knee_flexions_s1_indices[9])

# ---- Session 2
[s2_kick1_phase2_t_array, s2_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick1_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[0])
[s2_kick2_phase2_t_array, s2_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick2_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[1])
[s2_kick3_phase2_t_array, s2_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick3_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[2])
[s2_kick4_phase2_t_array, s2_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick4_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[3])
[s2_kick5_phase2_t_array, s2_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick5_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[4])
[s2_kick6_phase2_t_array, s2_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick6_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[5])
[s2_kick7_phase2_t_array, s2_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick7_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[6])
[s2_kick8_phase2_t_array, s2_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick8_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[7])
[s2_kick9_phase2_t_array, s2_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick9_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[8])
[s2_kick10_phase2_t_array, s2_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick10_footoff_idx, knee_flx_exts.knee_flexions_s2_indices[9])

# ---- Session 3
[s3_kick1_phase2_t_array, s3_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick1_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[0])
[s3_kick2_phase2_t_array, s3_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick2_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[1])
[s3_kick3_phase2_t_array, s3_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick3_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[2])
[s3_kick4_phase2_t_array, s3_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick4_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[3])
[s3_kick5_phase2_t_array, s3_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick5_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[4])
[s3_kick6_phase2_t_array, s3_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick6_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[5])
[s3_kick7_phase2_t_array, s3_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick7_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[6])
[s3_kick8_phase2_t_array, s3_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick8_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[7])
[s3_kick9_phase2_t_array, s3_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick9_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[8])
[s3_kick10_phase2_t_array, s3_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick10_footoff_idx, knee_flx_exts.knee_flexions_s3_indices[9])

# ---- Session 4
[s4_kick1_phase2_t_array, s4_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick1_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[0])
[s4_kick2_phase2_t_array, s4_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick2_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[1])
[s4_kick3_phase2_t_array, s4_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick3_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[2])
[s4_kick4_phase2_t_array, s4_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick4_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[3])
[s4_kick5_phase2_t_array, s4_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick5_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[4])
[s4_kick6_phase2_t_array, s4_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick6_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[5])
[s4_kick7_phase2_t_array, s4_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick7_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[6])
[s4_kick8_phase2_t_array, s4_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick8_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[7])
[s4_kick9_phase2_t_array, s4_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick9_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[8])
[s4_kick10_phase2_t_array, s4_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick10_footoff_idx, knee_flx_exts.knee_flexions_s4_indices[9])

# ---- Session 5
[s5_kick1_phase2_t_array, s5_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick1_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[0])
[s5_kick2_phase2_t_array, s5_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick2_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[1])
[s5_kick3_phase2_t_array, s5_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick3_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[2])
[s5_kick4_phase2_t_array, s5_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick4_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[3])
[s5_kick5_phase2_t_array, s5_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick5_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[4])
[s5_kick6_phase2_t_array, s5_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick6_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[5])
[s5_kick7_phase2_t_array, s5_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick7_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[6])
[s5_kick8_phase2_t_array, s5_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick8_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[7])
[s5_kick9_phase2_t_array, s5_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick9_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[8])
[s5_kick10_phase2_t_array, s5_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick10_footoff_idx, knee_flx_exts.knee_flexions_s5_indices[9])

# ---- Session 6
[s6_kick1_phase2_t_array, s6_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick1_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[0])
[s6_kick2_phase2_t_array, s6_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick2_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[1])
[s6_kick3_phase2_t_array, s6_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick3_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[2])
[s6_kick4_phase2_t_array, s6_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick4_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[3])
[s6_kick5_phase2_t_array, s6_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick5_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[4])
[s6_kick6_phase2_t_array, s6_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick6_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[5])
[s6_kick7_phase2_t_array, s6_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick7_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[6])
[s6_kick8_phase2_t_array, s6_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick8_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[7])
[s6_kick9_phase2_t_array, s6_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick9_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[8])
[s6_kick10_phase2_t_array, s6_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick10_footoff_idx, knee_flx_exts.knee_flexions_s6_indices[9])

# ---- Session 7
[s7_kick1_phase2_t_array, s7_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick1_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[0])
[s7_kick2_phase2_t_array, s7_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick2_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[1])
[s7_kick3_phase2_t_array, s7_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick3_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[2])
[s7_kick4_phase2_t_array, s7_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick4_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[3])
[s7_kick5_phase2_t_array, s7_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick5_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[4])
[s7_kick6_phase2_t_array, s7_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick6_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[5])
[s7_kick7_phase2_t_array, s7_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick7_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[6])
[s7_kick8_phase2_t_array, s7_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick8_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[7])
[s7_kick9_phase2_t_array, s7_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick9_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[8])
[s7_kick10_phase2_t_array, s7_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick10_footoff_idx, knee_flx_exts.knee_flexions_s7_indices[9])

# ---- Session 8
[s8_kick1_phase2_t_array, s8_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick1_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[0])
[s8_kick2_phase2_t_array, s8_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick2_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[1])
[s8_kick3_phase2_t_array, s8_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick3_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[2])
[s8_kick4_phase2_t_array, s8_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick4_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[3])
[s8_kick5_phase2_t_array, s8_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick5_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[4])
[s8_kick6_phase2_t_array, s8_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick6_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[5])
[s8_kick7_phase2_t_array, s8_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick7_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[6])
[s8_kick8_phase2_t_array, s8_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick8_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[7])
[s8_kick9_phase2_t_array, s8_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick9_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[8])
[s8_kick10_phase2_t_array, s8_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick10_footoff_idx, knee_flx_exts.knee_flexions_s8_indices[9])

# ---- Expert
[expert_kick1_phase2_t_array, expert_kick1_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick1_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[0])
[expert_kick2_phase2_t_array, expert_kick2_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick2_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[1])
[expert_kick3_phase2_t_array, expert_kick3_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick3_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[2])
[expert_kick4_phase2_t_array, expert_kick4_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick4_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[3])
[expert_kick5_phase2_t_array, expert_kick5_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick5_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[4])
[expert_kick6_phase2_t_array, expert_kick6_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick6_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[5])
[expert_kick7_phase2_t_array, expert_kick7_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick7_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[6])
[expert_kick8_phase2_t_array, expert_kick8_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick8_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[7])
[expert_kick9_phase2_t_array, expert_kick9_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick9_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[8])
[expert_kick10_phase2_t_array, expert_kick10_phase2_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick10_footoff_idx, knee_flx_exts.knee_flexions_expert_indices[9]) # UNSTABLE


''' PHASE 3: From max knee flexion to max knee extension (Retraction) '''
# -- Calculate time arrays (scaled and non-scaled)
# ---- Session 1
[s1_kick1_phase3_t_array, s1_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[0], knee_flx_exts.knee_extensions_s1_indices[0])
[s1_kick2_phase3_t_array, s1_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[1], knee_flx_exts.knee_extensions_s1_indices[1])
[s1_kick3_phase3_t_array, s1_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[2], knee_flx_exts.knee_extensions_s1_indices[2])
[s1_kick4_phase3_t_array, s1_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[3], knee_flx_exts.knee_extensions_s1_indices[3])
[s1_kick5_phase3_t_array, s1_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[4], knee_flx_exts.knee_extensions_s1_indices[4])
[s1_kick6_phase3_t_array, s1_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[5], knee_flx_exts.knee_extensions_s1_indices[5])
[s1_kick7_phase3_t_array, s1_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[6], knee_flx_exts.knee_extensions_s1_indices[6])
[s1_kick8_phase3_t_array, s1_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[7], knee_flx_exts.knee_extensions_s1_indices[7])
[s1_kick9_phase3_t_array, s1_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[8], knee_flx_exts.knee_extensions_s1_indices[8])
[s1_kick10_phase3_t_array, s1_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s1_indices[9], knee_flx_exts.knee_extensions_s1_indices[9])

# ---- Session 2
[s2_kick1_phase3_t_array, s2_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[0], knee_flx_exts.knee_extensions_s2_indices[0])
[s2_kick2_phase3_t_array, s2_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[1], knee_flx_exts.knee_extensions_s2_indices[1])
[s2_kick3_phase3_t_array, s2_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[2], knee_flx_exts.knee_extensions_s2_indices[2])
[s2_kick4_phase3_t_array, s2_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[3], knee_flx_exts.knee_extensions_s2_indices[3])
[s2_kick5_phase3_t_array, s2_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[4], knee_flx_exts.knee_extensions_s2_indices[4])
[s2_kick6_phase3_t_array, s2_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[5], knee_flx_exts.knee_extensions_s2_indices[5])
[s2_kick7_phase3_t_array, s2_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[6], knee_flx_exts.knee_extensions_s2_indices[6])
[s2_kick8_phase3_t_array, s2_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[7], knee_flx_exts.knee_extensions_s2_indices[7])
[s2_kick9_phase3_t_array, s2_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[8], knee_flx_exts.knee_extensions_s2_indices[8])
[s2_kick10_phase3_t_array, s2_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s2_indices[9], knee_flx_exts.knee_extensions_s2_indices[9])

# ---- Session 3
[s3_kick1_phase3_t_array, s3_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[0], knee_flx_exts.knee_extensions_s3_indices[0])
[s3_kick2_phase3_t_array, s3_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[1], knee_flx_exts.knee_extensions_s3_indices[1])
[s3_kick3_phase3_t_array, s3_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[2], knee_flx_exts.knee_extensions_s3_indices[2])
[s3_kick4_phase3_t_array, s3_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[3], knee_flx_exts.knee_extensions_s3_indices[3])
[s3_kick5_phase3_t_array, s3_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[4], knee_flx_exts.knee_extensions_s3_indices[4])
[s3_kick6_phase3_t_array, s3_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[5], knee_flx_exts.knee_extensions_s3_indices[5])
[s3_kick7_phase3_t_array, s3_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[6], knee_flx_exts.knee_extensions_s3_indices[6])
[s3_kick8_phase3_t_array, s3_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[7], knee_flx_exts.knee_extensions_s3_indices[7])
[s3_kick9_phase3_t_array, s3_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[8], knee_flx_exts.knee_extensions_s3_indices[8])
[s3_kick10_phase3_t_array, s3_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s3_indices[9], knee_flx_exts.knee_extensions_s3_indices[9])

# ---- Session 4
[s4_kick1_phase3_t_array, s4_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[0], knee_flx_exts.knee_extensions_s4_indices[0])
[s4_kick2_phase3_t_array, s4_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[1], knee_flx_exts.knee_extensions_s4_indices[1])
[s4_kick3_phase3_t_array, s4_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[2], knee_flx_exts.knee_extensions_s4_indices[2])
[s4_kick4_phase3_t_array, s4_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[3], knee_flx_exts.knee_extensions_s4_indices[3])
[s4_kick5_phase3_t_array, s4_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[4], knee_flx_exts.knee_extensions_s4_indices[4])
[s4_kick6_phase3_t_array, s4_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[5], knee_flx_exts.knee_extensions_s4_indices[5])
[s4_kick7_phase3_t_array, s4_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[6], knee_flx_exts.knee_extensions_s4_indices[6])
[s4_kick8_phase3_t_array, s4_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[7], knee_flx_exts.knee_extensions_s4_indices[7])
[s4_kick9_phase3_t_array, s4_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[8], knee_flx_exts.knee_extensions_s4_indices[8])
[s4_kick10_phase3_t_array, s4_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s4_indices[9], knee_flx_exts.knee_extensions_s4_indices[9])

# ---- Session 5
[s5_kick1_phase3_t_array, s5_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[0], knee_flx_exts.knee_extensions_s5_indices[0])
[s5_kick2_phase3_t_array, s5_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[1], knee_flx_exts.knee_extensions_s5_indices[1])
[s5_kick3_phase3_t_array, s5_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[2], knee_flx_exts.knee_extensions_s5_indices[2])
[s5_kick4_phase3_t_array, s5_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[3], knee_flx_exts.knee_extensions_s5_indices[3])
[s5_kick5_phase3_t_array, s5_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[4], knee_flx_exts.knee_extensions_s5_indices[4])
[s5_kick6_phase3_t_array, s5_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[5], knee_flx_exts.knee_extensions_s5_indices[5])
[s5_kick7_phase3_t_array, s5_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[6], knee_flx_exts.knee_extensions_s5_indices[6])
[s5_kick8_phase3_t_array, s5_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[7], knee_flx_exts.knee_extensions_s5_indices[7])
[s5_kick9_phase3_t_array, s5_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[8], knee_flx_exts.knee_extensions_s5_indices[8])
[s5_kick10_phase3_t_array, s5_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s5_indices[9], knee_flx_exts.knee_extensions_s5_indices[9])

# ---- Session 6
[s6_kick1_phase3_t_array, s6_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[0], knee_flx_exts.knee_extensions_s6_indices[0])
[s6_kick2_phase3_t_array, s6_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[1], knee_flx_exts.knee_extensions_s6_indices[1])
[s6_kick3_phase3_t_array, s6_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[2], knee_flx_exts.knee_extensions_s6_indices[2])
[s6_kick4_phase3_t_array, s6_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[3], knee_flx_exts.knee_extensions_s6_indices[3])
[s6_kick5_phase3_t_array, s6_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[4], knee_flx_exts.knee_extensions_s6_indices[4])
[s6_kick6_phase3_t_array, s6_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[5], knee_flx_exts.knee_extensions_s6_indices[5])
[s6_kick7_phase3_t_array, s6_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[6], knee_flx_exts.knee_extensions_s6_indices[6])
[s6_kick8_phase3_t_array, s6_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[7], knee_flx_exts.knee_extensions_s6_indices[7])
[s6_kick9_phase3_t_array, s6_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[8], knee_flx_exts.knee_extensions_s6_indices[8])
[s6_kick10_phase3_t_array, s6_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s6_indices[9], knee_flx_exts.knee_extensions_s6_indices[9])

# ---- Session 7
[s7_kick1_phase3_t_array, s7_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[0], knee_flx_exts.knee_extensions_s7_indices[0])
[s7_kick2_phase3_t_array, s7_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[1], knee_flx_exts.knee_extensions_s7_indices[1])
[s7_kick3_phase3_t_array, s7_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[2], knee_flx_exts.knee_extensions_s7_indices[2])
[s7_kick4_phase3_t_array, s7_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[3], knee_flx_exts.knee_extensions_s7_indices[3])
[s7_kick5_phase3_t_array, s7_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[4], knee_flx_exts.knee_extensions_s7_indices[4])
[s7_kick6_phase3_t_array, s7_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[5], knee_flx_exts.knee_extensions_s7_indices[5])
[s7_kick7_phase3_t_array, s7_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[6], knee_flx_exts.knee_extensions_s7_indices[6])
[s7_kick8_phase3_t_array, s7_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[7], knee_flx_exts.knee_extensions_s7_indices[7])
[s7_kick9_phase3_t_array, s7_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[8], knee_flx_exts.knee_extensions_s7_indices[8])
[s7_kick10_phase3_t_array, s7_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s7_indices[9], knee_flx_exts.knee_extensions_s7_indices[9])

# ---- Session 8
[s8_kick1_phase3_t_array, s8_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[0], knee_flx_exts.knee_extensions_s8_indices[0])
[s8_kick2_phase3_t_array, s8_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[1], knee_flx_exts.knee_extensions_s8_indices[1])
[s8_kick3_phase3_t_array, s8_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[2], knee_flx_exts.knee_extensions_s8_indices[2])
[s8_kick4_phase3_t_array, s8_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[3], knee_flx_exts.knee_extensions_s8_indices[3])
[s8_kick5_phase3_t_array, s8_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[4], knee_flx_exts.knee_extensions_s8_indices[4])
[s8_kick6_phase3_t_array, s8_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[5], knee_flx_exts.knee_extensions_s8_indices[5])
[s8_kick7_phase3_t_array, s8_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[6], knee_flx_exts.knee_extensions_s8_indices[6])
[s8_kick8_phase3_t_array, s8_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[7], knee_flx_exts.knee_extensions_s8_indices[7])
[s8_kick9_phase3_t_array, s8_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[8], knee_flx_exts.knee_extensions_s8_indices[8])
[s8_kick10_phase3_t_array, s8_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_s8_indices[9], knee_flx_exts.knee_extensions_s8_indices[9])

# ---- Expert
[expert_kick1_phase3_t_array, expert_kick1_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[0], knee_flx_exts.knee_extensions_expert_indices[0])
[expert_kick2_phase3_t_array, expert_kick2_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[1], knee_flx_exts.knee_extensions_expert_indices[1])
[expert_kick3_phase3_t_array, expert_kick3_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[2], knee_flx_exts.knee_extensions_expert_indices[2])
[expert_kick4_phase3_t_array, expert_kick4_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[3], knee_flx_exts.knee_extensions_expert_indices[3])
[expert_kick5_phase3_t_array, expert_kick5_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[4], knee_flx_exts.knee_extensions_expert_indices[4])
[expert_kick6_phase3_t_array, expert_kick6_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[5], knee_flx_exts.knee_extensions_expert_indices[5])
[expert_kick7_phase3_t_array, expert_kick7_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[6], knee_flx_exts.knee_extensions_expert_indices[6])
[expert_kick8_phase3_t_array, expert_kick8_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[7], knee_flx_exts.knee_extensions_expert_indices[7])
[expert_kick9_phase3_t_array, expert_kick9_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[8], knee_flx_exts.knee_extensions_expert_indices[8])
[expert_kick10_phase3_t_array, expert_kick10_phase3_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_flexions_expert_indices[9], knee_flx_exts.knee_extensions_expert_indices[9]) # UNSTABLE

''' PHASE 4: From max knee extension to foot on FP (Retraction) '''
# -- Calculate time arrays (scaled and non-scaled)
[s1_kick1_phase4_t_array, s1_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[0], fp_on_off.s1_kick1_footon_idx)
[s1_kick2_phase4_t_array, s1_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[1], fp_on_off.s1_kick2_footon_idx)
[s1_kick3_phase4_t_array, s1_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[2], fp_on_off.s1_kick3_footon_idx)
[s1_kick4_phase4_t_array, s1_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[3], fp_on_off.s1_kick4_footon_idx)
[s1_kick5_phase4_t_array, s1_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[4], fp_on_off.s1_kick5_footon_idx2)
[s1_kick6_phase4_t_array, s1_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[5], fp_on_off.s1_kick6_footon_idx)
[s1_kick7_phase4_t_array, s1_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[6], fp_on_off.s1_kick7_footon_idx)
[s1_kick8_phase4_t_array, s1_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[7], fp_on_off.s1_kick8_footon_idx)
[s1_kick9_phase4_t_array, s1_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[8], fp_on_off.s1_kick9_footon_idx)
[s1_kick10_phase4_t_array, s1_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s1_indices[9], fp_on_off.s1_kick10_footon_idx)

# ---- Session 2
[s2_kick1_phase4_t_array, s2_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[0], fp_on_off.s2_kick1_footon_idx)
[s2_kick2_phase4_t_array, s2_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[1], fp_on_off.s2_kick2_footon_idx)
[s2_kick3_phase4_t_array, s2_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[2], fp_on_off.s2_kick3_footon_idx)
[s2_kick4_phase4_t_array, s2_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[3], fp_on_off.s2_kick4_footon_idx)
[s2_kick5_phase4_t_array, s2_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[4], fp_on_off.s2_kick5_footon_idx)
[s2_kick6_phase4_t_array, s2_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[5], fp_on_off.s2_kick6_footon_idx)
[s2_kick7_phase4_t_array, s2_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[6], fp_on_off.s2_kick7_footon_idx)
[s2_kick8_phase4_t_array, s2_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[7], fp_on_off.s2_kick8_footon_idx)
[s2_kick9_phase4_t_array, s2_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[8], fp_on_off.s2_kick9_footon_idx)
[s2_kick10_phase4_t_array, s2_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s2_indices[9], fp_on_off.s2_kick10_footon_idx)

# ---- Session 3
[s3_kick1_phase4_t_array, s3_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[0], fp_on_off.s3_kick1_footon_idx)
[s3_kick2_phase4_t_array, s3_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[1], fp_on_off.s3_kick2_footon_idx)
[s3_kick3_phase4_t_array, s3_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[2], fp_on_off.s3_kick3_footon_idx)
[s3_kick4_phase4_t_array, s3_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[3], fp_on_off.s3_kick4_footon_idx)
[s3_kick5_phase4_t_array, s3_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[4], fp_on_off.s3_kick5_footon_idx)
[s3_kick6_phase4_t_array, s3_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[5], fp_on_off.s3_kick6_footon_idx)
[s3_kick7_phase4_t_array, s3_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[6], fp_on_off.s3_kick7_footon_idx)
[s3_kick8_phase4_t_array, s3_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[7], fp_on_off.s3_kick8_footon_idx)
[s3_kick9_phase4_t_array, s3_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[8], fp_on_off.s3_kick9_footon_idx)
[s3_kick10_phase4_t_array, s3_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s3_indices[9], fp_on_off.s3_kick10_footon_idx)

# ---- Session 4
[s4_kick1_phase4_t_array, s4_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[0], fp_on_off.s4_kick1_footon_idx)
[s4_kick2_phase4_t_array, s4_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[1], fp_on_off.s4_kick2_footon_idx)
[s4_kick3_phase4_t_array, s4_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[2], fp_on_off.s4_kick3_footon_idx)
[s4_kick4_phase4_t_array, s4_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[3], fp_on_off.s4_kick4_footon_idx)
[s4_kick5_phase4_t_array, s4_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[4], fp_on_off.s4_kick5_footon_idx)
[s4_kick6_phase4_t_array, s4_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[5], fp_on_off.s4_kick6_footon_idx)
[s4_kick7_phase4_t_array, s4_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[6], fp_on_off.s4_kick7_footon_idx)
[s4_kick8_phase4_t_array, s4_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[7], fp_on_off.s4_kick8_footon_idx)
[s4_kick9_phase4_t_array, s4_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[8], fp_on_off.s4_kick9_footon_idx)
[s4_kick10_phase4_t_array, s4_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s4_indices[9], fp_on_off.s4_kick10_footon_idx)

# ---- Session 5
[s5_kick1_phase4_t_array, s5_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[0], fp_on_off.s5_kick1_footon_idx)
[s5_kick2_phase4_t_array, s5_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[1], fp_on_off.s5_kick2_footon_idx)
[s5_kick3_phase4_t_array, s5_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[2], fp_on_off.s5_kick3_footon_idx)
[s5_kick4_phase4_t_array, s5_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[3], fp_on_off.s5_kick4_footon_idx)
[s5_kick5_phase4_t_array, s5_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[4], fp_on_off.s5_kick5_footon_idx)
[s5_kick6_phase4_t_array, s5_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[5], fp_on_off.s5_kick6_footon_idx)
[s5_kick7_phase4_t_array, s5_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[6], fp_on_off.s5_kick7_footon_idx)
[s5_kick8_phase4_t_array, s5_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[7], fp_on_off.s5_kick8_footon_idx)
[s5_kick9_phase4_t_array, s5_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[8], fp_on_off.s5_kick9_footon_idx)
[s5_kick10_phase4_t_array, s5_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s5_indices[9], fp_on_off.s5_kick10_footon_idx)

# ---- Session 6
[s6_kick1_phase4_t_array, s6_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[0], fp_on_off.s6_kick1_footon_idx)
[s6_kick2_phase4_t_array, s6_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[1], fp_on_off.s6_kick2_footon_idx)
[s6_kick3_phase4_t_array, s6_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[2], fp_on_off.s6_kick3_footon_idx)
[s6_kick4_phase4_t_array, s6_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[3], fp_on_off.s6_kick4_footon_idx)
[s6_kick5_phase4_t_array, s6_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[4], fp_on_off.s6_kick5_footon_idx)
[s6_kick6_phase4_t_array, s6_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[5], fp_on_off.s6_kick6_footon_idx)
[s6_kick7_phase4_t_array, s6_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[6], fp_on_off.s6_kick7_footon_idx)
[s6_kick8_phase4_t_array, s6_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[7], fp_on_off.s6_kick8_footon_idx)
[s6_kick9_phase4_t_array, s6_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[8], fp_on_off.s6_kick9_footon_idx)
[s6_kick10_phase4_t_array, s6_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s6_indices[9], fp_on_off.s6_kick10_footon_idx)

# ---- Session 7
[s7_kick1_phase4_t_array, s7_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[0], fp_on_off.s7_kick1_footon_idx)
[s7_kick2_phase4_t_array, s7_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[1], fp_on_off.s7_kick2_footon_idx)
[s7_kick3_phase4_t_array, s7_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[2], fp_on_off.s7_kick3_footon_idx)
[s7_kick4_phase4_t_array, s7_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[3], fp_on_off.s7_kick4_footon_idx)
[s7_kick5_phase4_t_array, s7_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[4], fp_on_off.s7_kick5_footon_idx)
[s7_kick6_phase4_t_array, s7_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[5], fp_on_off.s7_kick6_footon_idx)
[s7_kick7_phase4_t_array, s7_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[6], fp_on_off.s7_kick7_footon_idx)
[s7_kick8_phase4_t_array, s7_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[7], fp_on_off.s7_kick8_footon_idx)
[s7_kick9_phase4_t_array, s7_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[8], fp_on_off.s7_kick9_footon_idx)
[s7_kick10_phase4_t_array, s7_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s7_indices[9], fp_on_off.s7_kick10_footon_idx)

# ---- Session 8
[s8_kick1_phase4_t_array, s8_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[0], fp_on_off.s8_kick1_footon_idx)
[s8_kick2_phase4_t_array, s8_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[1], fp_on_off.s8_kick2_footon_idx)
[s8_kick3_phase4_t_array, s8_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[2], fp_on_off.s8_kick3_footon_idx)
[s8_kick4_phase4_t_array, s8_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[3], fp_on_off.s8_kick4_footon_idx)
[s8_kick5_phase4_t_array, s8_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[4], fp_on_off.s8_kick5_footon_idx)
[s8_kick6_phase4_t_array, s8_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[5], fp_on_off.s8_kick6_footon_idx)
[s8_kick7_phase4_t_array, s8_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[6], fp_on_off.s8_kick7_footon_idx)
[s8_kick8_phase4_t_array, s8_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[7], fp_on_off.s8_kick8_footon_idx)
[s8_kick9_phase4_t_array, s8_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[8], fp_on_off.s8_kick9_footon_idx)
[s8_kick10_phase4_t_array, s8_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_s8_indices[9], fp_on_off.s8_kick10_footon_idx)

# ---- Expert
[expert_kick1_phase4_t_array, expert_kick1_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[0], fp_on_off.expert_kick1_footon_idx)
[expert_kick2_phase4_t_array, expert_kick2_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[1], fp_on_off.expert_kick2_footon_idx)
[expert_kick3_phase4_t_array, expert_kick3_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[2], fp_on_off.expert_kick3_footon_idx)
[expert_kick4_phase4_t_array, expert_kick4_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[3], fp_on_off.expert_kick4_footon_idx)
[expert_kick5_phase4_t_array, expert_kick5_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[4], fp_on_off.expert_kick5_footon_idx)
[expert_kick6_phase4_t_array, expert_kick6_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[5], fp_on_off.expert_kick6_footon_idx)
[expert_kick7_phase4_t_array, expert_kick7_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[6], fp_on_off.expert_kick7_footon_idx)
[expert_kick8_phase4_t_array, expert_kick8_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[7], fp_on_off.expert_kick8_footon_idx)
[expert_kick9_phase4_t_array, expert_kick9_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[8], fp_on_off.expert_kick9_footon_idx)
[expert_kick10_phase4_t_array, expert_kick10_phase4_t_array_scaled] = obtain_phase_time_array_variations(knee_flx_exts.knee_extensions_expert_indices[9], fp_on_off.expert_kick10_footon_idx) # UNSTABLE

''' PHASE 5: From foot on FP to end of kick (Termination) '''
# -- Calculate time arrays (scaled and non-scaled)
# ---- Session 1
[s1_kick1_phase5_t_array, s1_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[0,1])
[s1_kick2_phase5_t_array, s1_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[1,1])
[s1_kick3_phase5_t_array, s1_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[2,1])
[s1_kick4_phase5_t_array, s1_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[3,1])
[s1_kick5_phase5_t_array, s1_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick5_footon_idx2, kick_frames.roundhousekicks_start_end_frames_s1_mocap[4,1])
[s1_kick6_phase5_t_array, s1_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[5,1])
[s1_kick7_phase5_t_array, s1_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[6,1])
[s1_kick8_phase5_t_array, s1_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[7,1])
[s1_kick9_phase5_t_array, s1_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[8,1])
[s1_kick10_phase5_t_array, s1_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s1_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_s1_mocap[9,1])

# ---- Session 2
[s2_kick1_phase5_t_array, s2_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[0,1])
[s2_kick2_phase5_t_array, s2_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[1,1])
[s2_kick3_phase5_t_array, s2_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[2,1])
[s2_kick4_phase5_t_array, s2_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[3,1])
[s2_kick5_phase5_t_array, s2_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick5_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[4,1])
[s2_kick6_phase5_t_array, s2_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[5,1])
[s2_kick7_phase5_t_array, s2_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[6,1])
[s2_kick8_phase5_t_array, s2_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[7,1])
[s2_kick9_phase5_t_array, s2_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[8,1])
[s2_kick10_phase5_t_array, s2_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s2_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_s2_mocap[9,1])

# ---- Session 3
[s3_kick1_phase5_t_array, s3_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[0,1])
[s3_kick2_phase5_t_array, s3_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[1,1])
[s3_kick3_phase5_t_array, s3_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[2,1])
[s3_kick4_phase5_t_array, s3_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[3,1])
[s3_kick5_phase5_t_array, s3_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick5_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[4,1])
[s3_kick6_phase5_t_array, s3_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[5,1])
[s3_kick7_phase5_t_array, s3_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[6,1])
[s3_kick8_phase5_t_array, s3_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[7,1])
[s3_kick9_phase5_t_array, s3_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[8,1])
[s3_kick10_phase5_t_array, s3_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s3_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_s3_mocap[9,1])

# ---- Session 4
[s4_kick1_phase5_t_array, s4_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[0,1])
[s4_kick2_phase5_t_array, s4_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[1,1])
[s4_kick3_phase5_t_array, s4_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[2,1])
[s4_kick4_phase5_t_array, s4_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[3,1])
[s4_kick5_phase5_t_array, s4_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick5_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[4,1])
[s4_kick6_phase5_t_array, s4_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[5,1])
[s4_kick7_phase5_t_array, s4_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[6,1])
[s4_kick8_phase5_t_array, s4_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[7,1])
[s4_kick9_phase5_t_array, s4_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[8,1])
[s4_kick10_phase5_t_array, s4_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s4_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_s4_mocap[9,1])

# ---- Session 5
[s5_kick1_phase5_t_array, s5_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[0,1])
[s5_kick2_phase5_t_array, s5_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[1,1])
[s5_kick3_phase5_t_array, s5_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[2,1])
[s5_kick4_phase5_t_array, s5_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[3,1])
[s5_kick5_phase5_t_array, s5_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick5_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[4,1])
[s5_kick6_phase5_t_array, s5_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[5,1])
[s5_kick7_phase5_t_array, s5_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[6,1])
[s5_kick8_phase5_t_array, s5_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[7,1])
[s5_kick9_phase5_t_array, s5_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[8,1])
[s5_kick10_phase5_t_array, s5_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s5_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_s5_mocap[9,1])

# ---- Session 6
[s6_kick1_phase5_t_array, s6_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[0,1])
[s6_kick2_phase5_t_array, s6_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[1,1])
[s6_kick3_phase5_t_array, s6_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[2,1])
[s6_kick4_phase5_t_array, s6_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[3,1])
[s6_kick5_phase5_t_array, s6_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick5_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[4,1])
[s6_kick6_phase5_t_array, s6_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[5,1])
[s6_kick7_phase5_t_array, s6_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[6,1])
[s6_kick8_phase5_t_array, s6_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[7,1])
[s6_kick9_phase5_t_array, s6_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[8,1])
[s6_kick10_phase5_t_array, s6_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s6_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_s6_mocap[9,1])

# ---- Session 7
[s7_kick1_phase5_t_array, s7_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[0,1])
[s7_kick2_phase5_t_array, s7_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[1,1])
[s7_kick3_phase5_t_array, s7_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[2,1])
[s7_kick4_phase5_t_array, s7_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[3,1])
[s7_kick5_phase5_t_array, s7_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick5_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[4,1])
[s7_kick6_phase5_t_array, s7_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[5,1])
[s7_kick7_phase5_t_array, s7_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[6,1])
[s7_kick8_phase5_t_array, s7_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[7,1])
[s7_kick9_phase5_t_array, s7_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[8,1])
[s7_kick10_phase5_t_array, s7_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s7_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_s7_mocap[9,1])

# ---- Session 8
[s8_kick1_phase5_t_array, s8_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[0,1])
[s8_kick2_phase5_t_array, s8_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[1,1])
[s8_kick3_phase5_t_array, s8_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[2,1])
[s8_kick4_phase5_t_array, s8_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[3,1])
[s8_kick5_phase5_t_array, s8_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick5_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[4,1])
[s8_kick6_phase5_t_array, s8_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[5,1])
[s8_kick7_phase5_t_array, s8_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[6,1])
[s8_kick8_phase5_t_array, s8_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[7,1])
[s8_kick9_phase5_t_array, s8_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[8,1])
[s8_kick10_phase5_t_array, s8_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.s8_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_s8_mocap[9,1])

# ---- Expert
[expert_kick1_phase5_t_array, expert_kick1_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick1_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[0,1])
[expert_kick2_phase5_t_array, expert_kick2_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick2_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[1,1])
[expert_kick3_phase5_t_array, expert_kick3_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick3_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[2,1])
[expert_kick4_phase5_t_array, expert_kick4_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick4_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[3,1])
[expert_kick5_phase5_t_array, expert_kick5_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick5_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[4,1])
[expert_kick6_phase5_t_array, expert_kick6_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick6_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[5,1])
[expert_kick7_phase5_t_array, expert_kick7_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick7_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[6,1])
[expert_kick8_phase5_t_array, expert_kick8_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick8_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[7,1])
[expert_kick9_phase5_t_array, expert_kick9_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick9_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[8,1])
[expert_kick10_phase5_t_array, expert_kick10_phase5_t_array_scaled] = obtain_phase_time_array_variations(fp_on_off.expert_kick10_footon_idx, kick_frames.roundhousekicks_start_end_frames_expert_mocap[9,1])


''' Concatenate phase time array variations '''
# ---- Session 1
[s1_kick1_allphases_t_array, s1_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick1_phase1_t_array, s1_kick1_phase1_t_array_scaled,
                                                                                                    s1_kick1_phase2_t_array, s1_kick1_phase2_t_array_scaled,
                                                                                                    s1_kick1_phase3_t_array, s1_kick1_phase3_t_array_scaled,
                                                                                                    s1_kick1_phase4_t_array, s1_kick1_phase4_t_array_scaled,
                                                                                                    s1_kick1_phase5_t_array, s1_kick1_phase5_t_array_scaled)
[s1_kick2_allphases_t_array, s1_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick2_phase1_t_array, s1_kick2_phase1_t_array_scaled,
                                                                                                    s1_kick2_phase2_t_array, s1_kick2_phase2_t_array_scaled,
                                                                                                    s1_kick2_phase3_t_array, s1_kick2_phase3_t_array_scaled,
                                                                                                    s1_kick2_phase4_t_array, s1_kick2_phase4_t_array_scaled,
                                                                                                    s1_kick2_phase5_t_array, s1_kick2_phase5_t_array_scaled)
[s1_kick3_allphases_t_array, s1_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick3_phase1_t_array, s1_kick3_phase1_t_array_scaled,
                                                                                                    s1_kick3_phase2_t_array, s1_kick3_phase2_t_array_scaled,
                                                                                                    s1_kick3_phase3_t_array, s1_kick3_phase3_t_array_scaled,
                                                                                                    s1_kick3_phase4_t_array, s1_kick3_phase4_t_array_scaled,
                                                                                                    s1_kick3_phase5_t_array, s1_kick3_phase5_t_array_scaled)
[s1_kick4_allphases_t_array, s1_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick4_phase1_t_array, s1_kick4_phase1_t_array_scaled,
                                                                                                    s1_kick4_phase2_t_array, s1_kick4_phase2_t_array_scaled,
                                                                                                    s1_kick4_phase3_t_array, s1_kick4_phase3_t_array_scaled,
                                                                                                    s1_kick4_phase4_t_array, s1_kick4_phase4_t_array_scaled,
                                                                                                    s1_kick4_phase5_t_array, s1_kick4_phase5_t_array_scaled)
[s1_kick5_allphases_t_array, s1_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick5_phase1_t_array, s1_kick5_phase1_t_array_scaled,
                                                                                                    s1_kick5_phase2_t_array, s1_kick5_phase2_t_array_scaled,
                                                                                                    s1_kick5_phase3_t_array, s1_kick5_phase3_t_array_scaled,
                                                                                                    s1_kick5_phase4_t_array, s1_kick5_phase4_t_array_scaled,
                                                                                                    s1_kick5_phase5_t_array, s1_kick5_phase5_t_array_scaled)
[s1_kick6_allphases_t_array, s1_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick6_phase1_t_array, s1_kick6_phase1_t_array_scaled,
                                                                                                    s1_kick6_phase2_t_array, s1_kick6_phase2_t_array_scaled,
                                                                                                    s1_kick6_phase3_t_array, s1_kick6_phase3_t_array_scaled,
                                                                                                    s1_kick6_phase4_t_array, s1_kick6_phase4_t_array_scaled,
                                                                                                    s1_kick6_phase5_t_array, s1_kick6_phase5_t_array_scaled)
[s1_kick7_allphases_t_array, s1_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick7_phase1_t_array, s1_kick7_phase1_t_array_scaled,
                                                                                                    s1_kick7_phase2_t_array, s1_kick7_phase2_t_array_scaled,
                                                                                                    s1_kick7_phase3_t_array, s1_kick7_phase3_t_array_scaled,
                                                                                                    s1_kick7_phase4_t_array, s1_kick7_phase4_t_array_scaled,
                                                                                                    s1_kick7_phase5_t_array, s1_kick7_phase5_t_array_scaled)
[s1_kick8_allphases_t_array, s1_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick8_phase1_t_array, s1_kick8_phase1_t_array_scaled,
                                                                                                    s1_kick8_phase2_t_array, s1_kick8_phase2_t_array_scaled,
                                                                                                    s1_kick8_phase3_t_array, s1_kick8_phase3_t_array_scaled,
                                                                                                    s1_kick8_phase4_t_array, s1_kick8_phase4_t_array_scaled,
                                                                                                    s1_kick8_phase5_t_array, s1_kick8_phase5_t_array_scaled)
[s1_kick9_allphases_t_array, s1_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick9_phase1_t_array, s1_kick9_phase1_t_array_scaled,
                                                                                                    s1_kick9_phase2_t_array, s1_kick9_phase2_t_array_scaled,
                                                                                                    s1_kick9_phase3_t_array, s1_kick9_phase3_t_array_scaled,
                                                                                                    s1_kick9_phase4_t_array, s1_kick9_phase4_t_array_scaled,
                                                                                                    s1_kick9_phase5_t_array, s1_kick9_phase5_t_array_scaled)
[s1_kick10_allphases_t_array, s1_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(s1_kick10_phase1_t_array, s1_kick10_phase1_t_array_scaled,
                                                                                                    s1_kick10_phase2_t_array, s1_kick10_phase2_t_array_scaled,
                                                                                                    s1_kick10_phase3_t_array, s1_kick10_phase3_t_array_scaled,
                                                                                                    s1_kick10_phase4_t_array, s1_kick10_phase4_t_array_scaled,
                                                                                                    s1_kick10_phase5_t_array, s1_kick10_phase5_t_array_scaled)

# ---- Session 2
[s2_kick1_allphases_t_array, s2_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick1_phase1_t_array, s2_kick1_phase1_t_array_scaled,
                                                                                                    s2_kick1_phase2_t_array, s2_kick1_phase2_t_array_scaled,
                                                                                                    s2_kick1_phase3_t_array, s2_kick1_phase3_t_array_scaled,
                                                                                                    s2_kick1_phase4_t_array, s2_kick1_phase4_t_array_scaled,
                                                                                                    s2_kick1_phase5_t_array, s2_kick1_phase5_t_array_scaled)
[s2_kick2_allphases_t_array, s2_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick2_phase1_t_array, s2_kick2_phase1_t_array_scaled,
                                                                                                    s2_kick2_phase2_t_array, s2_kick2_phase2_t_array_scaled,
                                                                                                    s2_kick2_phase3_t_array, s2_kick2_phase3_t_array_scaled,
                                                                                                    s2_kick2_phase4_t_array, s2_kick2_phase4_t_array_scaled,
                                                                                                    s2_kick2_phase5_t_array, s2_kick2_phase5_t_array_scaled)
[s2_kick3_allphases_t_array, s2_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick3_phase1_t_array, s2_kick3_phase1_t_array_scaled,
                                                                                                    s2_kick3_phase2_t_array, s2_kick3_phase2_t_array_scaled,
                                                                                                    s2_kick3_phase3_t_array, s2_kick3_phase3_t_array_scaled,
                                                                                                    s2_kick3_phase4_t_array, s2_kick3_phase4_t_array_scaled,
                                                                                                    s2_kick3_phase5_t_array, s2_kick3_phase5_t_array_scaled)
[s2_kick4_allphases_t_array, s2_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick4_phase1_t_array, s2_kick4_phase1_t_array_scaled,
                                                                                                    s2_kick4_phase2_t_array, s2_kick4_phase2_t_array_scaled,
                                                                                                    s2_kick4_phase3_t_array, s2_kick4_phase3_t_array_scaled,
                                                                                                    s2_kick4_phase4_t_array, s2_kick4_phase4_t_array_scaled,
                                                                                                    s2_kick4_phase5_t_array, s2_kick4_phase5_t_array_scaled)
[s2_kick5_allphases_t_array, s2_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick5_phase1_t_array, s2_kick5_phase1_t_array_scaled,
                                                                                                    s2_kick5_phase2_t_array, s2_kick5_phase2_t_array_scaled,
                                                                                                    s2_kick5_phase3_t_array, s2_kick5_phase3_t_array_scaled,
                                                                                                    s2_kick5_phase4_t_array, s2_kick5_phase4_t_array_scaled,
                                                                                                    s2_kick5_phase5_t_array, s2_kick5_phase5_t_array_scaled)
[s2_kick6_allphases_t_array, s2_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick6_phase1_t_array, s2_kick6_phase1_t_array_scaled,
                                                                                                    s2_kick6_phase2_t_array, s2_kick6_phase2_t_array_scaled,
                                                                                                    s2_kick6_phase3_t_array, s2_kick6_phase3_t_array_scaled,
                                                                                                    s2_kick6_phase4_t_array, s2_kick6_phase4_t_array_scaled,
                                                                                                    s2_kick6_phase5_t_array, s2_kick6_phase5_t_array_scaled)
[s2_kick7_allphases_t_array, s2_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick7_phase1_t_array, s2_kick7_phase1_t_array_scaled,
                                                                                                    s2_kick7_phase2_t_array, s2_kick7_phase2_t_array_scaled,
                                                                                                    s2_kick7_phase3_t_array, s2_kick7_phase3_t_array_scaled,
                                                                                                    s2_kick7_phase4_t_array, s2_kick7_phase4_t_array_scaled,
                                                                                                    s2_kick7_phase5_t_array, s2_kick7_phase5_t_array_scaled)
[s2_kick8_allphases_t_array, s2_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick8_phase1_t_array, s2_kick8_phase1_t_array_scaled,
                                                                                                    s2_kick8_phase2_t_array, s2_kick8_phase2_t_array_scaled,
                                                                                                    s2_kick8_phase3_t_array, s2_kick8_phase3_t_array_scaled,
                                                                                                    s2_kick8_phase4_t_array, s2_kick8_phase4_t_array_scaled,
                                                                                                    s2_kick8_phase5_t_array, s2_kick8_phase5_t_array_scaled)
[s2_kick9_allphases_t_array, s2_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick9_phase1_t_array, s2_kick9_phase1_t_array_scaled,
                                                                                                    s2_kick9_phase2_t_array, s2_kick9_phase2_t_array_scaled,
                                                                                                    s2_kick9_phase3_t_array, s2_kick9_phase3_t_array_scaled,
                                                                                                    s2_kick9_phase4_t_array, s2_kick9_phase4_t_array_scaled,
                                                                                                    s2_kick9_phase5_t_array, s2_kick9_phase5_t_array_scaled)
[s2_kick10_allphases_t_array, s2_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(s2_kick10_phase1_t_array, s2_kick10_phase1_t_array_scaled,
                                                                                                    s2_kick10_phase2_t_array, s2_kick10_phase2_t_array_scaled,
                                                                                                    s2_kick10_phase3_t_array, s2_kick10_phase3_t_array_scaled,
                                                                                                    s2_kick10_phase4_t_array, s2_kick10_phase4_t_array_scaled,
                                                                                                    s2_kick10_phase5_t_array, s2_kick10_phase5_t_array_scaled)

# ---- Session 3
[s3_kick1_allphases_t_array, s3_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick1_phase1_t_array, s3_kick1_phase1_t_array_scaled,
                                                                                                    s3_kick1_phase2_t_array, s3_kick1_phase2_t_array_scaled,
                                                                                                    s3_kick1_phase3_t_array, s3_kick1_phase3_t_array_scaled,
                                                                                                    s3_kick1_phase4_t_array, s3_kick1_phase4_t_array_scaled,
                                                                                                    s3_kick1_phase5_t_array, s3_kick1_phase5_t_array_scaled)
[s3_kick2_allphases_t_array, s3_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick2_phase1_t_array, s3_kick2_phase1_t_array_scaled,
                                                                                                    s3_kick2_phase2_t_array, s3_kick2_phase2_t_array_scaled,
                                                                                                    s3_kick2_phase3_t_array, s3_kick2_phase3_t_array_scaled,
                                                                                                    s3_kick2_phase4_t_array, s3_kick2_phase4_t_array_scaled,
                                                                                                    s3_kick2_phase5_t_array, s3_kick2_phase5_t_array_scaled)
[s3_kick3_allphases_t_array, s3_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick3_phase1_t_array, s3_kick3_phase1_t_array_scaled,
                                                                                                    s3_kick3_phase2_t_array, s3_kick3_phase2_t_array_scaled,
                                                                                                    s3_kick3_phase3_t_array, s3_kick3_phase3_t_array_scaled,
                                                                                                    s3_kick3_phase4_t_array, s3_kick3_phase4_t_array_scaled,
                                                                                                    s3_kick3_phase5_t_array, s3_kick3_phase5_t_array_scaled)
[s3_kick4_allphases_t_array, s3_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick4_phase1_t_array, s3_kick4_phase1_t_array_scaled,
                                                                                                    s3_kick4_phase2_t_array, s3_kick4_phase2_t_array_scaled,
                                                                                                    s3_kick4_phase3_t_array, s3_kick4_phase3_t_array_scaled,
                                                                                                    s3_kick4_phase4_t_array, s3_kick4_phase4_t_array_scaled,
                                                                                                    s3_kick4_phase5_t_array, s3_kick4_phase5_t_array_scaled)
[s3_kick5_allphases_t_array, s3_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick5_phase1_t_array, s3_kick5_phase1_t_array_scaled,
                                                                                                    s3_kick5_phase2_t_array, s3_kick5_phase2_t_array_scaled,
                                                                                                    s3_kick5_phase3_t_array, s3_kick5_phase3_t_array_scaled,
                                                                                                    s3_kick5_phase4_t_array, s3_kick5_phase4_t_array_scaled,
                                                                                                    s3_kick5_phase5_t_array, s3_kick5_phase5_t_array_scaled)
[s3_kick6_allphases_t_array, s3_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick6_phase1_t_array, s3_kick6_phase1_t_array_scaled,
                                                                                                    s3_kick6_phase2_t_array, s3_kick6_phase2_t_array_scaled,
                                                                                                    s3_kick6_phase3_t_array, s3_kick6_phase3_t_array_scaled,
                                                                                                    s3_kick6_phase4_t_array, s3_kick6_phase4_t_array_scaled,
                                                                                                    s3_kick6_phase5_t_array, s3_kick6_phase5_t_array_scaled)
[s3_kick7_allphases_t_array, s3_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick7_phase1_t_array, s3_kick7_phase1_t_array_scaled,
                                                                                                    s3_kick7_phase2_t_array, s3_kick7_phase2_t_array_scaled,
                                                                                                    s3_kick7_phase3_t_array, s3_kick7_phase3_t_array_scaled,
                                                                                                    s3_kick7_phase4_t_array, s3_kick7_phase4_t_array_scaled,
                                                                                                    s3_kick7_phase5_t_array, s3_kick7_phase5_t_array_scaled)
[s3_kick8_allphases_t_array, s3_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick8_phase1_t_array, s3_kick8_phase1_t_array_scaled,
                                                                                                    s3_kick8_phase2_t_array, s3_kick8_phase2_t_array_scaled,
                                                                                                    s3_kick8_phase3_t_array, s3_kick8_phase3_t_array_scaled,
                                                                                                    s3_kick8_phase4_t_array, s3_kick8_phase4_t_array_scaled,
                                                                                                    s3_kick8_phase5_t_array, s3_kick8_phase5_t_array_scaled)
[s3_kick9_allphases_t_array, s3_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick9_phase1_t_array, s3_kick9_phase1_t_array_scaled,
                                                                                                    s3_kick9_phase2_t_array, s3_kick9_phase2_t_array_scaled,
                                                                                                    s3_kick9_phase3_t_array, s3_kick9_phase3_t_array_scaled,
                                                                                                    s3_kick9_phase4_t_array, s3_kick9_phase4_t_array_scaled,
                                                                                                    s3_kick9_phase5_t_array, s3_kick9_phase5_t_array_scaled)
[s3_kick10_allphases_t_array, s3_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(s3_kick10_phase1_t_array, s3_kick10_phase1_t_array_scaled,
                                                                                                    s3_kick10_phase2_t_array, s3_kick10_phase2_t_array_scaled,
                                                                                                    s3_kick10_phase3_t_array, s3_kick10_phase3_t_array_scaled,
                                                                                                    s3_kick10_phase4_t_array, s3_kick10_phase4_t_array_scaled,
                                                                                                    s3_kick10_phase5_t_array, s3_kick10_phase5_t_array_scaled)

# ---- Session 4
[s4_kick1_allphases_t_array, s4_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick1_phase1_t_array, s4_kick1_phase1_t_array_scaled,
                                                                                                    s4_kick1_phase2_t_array, s4_kick1_phase2_t_array_scaled,
                                                                                                    s4_kick1_phase3_t_array, s4_kick1_phase3_t_array_scaled,
                                                                                                    s4_kick1_phase4_t_array, s4_kick1_phase4_t_array_scaled,
                                                                                                    s4_kick1_phase5_t_array, s4_kick1_phase5_t_array_scaled)
[s4_kick2_allphases_t_array, s4_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick2_phase1_t_array, s4_kick2_phase1_t_array_scaled,
                                                                                                    s4_kick2_phase2_t_array, s4_kick2_phase2_t_array_scaled,
                                                                                                    s4_kick2_phase3_t_array, s4_kick2_phase3_t_array_scaled,
                                                                                                    s4_kick2_phase4_t_array, s4_kick2_phase4_t_array_scaled,
                                                                                                    s4_kick2_phase5_t_array, s4_kick2_phase5_t_array_scaled)
[s4_kick3_allphases_t_array, s4_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick3_phase1_t_array, s4_kick3_phase1_t_array_scaled,
                                                                                                    s4_kick3_phase2_t_array, s4_kick3_phase2_t_array_scaled,
                                                                                                    s4_kick3_phase3_t_array, s4_kick3_phase3_t_array_scaled,
                                                                                                    s4_kick3_phase4_t_array, s4_kick3_phase4_t_array_scaled,
                                                                                                    s4_kick3_phase5_t_array, s4_kick3_phase5_t_array_scaled)
[s4_kick4_allphases_t_array, s4_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick4_phase1_t_array, s4_kick4_phase1_t_array_scaled,
                                                                                                    s4_kick4_phase2_t_array, s4_kick4_phase2_t_array_scaled,
                                                                                                    s4_kick4_phase3_t_array, s4_kick4_phase3_t_array_scaled,
                                                                                                    s4_kick4_phase4_t_array, s4_kick4_phase4_t_array_scaled,
                                                                                                    s4_kick4_phase5_t_array, s4_kick4_phase5_t_array_scaled)
[s4_kick5_allphases_t_array, s4_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick5_phase1_t_array, s4_kick5_phase1_t_array_scaled,
                                                                                                    s4_kick5_phase2_t_array, s4_kick5_phase2_t_array_scaled,
                                                                                                    s4_kick5_phase3_t_array, s4_kick5_phase3_t_array_scaled,
                                                                                                    s4_kick5_phase4_t_array, s4_kick5_phase4_t_array_scaled,
                                                                                                    s4_kick5_phase5_t_array, s4_kick5_phase5_t_array_scaled)
[s4_kick6_allphases_t_array, s4_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick6_phase1_t_array, s4_kick6_phase1_t_array_scaled,
                                                                                                    s4_kick6_phase2_t_array, s4_kick6_phase2_t_array_scaled,
                                                                                                    s4_kick6_phase3_t_array, s4_kick6_phase3_t_array_scaled,
                                                                                                    s4_kick6_phase4_t_array, s4_kick6_phase4_t_array_scaled,
                                                                                                    s4_kick6_phase5_t_array, s4_kick6_phase5_t_array_scaled)
[s4_kick7_allphases_t_array, s4_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick7_phase1_t_array, s4_kick7_phase1_t_array_scaled,
                                                                                                    s4_kick7_phase2_t_array, s4_kick7_phase2_t_array_scaled,
                                                                                                    s4_kick7_phase3_t_array, s4_kick7_phase3_t_array_scaled,
                                                                                                    s4_kick7_phase4_t_array, s4_kick7_phase4_t_array_scaled,
                                                                                                    s4_kick7_phase5_t_array, s4_kick7_phase5_t_array_scaled)
[s4_kick8_allphases_t_array, s4_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick8_phase1_t_array, s4_kick8_phase1_t_array_scaled,
                                                                                                    s4_kick8_phase2_t_array, s4_kick8_phase2_t_array_scaled,
                                                                                                    s4_kick8_phase3_t_array, s4_kick8_phase3_t_array_scaled,
                                                                                                    s4_kick8_phase4_t_array, s4_kick8_phase4_t_array_scaled,
                                                                                                    s4_kick8_phase5_t_array, s4_kick8_phase5_t_array_scaled)
[s4_kick9_allphases_t_array, s4_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick9_phase1_t_array, s4_kick9_phase1_t_array_scaled,
                                                                                                    s4_kick9_phase2_t_array, s4_kick9_phase2_t_array_scaled,
                                                                                                    s4_kick9_phase3_t_array, s4_kick9_phase3_t_array_scaled,
                                                                                                    s4_kick9_phase4_t_array, s4_kick9_phase4_t_array_scaled,
                                                                                                    s4_kick9_phase5_t_array, s4_kick9_phase5_t_array_scaled)
[s4_kick10_allphases_t_array, s4_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(s4_kick10_phase1_t_array, s4_kick10_phase1_t_array_scaled,
                                                                                                    s4_kick10_phase2_t_array, s4_kick10_phase2_t_array_scaled,
                                                                                                    s4_kick10_phase3_t_array, s4_kick10_phase3_t_array_scaled,
                                                                                                    s4_kick10_phase4_t_array, s4_kick10_phase4_t_array_scaled,
                                                                                                    s4_kick10_phase5_t_array, s4_kick10_phase5_t_array_scaled)

# ---- Session 5
[s5_kick1_allphases_t_array, s5_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick1_phase1_t_array, s5_kick1_phase1_t_array_scaled,
                                                                                                    s5_kick1_phase2_t_array, s5_kick1_phase2_t_array_scaled,
                                                                                                    s5_kick1_phase3_t_array, s5_kick1_phase3_t_array_scaled,
                                                                                                    s5_kick1_phase4_t_array, s5_kick1_phase4_t_array_scaled,
                                                                                                    s5_kick1_phase5_t_array, s5_kick1_phase5_t_array_scaled)
[s5_kick2_allphases_t_array, s5_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick2_phase1_t_array, s5_kick2_phase1_t_array_scaled,
                                                                                                    s5_kick2_phase2_t_array, s5_kick2_phase2_t_array_scaled,
                                                                                                    s5_kick2_phase3_t_array, s5_kick2_phase3_t_array_scaled,
                                                                                                    s5_kick2_phase4_t_array, s5_kick2_phase4_t_array_scaled,
                                                                                                    s5_kick2_phase5_t_array, s5_kick2_phase5_t_array_scaled)
[s5_kick3_allphases_t_array, s5_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick3_phase1_t_array, s5_kick3_phase1_t_array_scaled,
                                                                                                    s5_kick3_phase2_t_array, s5_kick3_phase2_t_array_scaled,
                                                                                                    s5_kick3_phase3_t_array, s5_kick3_phase3_t_array_scaled,
                                                                                                    s5_kick3_phase4_t_array, s5_kick3_phase4_t_array_scaled,
                                                                                                    s5_kick3_phase5_t_array, s5_kick3_phase5_t_array_scaled)
[s5_kick4_allphases_t_array, s5_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick4_phase1_t_array, s5_kick4_phase1_t_array_scaled,
                                                                                                    s5_kick4_phase2_t_array, s5_kick4_phase2_t_array_scaled,
                                                                                                    s5_kick4_phase3_t_array, s5_kick4_phase3_t_array_scaled,
                                                                                                    s5_kick4_phase4_t_array, s5_kick4_phase4_t_array_scaled,
                                                                                                    s5_kick4_phase5_t_array, s5_kick4_phase5_t_array_scaled)
[s5_kick5_allphases_t_array, s5_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick5_phase1_t_array, s5_kick5_phase1_t_array_scaled,
                                                                                                    s5_kick5_phase2_t_array, s5_kick5_phase2_t_array_scaled,
                                                                                                    s5_kick5_phase3_t_array, s5_kick5_phase3_t_array_scaled,
                                                                                                    s5_kick5_phase4_t_array, s5_kick5_phase4_t_array_scaled,
                                                                                                    s5_kick5_phase5_t_array, s5_kick5_phase5_t_array_scaled)
[s5_kick6_allphases_t_array, s5_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick6_phase1_t_array, s5_kick6_phase1_t_array_scaled,
                                                                                                    s5_kick6_phase2_t_array, s5_kick6_phase2_t_array_scaled,
                                                                                                    s5_kick6_phase3_t_array, s5_kick6_phase3_t_array_scaled,
                                                                                                    s5_kick6_phase4_t_array, s5_kick6_phase4_t_array_scaled,
                                                                                                    s5_kick6_phase5_t_array, s5_kick6_phase5_t_array_scaled)
[s5_kick7_allphases_t_array, s5_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick7_phase1_t_array, s5_kick7_phase1_t_array_scaled,
                                                                                                    s5_kick7_phase2_t_array, s5_kick7_phase2_t_array_scaled,
                                                                                                    s5_kick7_phase3_t_array, s5_kick7_phase3_t_array_scaled,
                                                                                                    s5_kick7_phase4_t_array, s5_kick7_phase4_t_array_scaled,
                                                                                                    s5_kick7_phase5_t_array, s5_kick7_phase5_t_array_scaled)
[s5_kick8_allphases_t_array, s5_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick8_phase1_t_array, s5_kick8_phase1_t_array_scaled,
                                                                                                    s5_kick8_phase2_t_array, s5_kick8_phase2_t_array_scaled,
                                                                                                    s5_kick8_phase3_t_array, s5_kick8_phase3_t_array_scaled,
                                                                                                    s5_kick8_phase4_t_array, s5_kick8_phase4_t_array_scaled,
                                                                                                    s5_kick8_phase5_t_array, s5_kick8_phase5_t_array_scaled)
[s5_kick9_allphases_t_array, s5_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick9_phase1_t_array, s5_kick9_phase1_t_array_scaled,
                                                                                                    s5_kick9_phase2_t_array, s5_kick9_phase2_t_array_scaled,
                                                                                                    s5_kick9_phase3_t_array, s5_kick9_phase3_t_array_scaled,
                                                                                                    s5_kick9_phase4_t_array, s5_kick9_phase4_t_array_scaled,
                                                                                                    s5_kick9_phase5_t_array, s5_kick9_phase5_t_array_scaled)
[s5_kick10_allphases_t_array, s5_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(s5_kick10_phase1_t_array, s5_kick10_phase1_t_array_scaled,
                                                                                                    s5_kick10_phase2_t_array, s5_kick10_phase2_t_array_scaled,
                                                                                                    s5_kick10_phase3_t_array, s5_kick10_phase3_t_array_scaled,
                                                                                                    s5_kick10_phase4_t_array, s5_kick10_phase4_t_array_scaled,
                                                                                                    s5_kick10_phase5_t_array, s5_kick10_phase5_t_array_scaled)

# ---- Session 6
[s6_kick1_allphases_t_array, s6_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick1_phase1_t_array, s6_kick1_phase1_t_array_scaled,
                                                                                                    s6_kick1_phase2_t_array, s6_kick1_phase2_t_array_scaled,
                                                                                                    s6_kick1_phase3_t_array, s6_kick1_phase3_t_array_scaled,
                                                                                                    s6_kick1_phase4_t_array, s6_kick1_phase4_t_array_scaled,
                                                                                                    s6_kick1_phase5_t_array, s6_kick1_phase5_t_array_scaled)
[s6_kick2_allphases_t_array, s6_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick2_phase1_t_array, s6_kick2_phase1_t_array_scaled,
                                                                                                    s6_kick2_phase2_t_array, s6_kick2_phase2_t_array_scaled,
                                                                                                    s6_kick2_phase3_t_array, s6_kick2_phase3_t_array_scaled,
                                                                                                    s6_kick2_phase4_t_array, s6_kick2_phase4_t_array_scaled,
                                                                                                    s6_kick2_phase5_t_array, s6_kick2_phase5_t_array_scaled)
[s6_kick3_allphases_t_array, s6_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick3_phase1_t_array, s6_kick3_phase1_t_array_scaled,
                                                                                                    s6_kick3_phase2_t_array, s6_kick3_phase2_t_array_scaled,
                                                                                                    s6_kick3_phase3_t_array, s6_kick3_phase3_t_array_scaled,
                                                                                                    s6_kick3_phase4_t_array, s6_kick3_phase4_t_array_scaled,
                                                                                                    s6_kick3_phase5_t_array, s6_kick3_phase5_t_array_scaled)
[s6_kick4_allphases_t_array, s6_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick4_phase1_t_array, s6_kick4_phase1_t_array_scaled,
                                                                                                    s6_kick4_phase2_t_array, s6_kick4_phase2_t_array_scaled,
                                                                                                    s6_kick4_phase3_t_array, s6_kick4_phase3_t_array_scaled,
                                                                                                    s6_kick4_phase4_t_array, s6_kick4_phase4_t_array_scaled,
                                                                                                    s6_kick4_phase5_t_array, s6_kick4_phase5_t_array_scaled)
[s6_kick5_allphases_t_array, s6_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick5_phase1_t_array, s6_kick5_phase1_t_array_scaled,
                                                                                                    s6_kick5_phase2_t_array, s6_kick5_phase2_t_array_scaled,
                                                                                                    s6_kick5_phase3_t_array, s6_kick5_phase3_t_array_scaled,
                                                                                                    s6_kick5_phase4_t_array, s6_kick5_phase4_t_array_scaled,
                                                                                                    s6_kick5_phase5_t_array, s6_kick5_phase5_t_array_scaled)
[s6_kick6_allphases_t_array, s6_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick6_phase1_t_array, s6_kick6_phase1_t_array_scaled,
                                                                                                    s6_kick6_phase2_t_array, s6_kick6_phase2_t_array_scaled,
                                                                                                    s6_kick6_phase3_t_array, s6_kick6_phase3_t_array_scaled,
                                                                                                    s6_kick6_phase4_t_array, s6_kick6_phase4_t_array_scaled,
                                                                                                    s6_kick6_phase5_t_array, s6_kick6_phase5_t_array_scaled)
[s6_kick7_allphases_t_array, s6_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick7_phase1_t_array, s6_kick7_phase1_t_array_scaled,
                                                                                                    s6_kick7_phase2_t_array, s6_kick7_phase2_t_array_scaled,
                                                                                                    s6_kick7_phase3_t_array, s6_kick7_phase3_t_array_scaled,
                                                                                                    s6_kick7_phase4_t_array, s6_kick7_phase4_t_array_scaled,
                                                                                                    s6_kick7_phase5_t_array, s6_kick7_phase5_t_array_scaled)
[s6_kick8_allphases_t_array, s6_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick8_phase1_t_array, s6_kick8_phase1_t_array_scaled,
                                                                                                    s6_kick8_phase2_t_array, s6_kick8_phase2_t_array_scaled,
                                                                                                    s6_kick8_phase3_t_array, s6_kick8_phase3_t_array_scaled,
                                                                                                    s6_kick8_phase4_t_array, s6_kick8_phase4_t_array_scaled,
                                                                                                    s6_kick8_phase5_t_array, s6_kick8_phase5_t_array_scaled)
[s6_kick9_allphases_t_array, s6_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick9_phase1_t_array, s6_kick9_phase1_t_array_scaled,
                                                                                                    s6_kick9_phase2_t_array, s6_kick9_phase2_t_array_scaled,
                                                                                                    s6_kick9_phase3_t_array, s6_kick9_phase3_t_array_scaled,
                                                                                                    s6_kick9_phase4_t_array, s6_kick9_phase4_t_array_scaled,
                                                                                                    s6_kick9_phase5_t_array, s6_kick9_phase5_t_array_scaled)
[s6_kick10_allphases_t_array, s6_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(s6_kick10_phase1_t_array, s6_kick10_phase1_t_array_scaled,
                                                                                                    s6_kick10_phase2_t_array, s6_kick10_phase2_t_array_scaled,
                                                                                                    s6_kick10_phase3_t_array, s6_kick10_phase3_t_array_scaled,
                                                                                                    s6_kick10_phase4_t_array, s6_kick10_phase4_t_array_scaled,
                                                                                                    s6_kick10_phase5_t_array, s6_kick10_phase5_t_array_scaled)

# ---- Session 7
[s7_kick1_allphases_t_array, s7_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick1_phase1_t_array, s7_kick1_phase1_t_array_scaled,
                                                                                                    s7_kick1_phase2_t_array, s7_kick1_phase2_t_array_scaled,
                                                                                                    s7_kick1_phase3_t_array, s7_kick1_phase3_t_array_scaled,
                                                                                                    s7_kick1_phase4_t_array, s7_kick1_phase4_t_array_scaled,
                                                                                                    s7_kick1_phase5_t_array, s7_kick1_phase5_t_array_scaled)
[s7_kick2_allphases_t_array, s7_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick2_phase1_t_array, s7_kick2_phase1_t_array_scaled,
                                                                                                    s7_kick2_phase2_t_array, s7_kick2_phase2_t_array_scaled,
                                                                                                    s7_kick2_phase3_t_array, s7_kick2_phase3_t_array_scaled,
                                                                                                    s7_kick2_phase4_t_array, s7_kick2_phase4_t_array_scaled,
                                                                                                    s7_kick2_phase5_t_array, s7_kick2_phase5_t_array_scaled)
[s7_kick3_allphases_t_array, s7_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick3_phase1_t_array, s7_kick3_phase1_t_array_scaled,
                                                                                                    s7_kick3_phase2_t_array, s7_kick3_phase2_t_array_scaled,
                                                                                                    s7_kick3_phase3_t_array, s7_kick3_phase3_t_array_scaled,
                                                                                                    s7_kick3_phase4_t_array, s7_kick3_phase4_t_array_scaled,
                                                                                                    s7_kick3_phase5_t_array, s7_kick3_phase5_t_array_scaled)
[s7_kick4_allphases_t_array, s7_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick4_phase1_t_array, s7_kick4_phase1_t_array_scaled,
                                                                                                    s7_kick4_phase2_t_array, s7_kick4_phase2_t_array_scaled,
                                                                                                    s7_kick4_phase3_t_array, s7_kick4_phase3_t_array_scaled,
                                                                                                    s7_kick4_phase4_t_array, s7_kick4_phase4_t_array_scaled,
                                                                                                    s7_kick4_phase5_t_array, s7_kick4_phase5_t_array_scaled)
[s7_kick5_allphases_t_array, s7_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick5_phase1_t_array, s7_kick5_phase1_t_array_scaled,
                                                                                                    s7_kick5_phase2_t_array, s7_kick5_phase2_t_array_scaled,
                                                                                                    s7_kick5_phase3_t_array, s7_kick5_phase3_t_array_scaled,
                                                                                                    s7_kick5_phase4_t_array, s7_kick5_phase4_t_array_scaled,
                                                                                                    s7_kick5_phase5_t_array, s7_kick5_phase5_t_array_scaled)
[s7_kick6_allphases_t_array, s7_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick6_phase1_t_array, s7_kick6_phase1_t_array_scaled,
                                                                                                    s7_kick6_phase2_t_array, s7_kick6_phase2_t_array_scaled,
                                                                                                    s7_kick6_phase3_t_array, s7_kick6_phase3_t_array_scaled,
                                                                                                    s7_kick6_phase4_t_array, s7_kick6_phase4_t_array_scaled,
                                                                                                    s7_kick6_phase5_t_array, s7_kick6_phase5_t_array_scaled)
[s7_kick7_allphases_t_array, s7_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick7_phase1_t_array, s7_kick7_phase1_t_array_scaled,
                                                                                                    s7_kick7_phase2_t_array, s7_kick7_phase2_t_array_scaled,
                                                                                                    s7_kick7_phase3_t_array, s7_kick7_phase3_t_array_scaled,
                                                                                                    s7_kick7_phase4_t_array, s7_kick7_phase4_t_array_scaled,
                                                                                                    s7_kick7_phase5_t_array, s7_kick7_phase5_t_array_scaled)
[s7_kick8_allphases_t_array, s7_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick8_phase1_t_array, s7_kick8_phase1_t_array_scaled,
                                                                                                    s7_kick8_phase2_t_array, s7_kick8_phase2_t_array_scaled,
                                                                                                    s7_kick8_phase3_t_array, s7_kick8_phase3_t_array_scaled,
                                                                                                    s7_kick8_phase4_t_array, s7_kick8_phase4_t_array_scaled,
                                                                                                    s7_kick8_phase5_t_array, s7_kick8_phase5_t_array_scaled)
[s7_kick9_allphases_t_array, s7_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick9_phase1_t_array, s7_kick9_phase1_t_array_scaled,
                                                                                                    s7_kick9_phase2_t_array, s7_kick9_phase2_t_array_scaled,
                                                                                                    s7_kick9_phase3_t_array, s7_kick9_phase3_t_array_scaled,
                                                                                                    s7_kick9_phase4_t_array, s7_kick9_phase4_t_array_scaled,
                                                                                                    s7_kick9_phase5_t_array, s7_kick9_phase5_t_array_scaled)
[s7_kick10_allphases_t_array, s7_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(s7_kick10_phase1_t_array, s7_kick10_phase1_t_array_scaled,
                                                                                                    s7_kick10_phase2_t_array, s7_kick10_phase2_t_array_scaled,
                                                                                                    s7_kick10_phase3_t_array, s7_kick10_phase3_t_array_scaled,
                                                                                                    s7_kick10_phase4_t_array, s7_kick10_phase4_t_array_scaled,
                                                                                                    s7_kick10_phase5_t_array, s7_kick10_phase5_t_array_scaled)

# ---- Session 8
[s8_kick1_allphases_t_array, s8_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick1_phase1_t_array, s8_kick1_phase1_t_array_scaled,
                                                                                                    s8_kick1_phase2_t_array, s8_kick1_phase2_t_array_scaled,
                                                                                                    s8_kick1_phase3_t_array, s8_kick1_phase3_t_array_scaled,
                                                                                                    s8_kick1_phase4_t_array, s8_kick1_phase4_t_array_scaled,
                                                                                                    s8_kick1_phase5_t_array, s8_kick1_phase5_t_array_scaled)
[s8_kick2_allphases_t_array, s8_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick2_phase1_t_array, s8_kick2_phase1_t_array_scaled,
                                                                                                    s8_kick2_phase2_t_array, s8_kick2_phase2_t_array_scaled,
                                                                                                    s8_kick2_phase3_t_array, s8_kick2_phase3_t_array_scaled,
                                                                                                    s8_kick2_phase4_t_array, s8_kick2_phase4_t_array_scaled,
                                                                                                    s8_kick2_phase5_t_array, s8_kick2_phase5_t_array_scaled)
[s8_kick3_allphases_t_array, s8_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick3_phase1_t_array, s8_kick3_phase1_t_array_scaled,
                                                                                                    s8_kick3_phase2_t_array, s8_kick3_phase2_t_array_scaled,
                                                                                                    s8_kick3_phase3_t_array, s8_kick3_phase3_t_array_scaled,
                                                                                                    s8_kick3_phase4_t_array, s8_kick3_phase4_t_array_scaled,
                                                                                                    s8_kick3_phase5_t_array, s8_kick3_phase5_t_array_scaled)
[s8_kick4_allphases_t_array, s8_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick4_phase1_t_array, s8_kick4_phase1_t_array_scaled,
                                                                                                    s8_kick4_phase2_t_array, s8_kick4_phase2_t_array_scaled,
                                                                                                    s8_kick4_phase3_t_array, s8_kick4_phase3_t_array_scaled,
                                                                                                    s8_kick4_phase4_t_array, s8_kick4_phase4_t_array_scaled,
                                                                                                    s8_kick4_phase5_t_array, s8_kick4_phase5_t_array_scaled)
[s8_kick5_allphases_t_array, s8_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick5_phase1_t_array, s8_kick5_phase1_t_array_scaled,
                                                                                                    s8_kick5_phase2_t_array, s8_kick5_phase2_t_array_scaled,
                                                                                                    s8_kick5_phase3_t_array, s8_kick5_phase3_t_array_scaled,
                                                                                                    s8_kick5_phase4_t_array, s8_kick5_phase4_t_array_scaled,
                                                                                                    s8_kick5_phase5_t_array, s8_kick5_phase5_t_array_scaled)
[s8_kick6_allphases_t_array, s8_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick6_phase1_t_array, s8_kick6_phase1_t_array_scaled,
                                                                                                    s8_kick6_phase2_t_array, s8_kick6_phase2_t_array_scaled,
                                                                                                    s8_kick6_phase3_t_array, s8_kick6_phase3_t_array_scaled,
                                                                                                    s8_kick6_phase4_t_array, s8_kick6_phase4_t_array_scaled,
                                                                                                    s8_kick6_phase5_t_array, s8_kick6_phase5_t_array_scaled)
[s8_kick7_allphases_t_array, s8_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick7_phase1_t_array, s8_kick7_phase1_t_array_scaled,
                                                                                                    s8_kick7_phase2_t_array, s8_kick7_phase2_t_array_scaled,
                                                                                                    s8_kick7_phase3_t_array, s8_kick7_phase3_t_array_scaled,
                                                                                                    s8_kick7_phase4_t_array, s8_kick7_phase4_t_array_scaled,
                                                                                                    s8_kick7_phase5_t_array, s8_kick7_phase5_t_array_scaled)
[s8_kick8_allphases_t_array, s8_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick8_phase1_t_array, s8_kick8_phase1_t_array_scaled,
                                                                                                    s8_kick8_phase2_t_array, s8_kick8_phase2_t_array_scaled,
                                                                                                    s8_kick8_phase3_t_array, s8_kick8_phase3_t_array_scaled,
                                                                                                    s8_kick8_phase4_t_array, s8_kick8_phase4_t_array_scaled,
                                                                                                    s8_kick8_phase5_t_array, s8_kick8_phase5_t_array_scaled)
[s8_kick9_allphases_t_array, s8_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick9_phase1_t_array, s8_kick9_phase1_t_array_scaled,
                                                                                                    s8_kick9_phase2_t_array, s8_kick9_phase2_t_array_scaled,
                                                                                                    s8_kick9_phase3_t_array, s8_kick9_phase3_t_array_scaled,
                                                                                                    s8_kick9_phase4_t_array, s8_kick9_phase4_t_array_scaled,
                                                                                                    s8_kick9_phase5_t_array, s8_kick9_phase5_t_array_scaled)
[s8_kick10_allphases_t_array, s8_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(s8_kick10_phase1_t_array, s8_kick10_phase1_t_array_scaled,
                                                                                                    s8_kick10_phase2_t_array, s8_kick10_phase2_t_array_scaled,
                                                                                                    s8_kick10_phase3_t_array, s8_kick10_phase3_t_array_scaled,
                                                                                                    s8_kick10_phase4_t_array, s8_kick10_phase4_t_array_scaled,
                                                                                                    s8_kick10_phase5_t_array, s8_kick10_phase5_t_array_scaled)

# ---- Expert
[expert_kick1_allphases_t_array, expert_kick1_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick1_phase1_t_array, expert_kick1_phase1_t_array_scaled,
                                                                                                    expert_kick1_phase2_t_array, expert_kick1_phase2_t_array_scaled,
                                                                                                    expert_kick1_phase3_t_array, expert_kick1_phase3_t_array_scaled,
                                                                                                    expert_kick1_phase4_t_array, expert_kick1_phase4_t_array_scaled,
                                                                                                    expert_kick1_phase5_t_array, expert_kick1_phase5_t_array_scaled)
[expert_kick2_allphases_t_array, expert_kick2_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick2_phase1_t_array, expert_kick2_phase1_t_array_scaled,
                                                                                                    expert_kick2_phase2_t_array, expert_kick2_phase2_t_array_scaled,
                                                                                                    expert_kick2_phase3_t_array, expert_kick2_phase3_t_array_scaled,
                                                                                                    expert_kick2_phase4_t_array, expert_kick2_phase4_t_array_scaled,
                                                                                                    expert_kick2_phase5_t_array, expert_kick2_phase5_t_array_scaled)
[expert_kick3_allphases_t_array, expert_kick3_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick3_phase1_t_array, expert_kick3_phase1_t_array_scaled,
                                                                                                    expert_kick3_phase2_t_array, expert_kick3_phase2_t_array_scaled,
                                                                                                    expert_kick3_phase3_t_array, expert_kick3_phase3_t_array_scaled,
                                                                                                    expert_kick3_phase4_t_array, expert_kick3_phase4_t_array_scaled,
                                                                                                    expert_kick3_phase5_t_array, expert_kick3_phase5_t_array_scaled)
[expert_kick4_allphases_t_array, expert_kick4_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick4_phase1_t_array, expert_kick4_phase1_t_array_scaled,
                                                                                                    expert_kick4_phase2_t_array, expert_kick4_phase2_t_array_scaled,
                                                                                                    expert_kick4_phase3_t_array, expert_kick4_phase3_t_array_scaled,
                                                                                                    expert_kick4_phase4_t_array, expert_kick4_phase4_t_array_scaled,
                                                                                                    expert_kick4_phase5_t_array, expert_kick4_phase5_t_array_scaled)
[expert_kick5_allphases_t_array, expert_kick5_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick5_phase1_t_array, expert_kick5_phase1_t_array_scaled,
                                                                                                    expert_kick5_phase2_t_array, expert_kick5_phase2_t_array_scaled,
                                                                                                    expert_kick5_phase3_t_array, expert_kick5_phase3_t_array_scaled,
                                                                                                    expert_kick5_phase4_t_array, expert_kick5_phase4_t_array_scaled,
                                                                                                    expert_kick5_phase5_t_array, expert_kick5_phase5_t_array_scaled)
[expert_kick6_allphases_t_array, expert_kick6_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick6_phase1_t_array, expert_kick6_phase1_t_array_scaled,
                                                                                                    expert_kick6_phase2_t_array, expert_kick6_phase2_t_array_scaled,
                                                                                                    expert_kick6_phase3_t_array, expert_kick6_phase3_t_array_scaled,
                                                                                                    expert_kick6_phase4_t_array, expert_kick6_phase4_t_array_scaled,
                                                                                                    expert_kick6_phase5_t_array, expert_kick6_phase5_t_array_scaled)
[expert_kick7_allphases_t_array, expert_kick7_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick7_phase1_t_array, expert_kick7_phase1_t_array_scaled,
                                                                                                    expert_kick7_phase2_t_array, expert_kick7_phase2_t_array_scaled,
                                                                                                    expert_kick7_phase3_t_array, expert_kick7_phase3_t_array_scaled,
                                                                                                    expert_kick7_phase4_t_array, expert_kick7_phase4_t_array_scaled,
                                                                                                    expert_kick7_phase5_t_array, expert_kick7_phase5_t_array_scaled)
[expert_kick8_allphases_t_array, expert_kick8_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick8_phase1_t_array, expert_kick8_phase1_t_array_scaled,
                                                                                                    expert_kick8_phase2_t_array, expert_kick8_phase2_t_array_scaled,
                                                                                                    expert_kick8_phase3_t_array, expert_kick8_phase3_t_array_scaled,
                                                                                                    expert_kick8_phase4_t_array, expert_kick8_phase4_t_array_scaled,
                                                                                                    expert_kick8_phase5_t_array, expert_kick8_phase5_t_array_scaled)
[expert_kick9_allphases_t_array, expert_kick9_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick9_phase1_t_array, expert_kick9_phase1_t_array_scaled,
                                                                                                    expert_kick9_phase2_t_array, expert_kick9_phase2_t_array_scaled,
                                                                                                    expert_kick9_phase3_t_array, expert_kick9_phase3_t_array_scaled,
                                                                                                    expert_kick9_phase4_t_array, expert_kick9_phase4_t_array_scaled,
                                                                                                    expert_kick9_phase5_t_array, expert_kick9_phase5_t_array_scaled)
[expert_kick10_allphases_t_array, expert_kick10_allphases_t_array_scaled] = concatenate_time_array_variations(expert_kick10_phase1_t_array, expert_kick10_phase1_t_array_scaled,
                                                                                                    expert_kick10_phase2_t_array, expert_kick10_phase2_t_array_scaled,
                                                                                                    expert_kick10_phase3_t_array, expert_kick10_phase3_t_array_scaled,
                                                                                                    expert_kick10_phase4_t_array, expert_kick10_phase4_t_array_scaled,
                                                                                                    expert_kick10_phase5_t_array, expert_kick10_phase5_t_array_scaled)