"""
Pelvis Rotation Analysis (Z-axis)
==================================
Calculates pelvis axial rotation from left/right hip joint positions,
averaged per subject, and compared across two user-defined groups.
 
Data format expected:
    - Rows    : integer frame index
    - Columns : MultiIndex (subject, trial, variable)
    - Variables used:
        L_HIP_POSITION_X, L_HIP_POSITION_Y
        R_HIP_POSITION_X, R_HIP_POSITION_Y
 
Rotation is computed as the angle of the pelvis medio-lateral axis
(vector from R_HIP to L_HIP) projected onto the XY plane,
relative to frame 0 (= 0°).
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import BigLoader
# =============================================================================
# USER CONFIGURATION — fill in your subject labels and data path
# =============================================================================
 
GROUP_1_LABEL = "Group 1"
GROUP_2_LABEL = "Group 2"
 
GROUP_1_SUBJECTS = [
    "E1","E2","E3"   # <-- replace with your subject labels
]
 
GROUP_2_SUBJECTS = [
    "N1","N2","N3","N4"   # <-- replace with your subject labels
]
 
# =============================================================================
# LOAD DATA  — adjust this block to match how your file is stored
# =============================================================================
 
df = BigLoader.loadallspecified(GROUP_1_SUBJECTS + GROUP_2_SUBJECTS, "uppercut")
# df = pd.read_csv(DATA_PATH, header=[0,1,2], index_col=0)   # if CSV with MultiIndex
df.columns.names = ["subject", "trial", "variable"]

# =============================================================================
# CORE CALCULATION
# =============================================================================
 
def compute_pelvis_rotation(df: pd.DataFrame, subject: str) -> pd.Series:
    """
    Compute pelvis axial rotation (Z-axis) for one subject.
 
    df[(subj, trial, var)] may return a 2D DataFrame when multiple repetitions
    share the same (subject, trial, variable) label — each column is one rep.
    Rotation is computed per rep then averaged across all reps and trials.
 
    Returns:
        pd.Series of mean rotation angle (degrees), length = n_frames.
    """
    # Sort MultiIndex once to avoid PerformanceWarning
    if not df.columns.is_monotonic_increasing:
        df.sort_index(axis=1, inplace=True)
 
    col_idx = df.columns
    mask = col_idx.get_level_values("subject") == subject
    trials = col_idx[mask].get_level_values("trial").unique()
 
    all_rotations = []
 
    for trial in trials:
        def get_2d(var):
            result = df[(subject, trial, var)]
            if isinstance(result, pd.Series):
                result = result.to_frame()
            return result.values  # shape: (n_frames, n_reps)
 
        try:
            lx = get_2d("L_HIP_POSITION_X")
            ly = get_2d("L_HIP_POSITION_Y")
            rx = get_2d("R_HIP_POSITION_X")
            ry = get_2d("R_HIP_POSITION_Y")
        except KeyError as e:
            print(f"  [WARNING] Missing variable {e} for {subject} / trial {trial} — skipping.")
            continue
 
        n_reps = lx.shape[1]
        for i in range(n_reps):
            vx = lx[:, i] - rx[:, i]
            vy = ly[:, i] - ry[:, i]
            angle_rad = np.unwrap(np.arctan2(vy, vx))
            angle_deg = np.rad2deg(angle_rad)
            angle_deg -= angle_deg[0]   # reference to frame 0
            all_rotations.append(angle_deg)
 
    if not all_rotations:
        raise ValueError(f"No valid data found for subject '{subject}'.")
 
    mean_rotation = np.array(all_rotations).mean(axis=0)  # (n_frames,)
    return pd.Series(mean_rotation, index=df.index)
 
 
def compute_group_curves(df, subjects):
    """Return a DataFrame where each column is one subject's mean rotation."""
    curves = {}
    for subj in subjects:
        if subj not in df.columns.get_level_values("subject"):
            print(f"  [WARNING] Subject '{subj}' not found in data — skipping.")
            continue
        curves[subj] = compute_pelvis_rotation(df, subj)
    return pd.DataFrame(curves)
 
 
print("Computing pelvis rotation …")
g1_curves = compute_group_curves(df, GROUP_1_SUBJECTS)
g2_curves = compute_group_curves(df, GROUP_2_SUBJECTS)
 
g1_mean = g1_curves.mean(axis=1)
g1_sd   = g1_curves.std(axis=1)
g2_mean = g2_curves.mean(axis=1)
g2_sd   = g2_curves.std(axis=1)
 
frames = df.index.values
 
 
# =============================================================================
# PLOTTING
# =============================================================================
 
GROUP_1_COLOR = "#2C7BB6"
GROUP_2_COLOR = "#D7191C"
 
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle("Pelvis Rotation about Z-axis\n(relative to frame 0)",
             fontsize=14, fontweight="bold", y=1.01)
 
def plot_group(ax, curves, mean, sd, color, label, title):
    """Plot individual subject traces + group mean ± SD."""
    # Individual traces
    n_subj = curves.shape[1]
    alpha_individual = max(0.15, 0.6 / max(n_subj, 1))
    for col in curves.columns:
        ax.plot(frames, curves[col], color=color, alpha=alpha_individual,
                linewidth=1.0, label="_nolegend_")
 
    # Mean ± SD band
    ax.fill_between(frames, mean - sd, mean + sd,
                    color=color, alpha=0.20, label="± 1 SD")
    ax.plot(frames, mean, color=color, linewidth=2.5, label=f"{label} mean")
 
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Pelvis Rotation (°)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    n_label = f"n = {curves.shape[1]} subject{'s' if curves.shape[1] != 1 else ''}"
    ax.text(0.02, 0.97, n_label, transform=ax.transAxes,
            fontsize=9, va="top", color="gray")
 
 
plot_group(axes[0], g1_curves, g1_mean, g1_sd,
           GROUP_1_COLOR, GROUP_1_LABEL,
           f"{GROUP_1_LABEL}\nIndividual Subjects + Group Mean ± SD")
 
plot_group(axes[1], g2_curves, g2_mean, g2_sd,
           GROUP_2_COLOR, GROUP_2_LABEL,
           f"{GROUP_2_LABEL}\nIndividual Subjects + Group Mean ± SD")
 
# Overlay comparison on a third panel
fig2, ax_compare = plt.subplots(figsize=(10, 5))
ax_compare.fill_between(frames, g1_mean - g1_sd, g1_mean + g1_sd,
                         color=GROUP_1_COLOR, alpha=0.20, label=f"{GROUP_1_LABEL} ± 1 SD")
ax_compare.fill_between(frames, g2_mean - g2_sd, g2_mean + g2_sd,
                         color=GROUP_2_COLOR, alpha=0.20, label=f"{GROUP_2_LABEL} ± 1 SD")
ax_compare.plot(frames, g1_mean, color=GROUP_1_COLOR, linewidth=2.5, label=f"{GROUP_1_LABEL} mean")
ax_compare.plot(frames, g2_mean, color=GROUP_2_COLOR, linewidth=2.5, label=f"{GROUP_2_LABEL} mean")
ax_compare.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax_compare.set_title("Group Comparison — Pelvis Rotation (Z-axis)", fontsize=13, fontweight="bold")
ax_compare.set_xlabel("Frame", fontsize=11)
ax_compare.set_ylabel("Pelvis Rotation (°)", fontsize=11)
ax_compare.legend(fontsize=10)
ax_compare.grid(True, linestyle=":", alpha=0.5)
 
fig.tight_layout()
fig2.tight_layout()
plt.show()
 
print("Done.")
 