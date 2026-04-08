"""
Biomechanics trial visualiser
==============================
Plots the mean ± one standard deviation of a chosen variable across all
subjects and trials in a MultiIndex DataFrame.
 
DataFrame index structure
--------------------------
  Level 0 : subject pseudonym  (e.g. "S01", "S02", …)
  Level 1 : trial identifier   (e.g. "trial_1", "trial_2", …)
  Level 2 : variable name      (e.g. "COG_x", "COG_y", "ankle_angle", …)
 
Row index : integer frame number or timestamp (the time axis).
 
Usage
------
  1. Set VARIABLE and the optional display options below.
  2. Replace the load_group() body with your own data loading logic.
  3. Run:  python plot_biomechanics.py
"""
 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection
import BigLoader
 
# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# Applied once at import time so every figure shares the same look.
# ──────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0,       # no y tick marks — rely on grid instead
    "ytick.major.size":   0,
    "xtick.major.size":   4,
    "xtick.direction":    "out",
    "axes.axisbelow":     True,
    "figure.dpi":         150,
})
 
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these
# ──────────────────────────────────────────────────────────────────────────────
experts = ["E1", "E2", "E3"]
novices = ["N1", "N2", "N3", "N4"]
 
VARIABLE    = "Pelvis_CoG_pos_Z"
SHADE_ALPHA = 0.20       # opacity of the ± 1 SD band
FADE_EDGES  = 0.12       # fraction of trial to fade in/out (0 = hard edges)
FIG_SIZE    = (15, 7.5)
 
SAVE_DIR = (
    "/home/paul/Schreibtisch/Bachelorarbeit/"
    "Bachelor_Muay_Thai/Plots/ComparisonMeanSdRoundhouse/ALLAMO"
)
 
# Thesis colour palette — muted, distinct, greyscale-safe
_C = {
    "expert":       "#C0392B",   # deep crimson
    "expert_light": "#E8A598",
    "novice":       "#2874A6",   # steel blue
    "novice_light": "#A9C4D9",
    "single":       "#1A6B5A",   # teal
    "single_light": "#A2C9C0",
    "trial":        "#B0B0B0",   # neutral grey for individual traces
}
 
# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
 
def load_group(subjects) -> pd.DataFrame:
    data = BigLoader.loadallspecified(subjects, "teep")
    data.columns.names = ["subject", "trial", "variable"]
    return data
 
# ──────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ──────────────────────────────────────────────────────────────────────────────
 
def build_matrix(df: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        time_axis : (n_frames,)
        matrix    : (n_trials, n_frames)
    """
    try:
        var_df = df.xs(variable, level="variable", axis=1)
    except KeyError:
        available = df.columns.get_level_values("variable").unique().tolist()
        raise KeyError(
            f"Variable '{variable}' not found.\nAvailable: {available}"
        )
    if var_df.empty:
        raise ValueError(f"No data found for variable '{variable}'.")
 
    matrix    = var_df.to_numpy(dtype=float).T
    time_axis = df.index.to_numpy(dtype=float)
    return time_axis, matrix
 
 
def _alpha_envelope(n: int, fade: float) -> np.ndarray:
    env = np.ones(n)
    if fade <= 0:
        return env
    k = max(1, int(round(fade * n)))
    ramp = np.linspace(0, 1, k)
    env[:k]  = ramp
    env[-k:] = ramp[::-1]
    return env
 
 
def _faded_band(ax, x, y_lo, y_hi, face_color, edge_color,
                max_alpha, fade, label=None, zorder=2):
    """
    Draws a per-segment shaded band that tapers to transparent at both ends.
    Also draws a matching thin border line on the upper and lower edges.
    """
    env = _alpha_envelope(len(x), fade)
    r, g, b = mpl.colors.to_rgb(face_color)
 
    verts, facecolors = [], []
    for i in range(len(x) - 1):
        a = max_alpha * 0.5 * (env[i] + env[i + 1])
        verts.append([
            (x[i],   y_lo[i]),   (x[i+1], y_lo[i+1]),
            (x[i+1], y_hi[i+1]), (x[i],   y_hi[i]),
        ])
        facecolors.append((r, g, b, a))
 
    col = PolyCollection(verts, facecolors=facecolors, linewidths=0,
                         zorder=zorder, label=label)
    ax.add_collection(col)
 
    # Thin border lines on band edges — use per-segment alpha too
    er, eg, eb = mpl.colors.to_rgb(edge_color)
    for i in range(len(x) - 1):
        a = 0.55 * 0.5 * (env[i] + env[i + 1])
        ax.plot(x[i:i+2], y_lo[i:i+2], color=(er, eg, eb, a),
                linewidth=0.6, zorder=zorder + 1)
        ax.plot(x[i:i+2], y_hi[i:i+2], color=(er, eg, eb, a),
                linewidth=0.6, zorder=zorder + 1)
 
 
def _apply_grid(ax):
    """Horizontal-only grid, single level, very faint."""
    ax.yaxis.grid(True,  color="0.88", linewidth=0.6, linestyle="-")
    ax.xaxis.grid(False)
    # Thin bottom axis line as x-axis reference
    ax.axhline(0, color="0.80", linewidth=0.5, zorder=0)
 
 
def _format_variable_label(variable: str) -> str:
    """Turn 'Pelvis_CoG_pos_Z' into a readable axis label."""
    return variable.replace("_", " ")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING — single group
# ──────────────────────────────────────────────────────────────────────────────
 
def plot_mean_sd(
    variable:    str,
    shade_alpha: float = SHADE_ALPHA,
    fade_edges:  float = FADE_EDGES,
    fig_size:    tuple = FIG_SIZE,
    group:       list  = None,
) -> None:
 
    df                = load_group(group)
    time_axis, matrix = build_matrix(df, variable)
 
    n_trials = matrix.shape[0]
    mean     = matrix.mean(axis=0)
    std      = matrix.std(axis=0, ddof=1)
 
    fig, ax = plt.subplots(figsize=fig_size)
 
    # Individual trial traces
    for row in matrix:
        ax.plot(time_axis, row,
                color=_C["trial"], alpha=0.18, linewidth=0.7, zorder=1)
 
    # ± 1 SD band
    _faded_band(ax, time_axis, mean - std, mean + std,
                face_color=_C["single"], edge_color=_C["single"],
                max_alpha=shade_alpha, fade=fade_edges,
                label=r"$\pm 1\,\mathrm{SD}$")
 
    # Mean line
    ax.plot(time_axis, mean,
            color=_C["single"], linewidth=2.0, label="Mean", zorder=4)
 
    _apply_grid(ax)
 
    ax.set_xlabel("Frame", fontsize=11, color="0.3", labelpad=6)
    ax.set_ylabel(_format_variable_label(variable), fontsize=11, color="0.3", labelpad=6)
    ax.set_title(
        f"{_format_variable_label(variable)}",
        fontsize=13, fontweight="medium", pad=12, loc="left",
    )
    ax.text(0.0, 1.02,
            f"Mean ± 1 SD  ·  {n_trials} trial{'s' if n_trials != 1 else ''}",
            transform=ax.transAxes, fontsize=9, color="0.5")
 
    ax.legend(frameon=True, fontsize=9, loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.8)
    ax.tick_params(labelsize=9, colors="0.4")
    ax.autoscale_view()
 
    plt.tight_layout()
    plt.show()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING — comparison
# ──────────────────────────────────────────────────────────────────────────────
 
def plot_mean_sd_comparison(
    variable:    str,
    shade_alpha: float = SHADE_ALPHA,
    fade_edges:  float = FADE_EDGES,
    fig_size:    tuple = FIG_SIZE,
) -> None:
 
    df_experts = load_group(experts)
    df_novices = load_group(novices)
 
    t_exp, mat_exp = build_matrix(df_experts, variable)
    t_nov, mat_nov = build_matrix(df_novices, variable)
 
    mean_exp = mat_exp.mean(axis=0);  std_exp = mat_exp.std(axis=0, ddof=1)
    mean_nov = mat_nov.mean(axis=0);  std_nov = mat_nov.std(axis=0, ddof=1)
 
    fig, ax = plt.subplots(figsize=fig_size)
 
    # Experts band + mean
    _faded_band(ax, t_exp, mean_exp - std_exp, mean_exp + std_exp,
                face_color=_C["expert"], edge_color=_C["expert"],
                max_alpha=shade_alpha, fade=fade_edges,
                label="Experts  ±1 SD")
    ax.plot(t_exp, mean_exp,
            color=_C["expert"], linewidth=2.0, label="Experts  mean", zorder=4)
 
    # Novices band + mean
    _faded_band(ax, t_nov, mean_nov - std_nov, mean_nov + std_nov,
                face_color=_C["novice"], edge_color=_C["novice"],
                max_alpha=shade_alpha, fade=fade_edges,
                label="Novices  ±1 SD")
    ax.plot(t_nov, mean_nov,
            color=_C["novice"], linewidth=2.0, label="Novices  mean", zorder=4)
 
    _apply_grid(ax)
 
    label = _format_variable_label(variable[:-1] if variable[-1] in "XYZ" else variable)
    ax.set_xlabel("Frame", fontsize=11, color="0.3", labelpad=6)
    ax.set_ylabel(label, fontsize=11, color="0.3", labelpad=6)
    ax.set_title(label, fontsize=13, fontweight="medium", pad=12, loc="left")
    ax.text(0.0, 1.02, "Experts vs Novices  ·  Mean ± 1 SD",
            transform=ax.transAxes, fontsize=9, color="0.5")
 
    ax.legend(frameon=True, fontsize=9, loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.8,
              ncol=2)
    ax.tick_params(labelsize=9, colors="0.4")
    ax.autoscale_view()
 
    plt.tight_layout()
    plt.show()
    
 
 
# ──────────────────────────────────────────────────────────────────────────────
# VARIABLE LISTS
# ──────────────────────────────────────────────────────────────────────────────
 
cogvariables = [
    "FullBody_CoG_pos_X", "FullBody_CoG_pos_Y", "FullBody_CoG_pos_Z",
    "L_Hand_CoG_pos_X",   "L_Hand_CoG_pos_Y",   "L_Hand_CoG_pos_Z",
    "R_Hand_CoG_pos_X",   "R_Hand_CoG_pos_Y",   "R_Hand_CoG_pos_Z",
    "L_Forearm_CoG_pos_X","L_Forearm_CoG_pos_Y","L_Forearm_CoG_pos_Z",
    "R_Forearm_CoG_pos_X","R_Forearm_CoG_pos_Y","R_Forearm_CoG_pos_Z",
    "L_UpperArm_CoG_pos_X","L_UpperArm_CoG_pos_Y","L_UpperArm_CoG_pos_Z",
    "R_UpperArm_CoG_pos_X","R_UpperArm_CoG_pos_Y","R_UpperArm_CoG_pos_Z",
    "Head_Shadow_CoG_pos_X","Head_Shadow_CoG_pos_Y","Head_Shadow_CoG_pos_Z",
    "Head_CoG_pos_X",     "Head_CoG_pos_Y",     "Head_CoG_pos_Z",
    "Trunk_CoG_pos_X",    "Trunk_CoG_pos_Y",    "Trunk_CoG_pos_Z",
    "Pelvis_CoG_pos_X",   "Pelvis_CoG_pos_Y",   "Pelvis_CoG_pos_Z",
    "L_Thigh_CoG_pos_X",  "L_Thigh_CoG_pos_Y",  "L_Thigh_CoG_pos_Z",
    "R_Thigh_CoG_pos_X",  "R_Thigh_CoG_pos_Y",  "R_Thigh_CoG_pos_Z",
    "L_Shank_CoG_pos_X",  "L_Shank_CoG_pos_Y",  "L_Shank_CoG_pos_Z",
    "R_Shank_CoG_pos_X",  "R_Shank_CoG_pos_Y",  "R_Shank_CoG_pos_Z",
    "L_Foot_CoG_pos_X",   "L_Foot_CoG_pos_Y",   "L_Foot_CoG_pos_Z",
    "R_Foot_CoG_pos_X",   "R_Foot_CoG_pos_Y",   "R_Foot_CoG_pos_Z",
    "L_Foot_Shadow_CoG_pos_X","L_Foot_Shadow_CoG_pos_Y","L_Foot_Shadow_CoG_pos_Z",
    "R_Foot_Shadow_CoG_pos_X","R_Foot_Shadow_CoG_pos_Y","R_Foot_Shadow_CoG_pos_Z",
]
 
AMAvariables = [
    "L_HandAMAC","R_HandAMAC","L_FAAMAC","R_FAAMAC","L_UAAMAC","R_UAAMAC",
    "HeadAMAC","TrunkAMAC","PelvisAMAC",
    "L_ThighAMAC","R_ThighAMAC","L_ShankAMAC","R_ShankAMAC",
    "L_FootAMAC","R_FootAMAC",
]
 
AMOvariables = [
    "L_HandAMOC","R_HandAMOC","L_FAAMOC","R_FAAMOC","L_UAAMOC","R_UAAMOC",
    "HeadAMOC","TrunkAMOC","PelvisAMOC",
    "L_ThighAMOC","R_ThighAMOC","L_ShankAMOC","R_ShankAMOC",
    "L_FootAMOC","R_FootAMOC",
]
 
# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    for variable in AMOvariables:
        plot_mean_sd_comparison(
            variable    = variable,
            shade_alpha = SHADE_ALPHA,
            fade_edges  = FADE_EDGES,
            fig_size    = FIG_SIZE,
        )
    plot_mean_sd(
        variable    = VARIABLE,
        shade_alpha = SHADE_ALPHA,
        fade_edges  = FADE_EDGES,
        fig_size    = FIG_SIZE,
        group       = experts,
    )
 