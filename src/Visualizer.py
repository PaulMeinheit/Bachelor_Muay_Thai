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
 
from pydoc import text

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
"""e1 = ["E1"]
e2 = ["E2"]
e3 = ["E3"]"""
experts = ["E1", "E2", "E3"]
novices = ["N1", "N2", "N3", "N4"]

SHADE_ALPHA = 0.20       # opacity of the ± 1 SD band
FADE_EDGES  = 0.12       # fraction of trial to fade in/out (0 = hard edges)
FIG_SIZE    = (15, 7.5)
 
SAVE_DIR = (
    "/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/Plots/ToLookAt/AngMomAndFriends/TotalAngMom"
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
 
def load_group(subjects, movement) -> pd.DataFrame:
    data = BigLoader.loadallspecified(subjects, movement)
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


def build_xyz(df: pd.DataFrame, base: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract X, Y, Z matrices for a position variable base name.

    Expects columns named  <base>_X,  <base>_Y,  <base>_Z  in the DataFrame.

    Returns
    -------
    time_axis : (n_frames,)
    mat_x     : (n_trials, n_frames)
    mat_y     : (n_trials, n_frames)
    mat_z     : (n_trials, n_frames)
    """
    time_axis, mat_x = build_matrix(df, f"{base}_X")
    _,         mat_y = build_matrix(df, f"{base}_Y")
    _,         mat_z = build_matrix(df, f"{base}_Z")
    return time_axis, mat_x, mat_y, mat_z

 
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


def _style_3d_ax(ax):
    """Apply consistent styling to a 3-D axes — matches the project's clean look."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("0.88")
    ax.yaxis.pane.set_edgecolor("0.88")
    ax.zaxis.pane.set_edgecolor("0.88")
    ax.grid(True, color="0.92", linewidth=0.5)
    ax.tick_params(labelsize=8, colors="0.4")


def build_norm_matrix(df: pd.DataFrame, base: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract X, Y, Z components for a variable base name and compute Euclidean norm.

    Expects columns named  <base>_X,  <base>_Y,  <base>_Z  in the DataFrame.

    Returns
    -------
    time_axis : (n_frames,)
    norm_mat  : (n_trials, n_frames)  containing sqrt(X² + Y² + Z²)
    """
    time_axis, mat_x = build_matrix(df, f"{base}_X")
    _,         mat_y = build_matrix(df, f"{base}_Y")
    _,         mat_z = build_matrix(df, f"{base}_Z")
    
    # Calculate Euclidean norm for each trial and frame
    norm_mat = np.sqrt(mat_x**2 + mat_y**2 + mat_z**2)
    return time_axis, norm_mat


def analyze_norm_max_per_trial(
    variable_base: str,
    movement:      str,
) -> None:
    """
    Analyze the maximum norm value for each trial across all frames.
    
    Calculates and prints:
    - Max norm value for each trial of each participant
    - Mean and std of max norms per participant
    - Group means and standard deviations (Experts vs Novices)
    
    Parameters
    ----------
    variable_base  : base name WITHOUT the _X/_Y/_Z suffix,
                     e.g. "R_shoulder_CoG_velocity".
    movement       : movement type string.
    """
    print(f"\n{'='*80}")
    print(f"Analysis of {variable_base} (Euclidean norm)")
    print(f"Movement: {movement.upper()}")
    print(f"{'='*80}\n")
    
    # Load data and extract X, Y, Z with subject/trial information intact
    df_experts = load_group(experts, movement)
    df_novices = load_group(novices, movement)
    
    # Extract X, Y, Z matrices (still have subject/trial in MultiIndex)
    _, mat_x_exp = build_matrix(df_experts, f"{variable_base}_X")
    _, mat_y_exp = build_matrix(df_experts, f"{variable_base}_Y")
    _, mat_z_exp = build_matrix(df_experts, f"{variable_base}_Z")
    
    _, mat_x_nov = build_matrix(df_novices, f"{variable_base}_X")
    _, mat_y_nov = build_matrix(df_novices, f"{variable_base}_Y")
    _, mat_z_nov = build_matrix(df_novices, f"{variable_base}_Z")
    
    # Calculate norms
    norm_exp = np.sqrt(mat_x_exp**2 + mat_y_exp**2 + mat_z_exp**2)  # (n_trials, n_frames)
    norm_nov = np.sqrt(mat_x_nov**2 + mat_y_nov**2 + mat_z_nov**2)
    
    # Get trial identifiers from column MultiIndex
    trial_indices_exp = df_experts.xs(f"{variable_base}_X", level="variable", axis=1).columns.tolist()
    trial_indices_nov = df_novices.xs(f"{variable_base}_X", level="variable", axis=1).columns.tolist()
    
    # Find max value for each trial
    max_vals_exp = np.max(norm_exp, axis=1)  # (n_trials,)
    max_vals_nov = np.max(norm_nov, axis=1)
    
    # Group by participant (subject)
    participant_maxes_exp = {}  # subject -> list of max values
    participant_maxes_nov = {}
    
    for trial_idx, max_val in zip(trial_indices_exp, max_vals_exp):
        subject, trial = trial_idx
        if subject not in participant_maxes_exp:
            participant_maxes_exp[subject] = []
        participant_maxes_exp[subject].append(max_val)
    
    for trial_idx, max_val in zip(trial_indices_nov, max_vals_nov):
        subject, trial = trial_idx
        if subject not in participant_maxes_nov:
            participant_maxes_nov[subject] = []
        participant_maxes_nov[subject].append(max_val)
    
    # === EXPERTS ===
    print("EXPERTS:")
    print(f"{'-'*60}")
    expert_all_maxes = []
    for subject in sorted(participant_maxes_exp.keys()):
        maxes = np.array(participant_maxes_exp[subject])
        mean = np.mean(maxes)
        std = np.std(maxes, ddof=1)
        expert_all_maxes.extend(maxes)
        print(f"  {subject}: {len(maxes):2d} trials  |  "
              f"Mean: {mean:8.3f}  |  Std: {std:8.3f}  |  "
              f"Individual: {', '.join([f'{m:8.2f}' for m in maxes])}")
    
    # Group statistics for experts
    expert_all_maxes = np.array(expert_all_maxes)
    expert_group_mean = np.mean(expert_all_maxes)
    expert_group_std = np.std(expert_all_maxes, ddof=1)
    print(f"\n  EXPERT GROUP: {len(expert_all_maxes)} total trials")
    print(f"  Group Mean: {expert_group_mean:.3f}  |  Group Std: {expert_group_std:.3f}\n")
    
    # === NOVICES ===
    print("NOVICES:")
    print(f"{'-'*60}")
    novice_all_maxes = []
    for subject in sorted(participant_maxes_nov.keys()):
        maxes = np.array(participant_maxes_nov[subject])
        mean = np.mean(maxes)
        std = np.std(maxes, ddof=1)
        novice_all_maxes.extend(maxes)
        print(f"  {subject}: {len(maxes):2d} trials  |  "
              f"Mean: {mean:8.3f}  |  Std: {std:8.3f}  |  "
              f"Individual: {', '.join([f'{m:8.2f}' for m in maxes])}")
    
    # Group statistics for novices
    novice_all_maxes = np.array(novice_all_maxes)
    novice_group_mean = np.mean(novice_all_maxes)
    novice_group_std = np.std(novice_all_maxes, ddof=1)
    print(f"\n  NOVICE GROUP: {len(novice_all_maxes)} total trials")
    print(f"  Group Mean: {novice_group_mean:.3f}  |  Group Std: {novice_group_std:.3f}\n")
    
    # === SUMMARY ===
    print(f"{'='*80}")
    print("SUMMARY:")
    print(f"{'-'*60}")
    print(f"Experts  Mean: {expert_group_mean:.3f}  ±  {expert_group_std:.3f}")
    print(f"Novices  Mean: {novice_group_mean:.3f}  ±  {novice_group_std:.3f}")
    print(f"Difference (Experts - Novices): {expert_group_mean - novice_group_mean:.3f}")
    print(f"{'='*80}\n")


def _format_variable_label(variable: str) -> str:
    """Format variable name for display as axis label."""
    return variable.replace("_", " ")


def _make_gradient_segments(x, y, z):
    """Build (n-1, 2, 3) segment array for Line3DCollection."""
    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)

 
# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING — single group (2-D mean ± SD)
# ──────────────────────────────────────────────────────────────────────────────
 
def plot_mean_sd(
    variable:    str,
    movement:    str,
    shade_alpha: float = SHADE_ALPHA,
    fade_edges:  float = FADE_EDGES,
    fig_size:    tuple = FIG_SIZE,
    group:       list  = None,
    text_size:   float = 1.0,
    title:       str   = None,
    x_label:     str   = None,
    y_label:     str   = None,
    show_n_trials: bool = False,
) -> None:
    df                = load_group(group, movement)
    time_axis, matrix = build_matrix(df, variable)
 
    n_trials = matrix.shape[0]
    mean     = matrix.mean(axis=0)
    std      = matrix.std(axis=0, ddof=1)
 
    fig, ax = plt.subplots(figsize=fig_size)
 
    # Individual trial traces
    for row in matrix:
        ax.plot(time_axis, row,
                color=_C["trial"], alpha=0.5, linewidth=0.7, zorder=1)
 
    # ± 1 SD band
    _faded_band(ax, time_axis, mean - std, mean + std,
                face_color=_C["single"], edge_color=_C["single"],
                max_alpha=shade_alpha, fade=fade_edges,
                label=r"$\pm 1\,\mathrm{SD}$")
 
    # Mean line
    ax.plot(time_axis, mean,
            color=_C["single"], linewidth=2.0, label="Mean", zorder=4)
 
    _apply_grid(ax)
 
    ax.set_xlabel(x_label if x_label is not None else "Frame", fontsize=int(11*text_size), color="0.3", labelpad=6)
    ax.set_ylabel(y_label if y_label is not None else _format_variable_label(variable), fontsize=int(11*text_size), color="0.3", labelpad=6)
    if title is None:
        title = _format_variable_label(variable)
    ax.set_title(
        title,
        fontsize=int(13*text_size), fontweight="medium", pad=12, loc="left",
    )
    ax.text(0.0, 1.02,
            f"Mean ± 1 SD  ·  {n_trials} trial{'s' if n_trials != 1 else ''}",
            transform=ax.transAxes, fontsize=int(9*text_size), color="0.5")

    ax.legend(frameon=True, fontsize=int(9*text_size), loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.8)
    ax.tick_params(labelsize=int(9*text_size), colors="0.4")
    ax.autoscale_view()
 
    plt.tight_layout()
    plt.show()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING — comparison (2-D mean ± SD)
# ──────────────────────────────────────────────────────────────────────────────
 
def plot_mean_sd_comparison(
    variable:    str,
    movement:    str,
    shade_alpha: float = SHADE_ALPHA,
    fade_edges:  float = FADE_EDGES,
    fig_size:    tuple = FIG_SIZE,
    label:       str   = None,
    text_size:   float = 1.0,
    title:       str   = None,
    x_label:     str   = None,
    y_label:     str   = None,
    show_n_trials: bool = False,
) -> None:
    df_experts = load_group(experts, movement)
    df_novices = load_group(novices, movement)
 
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

    ax.set_xlabel(x_label if x_label is not None else "scaled time", fontsize=int(16*text_size), color="0.3", labelpad=6)
    ax.set_ylabel(y_label if y_label is not None else label, fontsize=int(16*text_size), color="0.3", labelpad=6)
    if title is None:
        title = label
    #ax.set_title(title, fontsize=int(20*text_size), fontweight="medium", pad=12, loc="left")

    ax.legend(frameon=True, fontsize=int(9*text_size), loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.5,
              ncol=2)
    ax.tick_params(labelsize=int(14*text_size), colors="0.4")
    ax.autoscale_view()

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/_{movement}_{variable}_comparison.png", dpi=300)
    #plt.show()

def plot_mean_sd_FirstGradientComparison(
    variable_base: str,
    movement:      str,
    shade_alpha:   float = SHADE_ALPHA,
    fade_edges:    float = FADE_EDGES,
    fig_size:      tuple = FIG_SIZE,
    label:         str   = None,
    text_size:     float = 1.0,
    title:         str   = None,
    x_label:       str   = None,
    y_label:       str   = None,
    show_n_trials: bool  = False,
) -> None:
    """
    Plot the Euclidean norm of the first time-derivative (gradient) of a
    3-D vector variable, comparing Experts vs Novices with mean ± SD bands.

    For angular momentum L = (L_x, L_y, L_z) this yields ||dL/dt||,
    i.e. the magnitude of the net torque acting on the segment over time.

    Parameters
    ----------
    variable_base : base name WITHOUT the _X/_Y/_Z suffix,
                    e.g. "R_shoulder_angular_momentum"
                    → uses _X, _Y, _Z columns.
    movement      : movement type string passed to load_group().
    """
    df_experts = load_group(experts, movement)
    df_novices = load_group(novices, movement)

    # ── Load all three components ─────────────────────────────────────────────
    t_exp, mat_x_exp = build_matrix(df_experts, f"{variable_base}_X")
    _,     mat_y_exp = build_matrix(df_experts, f"{variable_base}_Y")
    _,     mat_z_exp = build_matrix(df_experts, f"{variable_base}_Z")

    t_nov, mat_x_nov = build_matrix(df_novices, f"{variable_base}_X")
    _,     mat_y_nov = build_matrix(df_novices, f"{variable_base}_Y")
    _,     mat_z_nov = build_matrix(df_novices, f"{variable_base}_Z")

    # ── Gradient of each component per trial, then norm ───────────────────────
    # grad shape: (n_trials, n_frames)
    grad_norm_exp = np.sqrt(
        np.array([np.gradient(row, t_exp) for row in mat_x_exp]) ** 2 +
        np.array([np.gradient(row, t_exp) for row in mat_y_exp]) ** 2 +
        np.array([np.gradient(row, t_exp) for row in mat_z_exp]) ** 2
    )

    grad_norm_nov = np.sqrt(
        np.array([np.gradient(row, t_nov) for row in mat_x_nov]) ** 2 +
        np.array([np.gradient(row, t_nov) for row in mat_y_nov]) ** 2 +
        np.array([np.gradient(row, t_nov) for row in mat_z_nov]) ** 2
    )

    # ── Statistics ────────────────────────────────────────────────────────────
    mean_exp = grad_norm_exp.mean(axis=0);  std_exp = grad_norm_exp.std(axis=0, ddof=1)
    mean_nov = grad_norm_nov.mean(axis=0);  std_nov = grad_norm_nov.std(axis=0, ddof=1)

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

    # ── Labels ────────────────────────────────────────────────────────────────
    ax.set_xlabel(
        x_label if x_label is not None else "scaled time",
        fontsize=int(16 * text_size), color="0.3", labelpad=6,
    )

    if label is None:
        label = f"‖d/dt {_format_variable_label(variable_base)}‖"

    ax.set_ylabel(
        y_label if y_label is not None else label,
        fontsize=int(16 * text_size), color="0.3", labelpad=6,
    )

    if title is None:
        title = label
    ax.set_title(title, fontsize=int(20 * text_size), fontweight="medium", pad=12, loc="left")
    ax.text(0.0, 1.02, "Experts vs Novices  ·  Mean ± 1 SD  ·  ‖dL/dt‖",
            transform=ax.transAxes, fontsize=int(9 * text_size), color="0.5")

    ax.legend(frameon=True, fontsize=int(9 * text_size), loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.5, ncol=2)
    ax.tick_params(labelsize=int(14 * text_size), colors="0.4")
    ax.autoscale_view()

    plt.tight_layout()
    plt.savefig(
        f"{SAVE_DIR}/{variable_base}_{movement}_gradient_norm_comparison.png",
        dpi=300,
    )
    #plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING — norm comparison (2-D mean ± SD)
# ──────────────────────────────────────────────────────────────────────────────

def plot_mean_sd_norm(
    variable_base: str,
    movement:      str,
    shade_alpha:   float = SHADE_ALPHA,
    fade_edges:    float = FADE_EDGES,
    fig_size:      tuple = FIG_SIZE,
    group:         list  = None,
    text_size:     float = 1.0,
    title:         str   = None,
    x_label:       str   = None,
    y_label:       str   = None,
    show_n_trials: bool = False,
) -> None:
    """
    Plot the Euclidean norm of a multivariate variable with mean ± SD band.
    
    Parameters
    ----------
    variable_base  : base name WITHOUT the _X/_Y/_Z suffix,
                     e.g. "R_shoulder_CoG_velocity"  →  uses _X, _Y, _Z columns.
    movement       : movement type string.
    shade_alpha    : opacity of the ± 1 SD band.
    fade_edges     : fraction of trial to fade in/out (0 = hard edges).
    fig_size       : figure size tuple.
    group          : list of subject IDs (if None, uses all subjects).
    """
    if group is None:
        group = experts + novices
    
    df = load_group(group, movement)
    time_axis, norm_mat = build_norm_matrix(df, variable_base)
    
    n_trials = norm_mat.shape[0]
    mean     = norm_mat.mean(axis=0)
    std      = norm_mat.std(axis=0, ddof=1)
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Individual trial traces
    for row in norm_mat:
        ax.plot(time_axis, row,
                color=_C["trial"], alpha=0.5, linewidth=0.7, zorder=1)
    
    # ± 1 SD band
    _faded_band(ax, time_axis, mean - std, mean + std,
                face_color=_C["single"], edge_color=_C["single"],
                max_alpha=shade_alpha, fade=fade_edges,
                label=r"$\pm 1\,\mathrm{SD}$")
    
    # Mean line
    ax.plot(time_axis, mean,
            color=_C["single"], linewidth=2.0, label="Mean", zorder=4)
    
    _apply_grid(ax)
    
    ax.set_xlabel(x_label if x_label is not None else "Frame", fontsize=int(11*text_size), color="0.3", labelpad=6)
    default_y_label = f"{_format_variable_label(variable_base)} (norm)"
    ax.set_ylabel(y_label if y_label is not None else default_y_label, fontsize=int(11*text_size), color="0.3", labelpad=6)
    if title is None:
        title = default_y_label
    ax.set_title(
        title,
        fontsize=int(13*text_size), fontweight="medium", pad=12, loc="left",
    )
    ax.text(0.0, 1.02,
            f"Mean ± 1 SD  ·  {n_trials} trial{'s' if n_trials != 1 else ''}",
            transform=ax.transAxes, fontsize=int(9*text_size), color="0.5")
    
    ax.legend(frameon=True, fontsize=int(9*text_size), loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.8)
    ax.tick_params(labelsize=int(9*text_size), colors="0.4")
    ax.autoscale_view()
    
    plt.tight_layout()
    plt.show()


def plot_mean_sd_comparison_norm(
    variable_base: str,
    movement:      str,
    shade_alpha:   float = SHADE_ALPHA,
    fade_edges:    float = FADE_EDGES,
    fig_size:      tuple = FIG_SIZE,
    label:         str   = None,
    text_size:     float = 1.0,
    title:         str   = None,
    x_label:       str   = None,
    y_label:       str   = None,
    show_n_trials: bool = False,
) -> None:
    """
    Plot the Euclidean norm of a multivariate variable comparing Experts vs Novices
    with mean ± SD bands.
    
    Parameters
    ----------
    variable_base  : base name WITHOUT the _X/_Y/_Z suffix,
                     e.g. "R_shoulder_CoG_velocity"  →  uses _X, _Y, _Z columns.
    movement       : movement type string.
    shade_alpha    : opacity of the ± 1 SD band.
    fade_edges     : fraction of trial to fade in/out (0 = hard edges).
    fig_size       : figure size tuple.
    label          : custom label for the y-axis (if None, auto-generated).
    """
    df_experts = load_group(experts, movement)
    df_novices = load_group(novices, movement)
    
    t_exp, norm_exp = build_norm_matrix(df_experts, variable_base)
    t_nov, norm_nov = build_norm_matrix(df_novices, variable_base)
    
    mean_exp = norm_exp.mean(axis=0);  std_exp = norm_exp.std(axis=0, ddof=1)
    mean_nov = norm_nov.mean(axis=0);  std_nov = norm_nov.std(axis=0, ddof=1)
    
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
    
    ax.set_xlabel("scaled time", fontsize=int(11*text_size), color="0.3", labelpad=6)
    
    # Auto-generate label if not provided
    if label is None:
        label = f"{_format_variable_label(variable_base)} (norm)"
    
    ax.set_ylabel(y_label if y_label is not None else label, fontsize=int(11*text_size), color="0.3", labelpad=6)
    if title is None:
        title = ""
    ax.set_title(title, fontsize=int(13*text_size), fontweight="medium", pad=12, loc="left")
    
    
    ax.legend(frameon=True, fontsize=int(9*text_size), loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.8,
              ncol=2)
    ax.tick_params(labelsize=int(9*text_size), colors="0.4")
    ax.autoscale_view()
    #plt.savefig(f"{SAVE_DIR}/_{movement}_{variable_base}_TotalAngMomcomparison.png", dpi=300)
    plt.tight_layout()
    plt.show()


def plot_mean_sd_comparisonFirstDiff_norm(
    variable_base: str,
    movement:      str,
    shade_alpha:   float = SHADE_ALPHA,
    fade_edges:    float = FADE_EDGES,
    fig_size:      tuple = FIG_SIZE,
    label:         str   = None,
    text_size:     float = 1.0,
    title:         str   = None,
    x_label:       str   = None,
    y_label:       str   = None,
    show_n_trials: bool = False,
) -> None:
    """
    Plot the Euclidean norm of a multivariate variable comparing Experts vs Novices
    with mean ± SD bands.
    
    Parameters
    ----------
    variable_base  : base name WITHOUT the _X/_Y/_Z suffix,
                     e.g. "R_shoulder_CoG_velocity"  →  uses _X, _Y, _Z columns.
    movement       : movement type string.
    shade_alpha    : opacity of the ± 1 SD band.
    fade_edges     : fraction of trial to fade in/out (0 = hard edges).
    fig_size       : figure size tuple.
    label          : custom label for the y-axis (if None, auto-generated).
    """
    df_experts = load_group(experts, movement)
    df_novices = load_group(novices, movement)
    
    t_exp, norm_exp = build_norm_matrix(df_experts, variable_base)
    t_nov, norm_nov = build_norm_matrix(df_novices, variable_base)
    
    mean_exp = norm_exp.mean(axis=0);  std_exp = norm_exp.std(axis=0, ddof=1)
    mean_nov = norm_nov.mean(axis=0);  std_nov = norm_nov.std(axis=0, ddof=1)
    
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
    
    ax.set_xlabel("scaled time", fontsize=int(11*text_size), color="0.3", labelpad=6)
    
    # Auto-generate label if not provided
    if label is None:
        label = f"{_format_variable_label(variable_base)} (norm)"
    
    ax.set_ylabel(y_label if y_label is not None else label, fontsize=int(11*text_size), color="0.3", labelpad=6)
    if title is None:
        title = label
    ax.set_title(title, fontsize=int(13*text_size), fontweight="medium", pad=12, loc="left")
    ax.text(0.0, 1.02, "Experts vs Novices  ·  Mean ± 1 SD",
            transform=ax.transAxes, fontsize=int(9*text_size), color="0.5")
    
    ax.legend(frameon=True, fontsize=int(9*text_size), loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.8,
              ncol=2)
    ax.tick_params(labelsize=int(9*text_size), colors="0.4")
    ax.autoscale_view()
    
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING — 3-D trajectory, single group (mean + faint individual trials)
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory_3d(
    variable_base: str,
    movement:      str,
    group:         list,
    group_label:   str   = "Group mean",
    mean_color:    str   = _C["single"],
    trial_alpha:   float = 0.15,
    mean_linewidth:float = 2.5,
    trial_linewidth:float = 0.8,
    show_start_end:bool  = True,
    fig_size:      tuple = (11, 8),
    title:         str   = None,
    elev:          float = 20,
    azim:          float = 45,
    text_size:     float = 1.0,
    x_label:       str   = None,
    y_label:       str   = None,
    z_label:       str   = None,
    show_n_trials: bool = False,
) -> None:
    """
    Plot individual trial trajectories as faint grey lines and their mean
    trajectory as a bold coloured line in 3-D space.

    Parameters
    ----------
    variable_base  : position variable WITHOUT the _X/_Y/_Z suffix,
                     e.g. "R_Hand_CoG_pos"  →  uses _X, _Y, _Z columns.
    group          : list of subject IDs, e.g. experts or novices.
    group_label    : legend / subtitle label for the mean line.
    mean_color     : colour of the mean trajectory (defaults to teal).
    trial_alpha    : opacity of individual faint trial lines (0–1).
    mean_linewidth : line width of the mean trajectory.
    trial_linewidth: line width of individual trial lines.
    show_start_end : mark the mean start (green) and end (red) points.
    fig_size       : figure size tuple.
    title          : custom title; auto-generated if None.
    elev / azim    : initial 3-D viewing angle.
    """
    df = load_group(group, movement)
    _, mat_x, mat_y, mat_z = build_xyz(df, variable_base)

    n_trials = mat_x.shape[0]

    fig = plt.figure(figsize=fig_size)
    ax  = fig.add_subplot(111, projection="3d")
    _style_3d_ax(ax)

    # Faint individual trial lines
    first = True
    for x_row, y_row, z_row in zip(mat_x, mat_y, mat_z):
        ax.plot(x_row, y_row, z_row,
                color=_C["trial"],
                linewidth=trial_linewidth,
                alpha=trial_alpha,
                label="Individual trials" if first else None)
        first = False

    # Mean trajectory
    mx = mat_x.mean(axis=0)
    my = mat_y.mean(axis=0)
    mz = mat_z.mean(axis=0)

    ax.plot(mx, my, mz,
            color=mean_color,
            linewidth=mean_linewidth,
            label=f"{group_label}  (n={n_trials})" if show_n_trials else group_label,
            zorder=5)

    if show_start_end:
        ax.scatter(mx[0],  my[0],  mz[0],
                   color="green", s=60, zorder=6, label="Start")
        ax.scatter(mx[-1], my[-1], mz[-1],
                   color="red",   s=60, zorder=6, label="End")

    ax.set_xlabel(x_label if x_label is not None else "X (mm)", fontsize=int(10*text_size), color="0.3", labelpad=6)
    ax.set_ylabel(y_label if y_label is not None else "Y (mm)", fontsize=int(10*text_size), color="0.3", labelpad=6)
    ax.set_zlabel(z_label if z_label is not None else "Z (mm)", fontsize=int(10*text_size), color="0.3", labelpad=6)

    if title is None:
        title = f"{variable_base}  ·  3-D trajectory  ·  {group_label}"
    ax.set_title(title,
                 fontsize=int(12*text_size), fontweight="medium", pad=12)
    ax.text2D(0.0, 0.97,
              f"Mean ± individual trials  ·  {n_trials} trial{'s' if n_trials != 1 else ''}" if show_n_trials else "Mean ± individual trials",
              transform=ax.transAxes, fontsize=int(9*text_size), color="0.5")

    ax.legend(frameon=True, fontsize=int(9*text_size), loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.8)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING — 3-D trajectory comparison (Experts vs Novices)
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory_3d_comparison(
    variable_base:  str,
    movement:       str,
    trial_alpha:    float = 0.12,
    mean_linewidth: float = 2.5,
    trial_linewidth:float = 0.8,
    show_start_end: bool  = True,
    fig_size:       tuple = (13, 9),
    title:          str   = None,
    elev:           float = 20,
    azim:           float = 45,
    text_size:      float = 1.0,
    x_label:        str   = None,
    y_label:        str   = None,
    z_label:        str   = None,
    show_n_trials:  bool = False,
) -> None:
    """
    Overlay Expert and Novice mean 3-D trajectories with their individual
    faint trial lines, using the same colour palette as the 2-D comparison.

    Parameters
    ----------
    variable_base  : position variable WITHOUT the _X/_Y/_Z suffix,
                     e.g. "R_Hand_CoG_pos".
    trial_alpha    : opacity of the faint individual trial lines (0–1).
    mean_linewidth : line width of the mean trajectories.
    trial_linewidth: line width of the individual trial lines.
    show_start_end : mark the start (circle) and end (triangle) of each mean.
    fig_size       : figure size tuple.
    title          : custom title; auto-generated if None.
    elev / azim    : initial 3-D viewing angle.
    """
    df_exp = load_group(experts, movement)
    df_nov = load_group(novices, movement)

    _, ex_x, ex_y, ex_z = build_xyz(df_exp, variable_base)
    _, nv_x, nv_y, nv_z = build_xyz(df_nov, variable_base)

    fig = plt.figure(figsize=fig_size)
    ax  = fig.add_subplot(111, projection="3d")
    _style_3d_ax(ax)

    # ── Experts ──────────────────────────────────────────────────────────────
    first = True
    for x_row, y_row, z_row in zip(ex_x, ex_y, ex_z):
        ax.plot(x_row, y_row, z_row,
                color=_C["expert_light"],
                linewidth=trial_linewidth,
                alpha=trial_alpha,
                label="Expert trials" if first else None)
        first = False

    mx_e = ex_x.mean(axis=0)
    my_e = ex_y.mean(axis=0)
    mz_e = ex_z.mean(axis=0)
    ax.plot(mx_e, my_e, mz_e,
            color=_C["expert"],
            linewidth=mean_linewidth,
            label=f"Experts mean  (n={ex_x.shape[0]})" if show_n_trials else "Experts mean",
            zorder=5)
    if show_start_end:
        ax.scatter(mx_e[0],  my_e[0],  mz_e[0],
                   color=_C["expert"], s=60, marker="o", zorder=6)
        ax.scatter(mx_e[-1], my_e[-1], mz_e[-1],
                   color=_C["expert"], s=60, marker="^", zorder=6)

    # ── Novices ───────────────────────────────────────────────────────────────
    first = True
    for x_row, y_row, z_row in zip(nv_x, nv_y, nv_z):
        ax.plot(x_row, y_row, z_row,
                color=_C["novice_light"],
                linewidth=trial_linewidth,
                alpha=trial_alpha,
                label="Novice trials" if first else None)
        first = False

    mx_n = nv_x.mean(axis=0)
    my_n = nv_y.mean(axis=0)
    mz_n = nv_z.mean(axis=0)
    ax.plot(mx_n, my_n, mz_n,
            color=_C["novice"],
            linewidth=mean_linewidth,
            label=f"Novices mean  (n={nv_x.shape[0]})" if show_n_trials else "Novices mean",
            zorder=5)
    if show_start_end:
        ax.scatter(mx_n[0],  my_n[0],  mz_n[0],
                   color=_C["novice"], s=60, marker="o", zorder=6)
        ax.scatter(mx_n[-1], my_n[-1], mz_n[-1],
                   color=_C["novice"], s=60, marker="^", zorder=6)

    ax.set_xlabel(x_label if x_label is not None else "X (mm)", fontsize=int(10*text_size), color="0.3", labelpad=6)
    ax.set_ylabel(y_label if y_label is not None else "Y (mm)", fontsize=int(10*text_size), color="0.3", labelpad=6)
    ax.set_zlabel(z_label if z_label is not None else "Z (mm)", fontsize=int(10*text_size), color="0.3", labelpad=6)

    if title is None:
        title = f"{variable_base}  ·  3-D trajectory  ·  Experts vs Novices"
    ax.set_title(title,
                 fontsize=int(12*text_size), fontweight="medium", pad=12)
    ax.text2D(0.0, 0.97, "Mean + individual trials  ·  circle = start, triangle = end",
              transform=ax.transAxes, fontsize=int(9*text_size), color="0.5")

    ax.legend(frameon=True, fontsize=int(9*text_size), loc="upper right",
              framealpha=0.92, edgecolor="0.88", borderpad=0.8, ncol=2)
    ax.view_init(elev=elev, azim=azim)
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
AngMomcols = ['FullBody_AngMom_X','FullBody_AngMom_Y','FullBody_AngMom_Z','L_Hand_X','L_Hand_Y','L_Hand_Z','R_Hand_X','R_Hand_Y','R_Hand_Z','L_FA_X','L_FA_Y','L_FA_Z','R_FA_X','R_FA_Y','R_FA_Z','L_UA_X','L_UA_Y','L_UA_Z','R_UA_X','R_UA_Y','R_UA_Z','Head_X','Head_Y','Head_Z','Trunk_X','Trunk_Y','Trunk_Z','Pelvis_X','Pelvis_Y','Pelvis_Z','L_Thigh_X','L_Thigh_Y','L_Thigh_Z','R_Thigh_X','R_Thigh_Y','R_Thigh_Z','L_Shank_X','L_Shank_Y','L_Shank_Z','R_Shank_X','R_Shank_Y','R_Shank_Z','L_Foot_X','L_Foot_Y','L_Foot_Z','R_Foot_X','R_Foot_Y','R_Foot_Z']
AngMomBases = [
    'FullBody_AngMom',
    'L_Hand',
    'R_Hand',
    'L_FA',
    'R_FA',
    'L_UA',
    'R_UA',
    'Head',
    'Trunk',
    'Pelvis',
    'L_Thigh',
    'R_Thigh',
    'L_Shank',
    'R_Shank',
    'L_Foot',
    'R_Foot',
]
# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    # Example usage with movement argument
    movement = "elbow"  # or "roundhouse", "teep", etc.
    
       
    """    # Single group — all expert trials + mean
    plot_trajectory_3d(
        variable_base = "R_WRIST_POSITION",
        movement      = movement,
        group         = e1,
        group_label   = "Experts",
        mean_color    = _C["expert"],
    )
    plot_trajectory_3d(
        variable_base = "R_WRIST_POSITION",
        movement      = movement,
        group         = e2,
        group_label   = "Experts",
        mean_color    = _C["expert"],
    )
    plot_trajectory_3d(
        variable_base = "L_W_POSITION",
        movement      = movement,
        group         = e3,
        group_label   = "Experts",
        mean_color    = _C["expert"],
    )
    
    # Expert vs Novice comparison
    
    plot_trajectory_3d_comparison(
        variable_base = "R_WRIST_POSITION",
        movement      = movement,
        text_size= 1.3
    )
    plot_mean_sd_comparison(
        variable = "R_WRIST_POSITION_X",
        movement = movement,
        label = "Right hand X position",
        text_size = 1.3
    )
    
    for variable in AngMomcols:
        plot_mean_sd_comparison(
            variable= variable,
            movement = movement,
            label = variable,
            text_size = 1.3
    )
    """
    

    plot_mean_sd_comparison_norm(
            variable_base= "FullBody_AngMom",
            movement= "elbow",
            y_label= "Total body angular momentum kg$\cdot$m²/s",
            text_size=1.5,
    )
"""
        plot_mean_sd_FirstGradientComparison(
            variable_base= base,
            movement= movement,
            label= base,
            text_size=1.2,
        )
    
    
    
    plot_mean_sd_norm(
        variable_base = "R_Foot_CoG_vel",
        movement      = movement,
        group=experts,
    )

    plot_mean_sd_comparison_norm(
        variable_base = "L_",
        movement      = movement,
        label         = "Right Foot CoG Velocity (norm)",
        text_size= 1.7
    )

    analyze_norm_max_per_trial(
        variable_base = "R_Hand_CoG_vel",
        movement      = movement,
    )"""
    
    