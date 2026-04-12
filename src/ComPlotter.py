"""
CoM ground projection visualiser
==================================
Plots the centre-of-mass trajectory projected onto the ground plane (X-Z)
across all subjects and trials.

For each frame the script computes:
  • the mean (X, Z) position across all trials  → drawn as a solid path
  • the 2-D covariance of (X, Z) across trials  → drawn as a 1-SD ellipse

Individual trial traces are shown as faint lines in the background.
A colourbar encodes normalised time (0 – 100 % of trial) so you can
read the direction of motion from the mean path.

DataFrame structure expected
-----------------------------
  Rows    : integer frame index
  Columns : MultiIndex  (subject, trial, variable)
             where variable includes the names set in X_VAR and Z_VAR below.

Usage
------
  1. Set X_VAR / Z_VAR to your horizontal-plane position column names.
  2. Replace load_data() with your real loading logic.
  3. Run:  python plot_com_projection.py
"""
import BigLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these
# ──────────────────────────────────────────────────────────────────────────────


experts = ["E1", "E2", "E3"]
novices = ["N1", "N2", "N3", "N4"]
# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  ← replace with your own loading logic
# ──────────────────────────────────────────────────────────────────────────────

def load_data(group,movement) -> pd.DataFrame:
    data = BigLoader.loadallspecified(group, movement)
    data.columns.names = ["subject", "trial", "variable"]
    return data

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_variable(df: pd.DataFrame, variable: str) -> np.ndarray:
    """
    Return a (n_frames, n_trials) array for `variable`.
    Raises a clear KeyError listing available variables when not found.
    """
    try:
        sub = df.xs(variable, level="variable", axis=1)
    except KeyError:
        available = df.columns.get_level_values("variable").unique().tolist()
        raise KeyError(
            f"Variable '{variable}' not found.\n"
            f"Available variables: {available}"
        )
    return sub.to_numpy(dtype=float)   # (n_frames, n_trials)

def make_coloured_segments(x: np.ndarray, z: np.ndarray, norm: mcolors.Normalize,
                           cmap) -> mcoll.LineCollection:
    """
    Build a LineCollection for a single (x, z) path where each segment is
    coloured by its normalised position along the path.
    """
    points = np.column_stack([x, z]).reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    t_vals = np.linspace(0, 1, len(x))
    lc     = mcoll.LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.8, zorder=3)
    lc.set_array(t_vals[:-1])
    return lc


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_com_projection(
    group,
    movement,
    x_var,
    y_var,
    x_Name,
    y_Name,
    n_ellipses,
    trial_alpha,
    fig_size,
):

    df = load_data(group, movement)

    X = get_variable(df, x_var)   # (n_frames, n_trials)
    Y = get_variable(df, y_var)   # (n_frames, n_trials)

    n_frames, n_trials = X.shape

    mean_x = X.mean(axis=1)       # (n_frames,)
    mean_z = Y.mean(axis=1)

    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=fig_size)

    # ── Individual trial traces ──────────────────────────────────────────────
    for i in range(n_trials):
        ax.plot(X[:, i], Y[:, i],
                color="steelblue", alpha=trial_alpha,
                linewidth=0.8, zorder=1)

    # ── Mean path (time-coloured) ────────────────────────────────────────────
    lc = make_coloured_segments(mean_x, mean_z, norm, cmap)
    ax.add_collection(lc)

    # ── Start / end markers on mean path ────────────────────────────────────
    ax.plot(*[mean_x[0],  mean_z[0]],  "o", color=cmap(0.0), markersize=7,
            zorder=4, label="Start")
    ax.plot(*[mean_x[-1], mean_z[-1]], "s", color=cmap(1.0), markersize=7,
            zorder=4, label="End")

    # ── Aesthetics ───────────────────────────────────────────────────────────
    ax.set_xlabel(x_Name, fontsize=11)
    ax.set_ylabel(y_Name, fontsize=11)
    ax.set_title(
        f"CoM ground projection\n"
        f"({n_trials} trials)",
        fontsize=12, pad=10,
    )
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=10, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(color="0.92", linewidth=0.6)

    plt.tight_layout()

    #plt.savefig("/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/Plots/PlotsToShow/ElbowPositionExperts.png", dpi=300)

    plt.show()


def plot_com_projection_comparison_clean(
    movement,
    x_var,
    y_var,
    x_Name,
    y_Name,
    fig_size=(12, 12),
    expert_color="crimson",
    novice_color="steelblue",
):
    """
    Plot CoM ground projection for both Experts and Novices in the same figure.
    Shows only the mean paths (no individual trial traces).
    
    Each group's mean path is time-coloured (0 = start, 1 = end).
    
    Parameters
    ----------
    movement      : movement type string (e.g., "roundhouse", "teep")
    x_var         : column name for X-axis variable
    y_var         : column name for Z-axis (vertical) variable
    x_Name        : label for X-axis
    y_Name        : label for Y-axis
    fig_size      : figure size tuple
    expert_color  : base color for expert path
    novice_color  : base color for novice path
    """
    # Load data for both groups
    df_exp = load_data(experts, movement)
    df_nov = load_data(novices, movement)
    
    # Extract X, Y data
    X_exp = get_variable(df_exp, x_var)   # (n_frames, n_trials_exp)
    Y_exp = get_variable(df_exp, y_var)
    
    X_nov = get_variable(df_nov, x_var)   # (n_frames, n_trials_nov)
    Y_nov = get_variable(df_nov, y_var)
    
    n_frames_exp, n_trials_exp = X_exp.shape
    n_frames_nov, n_trials_nov = X_nov.shape
    
    # Compute mean paths
    mean_x_exp = X_exp.mean(axis=1)
    mean_z_exp = Y_exp.mean(axis=1)
    
    mean_x_nov = X_nov.mean(axis=1)
    mean_z_nov = Y_nov.mean(axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # ── Experts mean path (solid color) ──────────────────────────────────────
    ax.plot(mean_x_exp, mean_z_exp, color="#C0392B", linewidth=2.8, 
            zorder=3, label=f"Experts  (n={n_trials_exp})")
    
    # Start/end markers for experts
    ax.plot(mean_x_exp[0],  mean_z_exp[0],  "o", color="#C0392B", 
            markersize=8, zorder=4, markeredgecolor="darkred", markeredgewidth=1.5)
    ax.plot(mean_x_exp[-1], mean_z_exp[-1], "s", color="#B33224", 
            markersize=8, zorder=4, markeredgecolor="darkred", markeredgewidth=1.5)
    
    # ── Novices mean path (solid color) ──────────────────────────────────────
    ax.plot(mean_x_nov, mean_z_nov, color="#2874A6", linewidth=2.8, 
            zorder=3, label=f"Novices  (n={n_trials_nov})")
    
    # Start/end markers for novices
    ax.plot(mean_x_nov[0],  mean_z_nov[0],  "o", color="#2874A6", 
            markersize=8, zorder=4, markeredgecolor="darkblue", markeredgewidth=1.5)
    ax.plot(mean_x_nov[-1], mean_z_nov[-1], "s", color="#2874A6", 
            markersize=8, zorder=4, markeredgecolor="darkblue", markeredgewidth=1.5)
    
    # ── Legend ───────────────────────────────────────────────────────────────
    # Get all lines to identify the mean paths
    all_lines = ax.get_lines()
    expert_line = all_lines[0]    # Expert mean path (first plot call)
    novice_line = all_lines[3]    # Novice mean path (fourth plot call, after 3 markers)
    
    start_marker = plt.Line2D([0], [0], marker="o", color="w", 
                              markerfacecolor="gray", markersize=7, label="Start")
    end_marker   = plt.Line2D([0], [0], marker="s", color="w", 
                              markerfacecolor="gray", markersize=7, label="End")
    
    ax.legend(handles=[expert_line, novice_line, start_marker, end_marker],
              frameon=True, fontsize=10, loc="best", framealpha=0.95)
    
    # ── Aesthetics ───────────────────────────────────────────────────────────
    ax.set_xlabel(x_Name, fontsize=12, fontweight="medium")
    ax.set_ylabel(y_Name, fontsize=12, fontweight="medium")
    ax.set_title(
        f"CoM Ground Projection: Experts vs Novices  ·  {movement.capitalize()}",
        fontsize=13, fontweight="medium", pad=12,
    )
    ax.set_aspect("equal")
    
    # ── Enforce square plot by equalizing axis ranges ──────────────────────
    ax.autoscale()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    
    # Center the limits and make them equal
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(color="0.92", linewidth=0.6, alpha=0.8)
    
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

X_VAR = "Pelvis_CoG_pos_X"   # variable name for the fore-aft axis
Y_VAR = "Pelvis_CoG_pos_Y"   # variable name for the vertical axis
# How many evenly-spaced frames at which to draw SD ellipses.
N_ELLIPSES = 0
# Opacity of individual trial traces.
TRIAL_ALPHA = 0.2
# Figure size in inches.
FIG_SIZE = (15, 10)
GROUP = experts
MOVEMENT = "teep"

if __name__ == "__main__":
    plot_com_projection(
        group       = GROUP,
        movement    = MOVEMENT,
        x_var       = X_VAR,
        y_var       = Y_VAR,
        x_Name      = X_VAR,
        y_Name      = Y_VAR,
        n_ellipses  = N_ELLIPSES,
        trial_alpha = TRIAL_ALPHA,
        fig_size    = FIG_SIZE,

    )
    plot_com_projection_comparison_clean(
        movement    = MOVEMENT,
        x_var       = X_VAR,
        y_var       = Y_VAR,
        x_Name      = X_VAR,
        y_Name      = Y_VAR,
        fig_size    = FIG_SIZE,
    )