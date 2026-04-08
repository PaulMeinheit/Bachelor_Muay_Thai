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

X_VAR = "FullBody_CoG_pos_X"   # variable name for the fore-aft axis
Y_VAR = "FullBody_CoG_pos_Y"   # variable name for the lateral axis

# How many evenly-spaced frames at which to draw SD ellipses.
N_ELLIPSES = 0

# Opacity of individual trial traces.
TRIAL_ALPHA = 0.12

# Figure size in inches.
FIG_SIZE = (15, 10)
experts = ["E1", "E2", "E3"]
novices = ["N1", "N2", "N3", "N4"]
# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  ← replace with your own loading logic
# ──────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    data = BigLoader.loadallspecified(novices, "roundhouse")
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
    x_var:      str   = X_VAR,
    z_var:      str   = Y_VAR,
    n_ellipses: int   = N_ELLIPSES,
    trial_alpha: float = TRIAL_ALPHA,
    fig_size:   tuple = FIG_SIZE,
) -> None:

    df = load_data()

    X = get_variable(df, x_var)   # (n_frames, n_trials)
    Z = get_variable(df, z_var)   # (n_frames, n_trials)

    n_frames, n_trials = X.shape

    mean_x = X.mean(axis=1)       # (n_frames,)
    mean_z = Z.mean(axis=1)

    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=fig_size)

    # ── Individual trial traces ──────────────────────────────────────────────
    for i in range(n_trials):
        ax.plot(X[:, i], Z[:, i],
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
    ax.set_xlabel(x_var, fontsize=11)
    ax.set_ylabel(z_var, fontsize=11)
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

    plt.savefig("/home/paul/Schreibtisch/Bachelorarbeit/Bachelor_Muay_Thai/Plots/CoMgroundProjection/FullBodyComNovices.png", dpi=300)

    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_com_projection(
        x_var       = X_VAR,
        z_var       = Y_VAR,
        n_ellipses  = N_ELLIPSES,
        trial_alpha = TRIAL_ALPHA,
        fig_size    = FIG_SIZE,
    )