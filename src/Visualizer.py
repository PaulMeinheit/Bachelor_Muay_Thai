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
  2. Replace the `load_data()` function body with your own data loading logic.
  3. Run:  python plot_biomechanics.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import BigLoader
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these
# ──────────────────────────────────────────────────────────────────────────────
experts = ["E1", "E2", "E3"]
novices = ["N1", "N2", "N3", "N4"]
# Variable to plot — must match a value at index level 2 of your DataFrame.
VARIABLE = "Pelvis_CoG_pos_Z"

# Shading opacity for the ± 1 SD band.
SHADE_ALPHA = 0.25

# Figure size in inches.
FIG_SIZE = (15, 7.5)

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
    Slice `variable` from the DataFrame and return:
        time_axis : 1-D array of frame indices  (n_frames,)
        matrix    : 2-D array of trial values   (n_trials, n_frames)

    Expects the MultiIndex (subject, trial, variable) on the columns
    and frame numbers as the row index.
    """
    try:
        var_df = df.xs(variable, level="variable", axis=1)
    except KeyError:
        available = df.columns.get_level_values("variable").unique().tolist()
        raise KeyError(
            f"Variable '{variable}' not found in the DataFrame.\n"
            f"Available variables: {available}"
        )

    if var_df.empty:
        raise ValueError(f"No data found for variable '{variable}'.")

    matrix    = var_df.to_numpy(dtype=float).T        # (n_trials, n_frames)
    time_axis = df.index.to_numpy(dtype=float)        # frame indices
    return time_axis, matrix


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING Single
# ──────────────────────────────────────────────────────────────────────────────

def plot_mean_sd(
    variable:    str,
    shade_alpha: float = 0.25,
    fig_size:    tuple = (10, 5),
    group: [str] = None,
     
) -> None:
 
    df                = load_group(group)
    time_axis, matrix = build_matrix(df, variable)
 
    n_trials = matrix.shape[0]
    mean     = matrix.mean(axis=0)
    std      = matrix.std(axis=0, ddof=1)
 
    fig, ax = plt.subplots(figsize=fig_size)
 
    # Individual trial traces (faint, for context)
    for row in matrix:
        ax.plot(time_axis, row, color="steelblue", alpha=0.12, linewidth=0.8, zorder=1)
 
    # ± 1 SD shaded band
    ax.fill_between(
        time_axis,
        mean - std,
        mean + std,
        color="steelblue",
        alpha=shade_alpha,
        linewidth=0,
        label=r"$\pm 1\,\sigma$",
        zorder=2,
    )
 
    # Mean line
    ax.plot(
        time_axis, mean,
        color="steelblue",
        linewidth=2.2,
        label="Mean",
        zorder=3,
    )
 
    # Aesthetics
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel(variable, fontsize=11)
    ax.set_title(
        f"{variable}  —  mean ± 1 SD  "
        f"({n_trials} trial{'s' if n_trials != 1 else ''})",
        fontsize=13,
        pad=10,
    )
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(which="major", color="0.90", linewidth=0.7)
    ax.grid(which="minor", color="0.95", linewidth=0.4)
 
    plt.tight_layout()
    plt.show()
# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING Comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_mean_sd_comparison(
    variable: str,
    shade_alpha: float = 0.25,
    fig_size: tuple = (10, 5),
) -> None:

    # Load both groups
    df_experts = load_group(experts)
    df_novices = load_group(novices)

    # Build matrices
    t_exp, mat_exp = build_matrix(df_experts, variable)
    t_nov, mat_nov = build_matrix(df_novices, variable)

    # --- Stats ---
    mean_exp = mat_exp.mean(axis=0)
    std_exp  = mat_exp.std(axis=0, ddof=1)

    mean_nov = mat_nov.mean(axis=0)
    std_nov  = mat_nov.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=fig_size)

    # --- Experts ---
    ax.fill_between(
        t_exp,
        mean_exp - std_exp,
        mean_exp + std_exp,
        color="red",
        alpha=shade_alpha,
        linewidth=0,
        label="Experts ±1 SD",
    )

    ax.plot(
        t_exp, mean_exp,
        color="red",
        linewidth=2.2,
        label="Experts Mean",
    )

    # --- Novices ---
    ax.fill_between(
        t_nov,
        mean_nov - std_nov,
        mean_nov + std_nov,
        color="blue",
        alpha=shade_alpha,
        linewidth=0,
        label="Novices ±1 SD",
    )

    ax.plot(
        t_nov, mean_nov,
        color="blue",
        linewidth=2.2,
        label="Novices Mean",
    )

    # Aesthetics
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel(variable, fontsize=11)
    ax.set_title(f"{variable} — Experts vs Novices", fontsize=13)

    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="major", color="0.90", linewidth=0.7)

    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_mean_sd_comparison(
        variable    = VARIABLE,
        shade_alpha = SHADE_ALPHA,
        fig_size    = FIG_SIZE,
    )
    plot_mean_sd(
        variable    = VARIABLE,
        shade_alpha = SHADE_ALPHA,
        fig_size    = FIG_SIZE,
        group       = experts,
    )