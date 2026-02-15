"""
Compute AMAC/AMOC (AMDP/AMOP) from *_AngMom_Segment_BodyCOM.csv exports.

Overview
--------
This script reads segment and full-body angular momentum vectors (about Body COM) from
CSV files produced by the processing pipeline, then computes two scalar metrics for
each segment group (e.g., RightArm, LeftLeg):

    AMDP (renamed for display as AMAC): component of group angular momentum parallel
    to the full-body angular momentum direction

        AMDP(t) = (S(t) · H_total(t)) / ||H_total(t)||

    AMOP (renamed for display as AMOC): magnitude of the component orthogonal to the
    full-body angular momentum direction

        AMOP(t) = ||S(t) × H_total(t)|| / ||H_total(t)||

where S(t) is the vector sum of angular momentum vectors for segments in the group.

Normalization
-------------
Optionally, AMDP and AMOP are normalized by (mass * height^2). With mass in kg and
height in meters, AMDP_norm and AMOP_norm have units of s^-1.

Time Normalization
------------------
Each trial can be resampled to 0–100% (101 points) using linear interpolation to
support across-trial comparisons.

Outputs
-------
For each trial block:
- Excel workbook (.xlsx): one sheet per group; optional per-segment sheets if enabled
- Summary CSV (_summary.csv): descriptive statistics for each group and metric
- PNG plots per group and metric; optional segment overlay plots if enabled

Usage examples
--------------
python amdp_amop_from_csv.py --root "E:/theia_data_fp/c3d" --outdir "E:/amdp_amop_out"
python amdp_amop_from_csv.py --root "..." --outdir "..." --ylims_mode minmax --no_time_norm
python amdp_amop_from_csv.py --root "..." --outdir "..." --overlay_segments

Notes
-----
- The script assumes each vector variable is stored in three consecutive columns
  (x, y, z) and repeated across blocks within the same CSV.
- All vectors used in dot/cross products must be expressed in the same coordinate
  frame and reference (here: *_wrt_BodyCOM).
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Path parsing helper (used for plot titles and output folder structure)
# ---------------------------------------------------------------------

PATH_RE = re.compile(
    r"(?i)(?P<subject>Subject[^\\/]+)[\\/](?P<move>[^\\/]+)[\\/](?P<trial>[^\\/]+)[\\/][^\\/]*\.c3d"
)

def parse_subject_move_trial(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract Subject/Move/Trial from a file path string that contains a .c3d filename.
    Returns (subject, move, trial) or (None, None, None) if not matched.
    """
    if isinstance(s, str):
        m = PATH_RE.search(s)
        if m:
            return m.group("subject"), m.group("move"), m.group("trial")
    return None, None, None


# ---------------------------------------------------------------------
# Column indexing utilities
# ---------------------------------------------------------------------

def build_triplet_index_map(names_row1: List[str]) -> Dict[str, List[List[int]]]:
    """
    Build a mapping from variable base name -> list of [ix, iy, iz] triplets.

    The CSV is assumed to contain repeated blocks where the same variable names appear
    multiple times; each occurrence corresponds to one block (e.g., a trial instance).

    Parameters
    ----------
    names_row1 : List[str]
        The variable-name row from the CSV (row index 1 in the current format).

    Returns
    -------
    Dict[str, List[List[int]]]
        base_name -> [[i0, i1, i2], [i3, i4, i5], ...]
    """
    by_name: Dict[str, List[int]] = {}
    for j, nm in enumerate(names_row1):
        by_name.setdefault(nm, []).append(j)

    trip_map: Dict[str, List[List[int]]] = {}
    for base, idxs in by_name.items():
        idxs_sorted = sorted(idxs)

        # Keep only complete x/y/z sets
        usable = len(idxs_sorted) - (len(idxs_sorted) % 3)
        idxs_sorted = idxs_sorted[:usable]

        triplets = [idxs_sorted[k:k+3] for k in range(0, len(idxs_sorted), 3)]
        if triplets:
            trip_map[base] = triplets

    return trip_map


def default_groups() -> Dict[str, List[str]]:
    """Default segment groupings for upper limbs, lower limbs, and trunk."""
    return {
        "RightArm": ["R_UA_AngMom_wrt_BodyCOM", "R_FA_AngMom_wrt_BodyCOM", "R_Hand_AngMom_wrt_BodyCOM"],
        "LeftArm":  ["L_UA_AngMom_wrt_BodyCOM", "L_FA_AngMom_wrt_BodyCOM", "L_Hand_AngMom_wrt_BodyCOM"],
        "RightLeg": ["R_Thigh_AngMom_wrt_BodyCOM", "R_Shank_AngMom_wrt_BodyCOM", "R_Foot_AngMom_wrt_BodyCOM"],
        "LeftLeg":  ["L_Thigh_AngMom_wrt_BodyCOM", "L_Shank_AngMom_wrt_BodyCOM", "L_Foot_AngMom_wrt_BodyCOM"],
        "Trunk":    ["Torso_AngMom_wrt_BodyCOM", "Pelvis_AngMom_wrt_BodyCOM", "Head_AngMom_wrt_BodyCOM"],
    }


def get_segment_display_name(segment_name: str) -> str:
    """Convert internal segment variable names to human-readable labels."""
    name_map = {
        "R_UA_AngMom_wrt_BodyCOM": "R Upper Arm",
        "R_FA_AngMom_wrt_BodyCOM": "R Forearm",
        "R_Hand_AngMom_wrt_BodyCOM": "R Hand",
        "L_UA_AngMom_wrt_BodyCOM": "L Upper Arm",
        "L_FA_AngMom_wrt_BodyCOM": "L Forearm",
        "L_Hand_AngMom_wrt_BodyCOM": "L Hand",
        "R_Thigh_AngMom_wrt_BodyCOM": "R Thigh",
        "R_Shank_AngMom_wrt_BodyCOM": "R Shank",
        "R_Foot_AngMom_wrt_BodyCOM": "R Foot",
        "L_Thigh_AngMom_wrt_BodyCOM": "L Thigh",
        "L_Shank_AngMom_wrt_BodyCOM": "L Shank",
        "L_Foot_AngMom_wrt_BodyCOM": "L Foot",
        "Torso_AngMom_wrt_BodyCOM": "Torso",
        "Pelvis_AngMom_wrt_BodyCOM": "Pelvis",
        "Head_AngMom_wrt_BodyCOM": "Head",
    }
    return name_map.get(segment_name, segment_name)


def get_metric_display_name(metric: str) -> str:
    """
    Convert internal metric names to display names with units.

    AMDP -> AMAC label for display
    AMOP -> AMOC label for display
    """
    if metric == "AMDP_norm":
        return "AMAC - Normalized Angular Momentum (s⁻¹)"
    if metric == "AMOP_norm":
        return "AMOC - Normalized Angular Momentum (s⁻¹)"
    if metric == "AMDP":
        return "AMAC (kg·m²/s)"
    if metric == "AMOP":
        return "AMOC (kg·m²/s)"
    return metric


# ---------------------------------------------------------------------
# Time normalization
# ---------------------------------------------------------------------

def resample_df_to_percent(df: pd.DataFrame, n_points: int = 101) -> pd.DataFrame:
    """
    Resample each numeric column to a 0–100% timeline using linear interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data per frame.
    n_points : int
        Number of samples between 0 and 100% inclusive.

    Returns
    -------
    pd.DataFrame
        Resampled data with a FramePct column in [0,100].
    """
    if len(df) < 2:
        out = df.copy()
        out.insert(0, "FramePct", 0.0)
        return out

    x = np.linspace(0.0, 1.0, len(df))
    xp = np.linspace(0.0, 1.0, n_points)

    out = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            y = df[col].to_numpy()
            y = pd.Series(y).interpolate("linear", limit_direction="both").to_numpy()
            out[col] = np.interp(xp, x, y)

    res = pd.DataFrame(out)
    res.insert(0, "FramePct", xp * 100.0)
    return res


# ---------------------------------------------------------------------
# Global y-axis limits support
# ---------------------------------------------------------------------

def compute_global_ylims(
    all_results_for_file,
    mode: str = "percentile",
    low_pct: float = 1.0,
    high_pct: float = 99.9,
    smart_cap_tol: float = 0.05
):
    """
    Compute shared y-axis limits per (Group, Metric) across all blocks within a file.

    Parameters
    ----------
    all_results_for_file : list of (group_name, df)
        Each df contains AMDP/AMOP and optional normalized columns.
    mode : {"percentile","minmax"}
        'percentile' is robust to outliers; 'minmax' uses true range with padding.
    low_pct, high_pct : float
        Percentiles used in percentile mode.
    smart_cap_tol : float
        If the true max is within (1+tol) of the upper percentile, use the true max
        to avoid clipping near-peaks.

    Returns
    -------
    dict
        {(group, metric): (ymin, ymax)}
    """
    values = defaultdict(list)

    for g, df in all_results_for_file:
        for metric in df.columns:
            if metric == "FramePct":
                continue
            v = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
            if v.size:
                values[(g, metric)].append(v)

    ylims = {}
    for key, chunks in values.items():
        v = np.concatenate(chunks)
        vmin, vmax = float(np.min(v)), float(np.max(v))

        if mode == "minmax":
            rng = vmax - vmin
            pad = 0.05 * (rng if rng > 0 else max(1e-6, abs(vmax)))
            ylims[key] = (vmin - pad, vmax + pad)
        else:
            lo = float(np.percentile(v, low_pct))
            hi_p = float(np.percentile(v, high_pct))
            hi = vmax if vmax <= hi_p * (1.0 + smart_cap_tol) else hi_p

            if lo == hi:
                pad = max(1e-6, abs(lo) * 0.05)
                lo, hi = lo - pad, hi + pad

            ylims[key] = (lo, hi)

    return ylims


# ---------------------------------------------------------------------
# AMDP/AMOP computation
# ---------------------------------------------------------------------

def compute_amdp_amop_from_block(
    raw_data: pd.DataFrame,
    trip_map: Dict[str, List[List[int]]],
    block_idx: int,
    groups: Dict[str, List[str]],
    eps: float = 1e-12,
    norm_denom: Optional[float] = None,
    include_individual_segments: bool = False,

    htot_min_norm: Optional[float] = None
) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, Dict[str, pd.DataFrame]]]]:
    """
    Compute AMDP/AMOP for each segment group in one block.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Numeric data rows from the CSV (frames x columns).
    trip_map : dict
        base_name -> list of [ix,iy,iz] triplets across blocks.
    block_idx : int
        Which block to compute.
    groups : dict
        group_name -> list of segment base_names.
    eps : float
        Small number added to ||H_total|| to avoid divide-by-zero.
    norm_denom : float, optional
        If provided, append AMDP_norm and AMOP_norm as AMDP/(m*h^2) and AMOP/(m*h^2).
    include_individual_segments : bool
        If True, compute per-segment AMDP/AMOP (and normalized) for overlay plots.
    htot_min_norm : float, optional
        If provided, frames with ||H_total|| < threshold are set to NaN in outputs.

    Returns
    -------
    results : dict
        group_name -> DataFrame with AMDP/AMOP and optional normalized columns.
    segment_results : dict or None
        If include_individual_segments: group_name -> {segment_name -> DataFrame}
    """
    fb = "FullBody_AngMom_wrt_BodyCOM"
    if fb not in trip_map or block_idx >= len(trip_map[fb]):
        raise ValueError("FullBody_AngMom_wrt_BodyCOM not available for this block.")

    fcols = trip_map[fb][block_idx]
    Htot = raw_data.iloc[:, fcols].to_numpy()
    Htot_norm_raw = np.linalg.norm(Htot, axis=1)


    Htot_norm = Htot_norm_raw + eps

    # Optional masking (disabled by default)
    mask_small = None
    if htot_min_norm is not None:
        mask_small = Htot_norm_raw < htot_min_norm

    results: Dict[str, pd.DataFrame] = {}
    segment_results: Dict[str, Dict[str, pd.DataFrame]] = {} if include_individual_segments else None

    for gname, parts in groups.items():
        S = None
        segment_data = {} if include_individual_segments else None

        for base in parts:
            if base not in trip_map or block_idx >= len(trip_map[base]):
                S = None
                break

            cols = trip_map[base][block_idx]
            v = raw_data.iloc[:, cols].to_numpy()

            if include_individual_segments:
                seg_dot = np.sum(v * Htot, axis=1)
                seg_cross_mag = np.linalg.norm(np.cross(v, Htot), axis=1)

                seg_amdp = seg_dot / Htot_norm
                seg_amop = seg_cross_mag / Htot_norm

                if mask_small is not None and mask_small.any():
                    seg_amdp = seg_amdp.astype(float)
                    seg_amop = seg_amop.astype(float)
                    seg_amdp[mask_small] = np.nan
                    seg_amop[mask_small] = np.nan

                seg_df = pd.DataFrame({"AMDP": seg_amdp, "AMOP": seg_amop})

                if norm_denom and norm_denom > 0:
                    seg_df["AMDP_norm"] = seg_df["AMDP"] / norm_denom
                    seg_df["AMOP_norm"] = seg_df["AMOP"] / norm_denom

                segment_data[base] = seg_df

            S = v if S is None else (S + v)

        if S is None:
            continue

        dot = np.sum(S * Htot, axis=1)
        cross_mag = np.linalg.norm(np.cross(S, Htot), axis=1)

        amdp = dot / Htot_norm
        amop = cross_mag / Htot_norm

        if mask_small is not None and mask_small.any():
            amdp = amdp.astype(float)
            amop = amop.astype(float)
            amdp[mask_small] = np.nan
            amop[mask_small] = np.nan

        df = pd.DataFrame({"AMDP": amdp, "AMOP": amop})

        if norm_denom and norm_denom > 0:
            df["AMDP_norm"] = df["AMDP"] / norm_denom
            df["AMOP_norm"] = df["AMOP"] / norm_denom

        results[gname] = df

        if include_individual_segments:
            segment_results[gname] = segment_data

    return results, segment_results


# ---------------------------------------------------------------------
# File discovery and subject metadata
# ---------------------------------------------------------------------

def gather_files(root: Optional[str], input_glob: Optional[str]) -> List[str]:
    """Find all *_AngMom_Segment_BodyCOM.csv files from a root directory and/or a glob."""
    files: List[str] = []
    if root:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith("_AngMom_Segment_BodyCOM.csv"):
                    files.append(os.path.join(dirpath, fn))
    if input_glob:
        files.extend(glob.glob(input_glob, recursive=True))
    return sorted(set(files))


def get_subject_mass_height(subject: str, known: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """
    Prompt for subject mass and height once per subject, then cache values.

    Height may be entered in meters or centimeters (values > 10 are interpreted as cm).
    """
    if subject in known:
        return known[subject]

    print(f"\n--- Subject info required for {subject} ---")
    while True:
        try:
            mass = float(input(f"Enter mass [kg] for {subject}: ").strip())
            height = float(input(f"Enter height [m or cm] for {subject}: ").strip())
            if height > 10.0:  # interpret as cm
                height /= 100.0
            if mass > 0 and height > 0:
                known[subject] = (mass, height)
                return mass, height
        except ValueError:
            pass
        print("Invalid input. Example: mass=72.5, height=1.78 (or 178).")


def ask_for_segment_overlay() -> bool:
    """Interactive prompt to enable/disable per-segment overlay plots."""
    print("\n" + "=" * 60)
    print("SEGMENT OVERLAY OPTION")
    print("=" * 60)
    print("Create overlay plots showing individual segments within each group?")
    while True:
        response = input("Create segment overlay plots? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


# ---------------------------------------------------------------------
# Output naming and plotting
# ---------------------------------------------------------------------

def _infer_context_from_base(base_outpath: str) -> str:
    """
    Infer 'Subject • Move • Trial' from output path structure:
        .../<Subject>/<Move>/<Trial>/...
    """
    parts = os.path.dirname(base_outpath).replace("\\", "/").split("/")
    if len(parts) >= 3:
        subj, move, trial = parts[-3], parts[-2], parts[-1]
        bits = [b for b in (subj, move, trial) if b and str(b).strip()]
        if bits:
            return " • ".join(bits)
    return ""


def resample_and_save(
    base_outpath: str,
    results: Dict[str, pd.DataFrame],
    segment_results: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ylims: dict = None,
    time_to_percent: bool = True,
    n_percent_points: int = 101,
    subject: str = "",
    move: str = "",
    trial: str = ""
) -> None:
    """
    Save processed data and plots for one block.

    Writes:
    - base_outpath + ".xlsx"
    - base_outpath + "_summary.csv"
    - PNG plots in the same directory as base_outpath
    """
    # Resample group results
    proc = {}
    for g, df in results.items():
        proc[g] = resample_df_to_percent(df, n_percent_points) if time_to_percent else df.copy()

    # Resample segment results (optional)
    proc_segments = None
    if segment_results:
        proc_segments = {}
        for g, seg_dict in segment_results.items():
            proc_segments[g] = {}
            for seg_name, seg_df in seg_dict.items():
                proc_segments[g][seg_name] = (
                    resample_df_to_percent(seg_df, n_percent_points)
                    if time_to_percent else seg_df.copy()
                )

    # Excel output
    with pd.ExcelWriter(base_outpath + ".xlsx") as w:
        for g, df in proc.items():
            df.to_excel(w, sheet_name=g, index=False)

        if proc_segments:
            for g, seg_dict in proc_segments.items():
                for seg_name, seg_df in seg_dict.items():
                    sheet_name = f"{g}_{get_segment_display_name(seg_name)}"[:31]
                    seg_df.to_excel(w, sheet_name=sheet_name, index=False)

    # Summary CSV
    rows = []
    for g, df in proc.items():
        cols = [c for c in df.columns if c != "FramePct"]
        for metric in cols:
            s = df[metric].describe()
            rows.append({
                "Group": g, "Metric": metric,
                "N": int(s["count"]),
                "Mean": float(s["mean"]),
                "Std": float(s["std"]),
                "Min": float(s["min"]),
                "25%": float(s["25%"]),
                "50%": float(s["50%"]),
                "75%": float(s["75%"]),
                "Max": float(s["max"]),
            })
    pd.DataFrame(rows).to_csv(base_outpath + "_summary.csv", index=False)

    # Title context
    ctx = " • ".join([x for x in (subject, move, trial) if x and str(x).strip()]) or _infer_context_from_base(base_outpath)

    # Group plots
    for g, df in proc.items():
        x = df["FramePct"] if "FramePct" in df else df.index
        xlab = "Normalized time (%)" if "FramePct" in df else "Frame (raw)"

        for metric in [c for c in df.columns if c != "FramePct"]:
            plt.figure(figsize=(10, 6))
            plt.plot(x, df[metric], linewidth=2)
            plt.xlabel(xlab, fontsize=11)
            plt.ylabel(get_metric_display_name(metric), fontsize=11)

            title = f"{ctx}\n{g} – {get_metric_display_name(metric)}" if ctx else f"{g} – {get_metric_display_name(metric)}"
            plt.title(title)

            if ylims and (g, metric) in ylims:
                plt.ylim(*ylims[(g, metric)])

            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            outpath = os.path.join(os.path.dirname(base_outpath), f"{g}_{metric}.png")
            plt.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close()

    # Segment overlay plots (normalized metrics only)
    if proc_segments:
        for g, seg_dict in proc_segments.items():
            if not seg_dict:
                continue

            first_seg = next(iter(seg_dict.values()))
            x = first_seg["FramePct"] if "FramePct" in first_seg else first_seg.index
            xlab = "Normalized time (%)" if "FramePct" in first_seg else "Frame (raw)"

            for metric in ["AMDP_norm", "AMOP_norm"]:
                # Only plot if at least one segment has this metric
                if not any(metric in seg_df.columns for seg_df in seg_dict.values()):
                    continue

                fig, ax = plt.subplots(figsize=(12, 7))

                for seg_name, seg_df in seg_dict.items():
                    if metric in seg_df.columns:
                        ax.plot(
                            x,
                            seg_df[metric],
                            label=get_segment_display_name(seg_name),
                            linewidth=2,
                            alpha=0.8
                        )

                # Overlay group sum if available
                if g in proc and metric in proc[g].columns:
                    ax.plot(
                        x,
                        proc[g][metric],
                        label=f"{g} (Sum)",
                        linewidth=3,
                        linestyle="--",
                        color="black",
                        alpha=0.9
                    )

                ax.set_xlabel(xlab, fontsize=11)
                ax.set_ylabel(get_metric_display_name(metric), fontsize=11)

                title = f"{ctx}\n{g} – {get_metric_display_name(metric)} (Segment Breakdown)" if ctx else \
                        f"{g} – {get_metric_display_name(metric)} (Segment Breakdown)"
                ax.set_title(title, fontsize=12)

                ax.legend(loc="best", frameon=True, fontsize=9)
                ax.grid(True, alpha=0.3)

                if ylims and (g, metric) in ylims:
                    ax.set_ylim(*ylims[(g, metric)])

                plt.tight_layout()
                outpath = os.path.join(os.path.dirname(base_outpath), f"{g}_{metric}_SEGMENTS.png")
                plt.savefig(outpath, dpi=150, bbox_inches="tight")
                plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute AMDP/AMOP (displayed as AMAC/AMOC), with optional mass·height² normalization and time resampling."
    )
    ap.add_argument("--root", help="Root folder to search recursively.")
    ap.add_argument("--input", help="Alternative: glob like Subject*/**/*_AngMom_Segment_BodyCOM.csv")
    ap.add_argument("--outdir", required=True, help="Output base folder")

    ap.add_argument("--no_time_norm", action="store_true", help="Disable 0–100% time normalization")
    ap.add_argument("--ylims_mode", choices=["percentile", "minmax"], default="percentile",
                    help="Global y-limits strategy")
    ap.add_argument("--low_pct", type=float, default=1.0, help="Lower percentile (percentile mode)")
    ap.add_argument("--high_pct", type=float, default=99.9, help="Upper percentile (percentile mode)")
    ap.add_argument("--smart_cap_tol", type=float, default=0.05, help="Cap-to-max tolerance (percentile mode)")

    ap.add_argument("--overlay_segments", action="store_true", help="Create overlay plots showing individual segments")
    args = ap.parse_args()

    files = gather_files(args.root, args.input)
    if not files:
        raise SystemExit("No files found. Use --root or --input.")

    os.makedirs(args.outdir, exist_ok=True)
    groups = default_groups()
    subj_info: Dict[str, Tuple[float, float]] = {}

    include_segments = args.overlay_segments or ask_for_segment_overlay()

    for fpath in files:
        print(f"\nProcessing file: {fpath}")
        raw = pd.read_csv(fpath, header=None)

        row0 = list(raw.iloc[0])       # source paths
        names_row1 = list(raw.iloc[1]) # variable names

        data = raw.iloc[3:].reset_index(drop=True)
        data = data.apply(pd.to_numeric, errors="coerce")

        trip_map = build_triplet_index_map(names_row1)

        fb = "FullBody_AngMom_wrt_BodyCOM"
        if fb not in trip_map:
            print("  WARNING: FullBody_AngMom_wrt_BodyCOM not found; skipping.")
            continue

        num_blocks = len(trip_map[fb])
        print(f"  Found {num_blocks} block(s) in this file.")

        all_results_for_file = []
        per_block_results = []

        for b in range(num_blocks):
            fb_cols = trip_map[fb][b]
            path_hint = None
            for ci in fb_cols:
                if ci < len(row0) and isinstance(row0[ci], str) and row0[ci]:
                    path_hint = row0[ci]
                    break

            subject, move, trial = parse_subject_move_trial(path_hint or fpath)
            subject = subject or "Subject_Unknown"
            move = move or "Move_Unknown"
            trial = trial or f"Block_{b+1:02d}"

            mass, height = get_subject_mass_height(subject, subj_info)
            norm_denom = mass * (height ** 2)
            print(f"  Block {b+1}/{num_blocks}: {subject} / {move} / {trial} | m·h² = {norm_denom:.4f} kg·m²")

            try:
                results, segment_results = compute_amdp_amop_from_block(
                    data,
                    trip_map,
                    b,
                    groups,
                    norm_denom=norm_denom,
                    include_individual_segments=include_segments
                )
            except Exception as e:
                print(f"    ERROR computing block {b+1}: {e}")
                continue

            if not results:
                print("    No groups produced (missing segment data). Skipping block.")
                continue

            per_block_results.append((subject, move, trial, results, segment_results))
            for g, df in results.items():
                all_results_for_file.append((g, df))

        if not per_block_results:
            continue

        ylims = compute_global_ylims(
            all_results_for_file,
            mode=args.ylims_mode,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            smart_cap_tol=args.smart_cap_tol
        )

        for subject, move, trial, results, segment_results in per_block_results:
            outdir = os.path.join(args.outdir, subject, move, trial)
            os.makedirs(outdir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(fpath))[0]
            base_outpath = os.path.join(outdir, base_name + "_AMDP_AMOP")

            resample_and_save(
                base_outpath,
                results,
                segment_results=segment_results,
                ylims=ylims,
                time_to_percent=not args.no_time_norm,
                n_percent_points=101,
                subject=subject,
                move=move,
                trial=trial
            )

            print(f"    -> wrote outputs in {outdir}")


if __name__ == "__main__":
    main()
