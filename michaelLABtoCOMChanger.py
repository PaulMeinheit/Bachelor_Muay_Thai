"""
==============================================================
Angular Momentum Conversion: LAB → Body COM
==============================================================

This script converts segment angular momentum signals from
"angular momentum about the LAB/origin" to 
"angular momentum about the Whole-Body Center of Mass (Body COM)".

--------------------------------------------------------------
Required Inputs
--------------------------------------------------------------

--ang <file>
    CSV file containing the angular momentum of each segment
    with respect to the LAB origin. 
    These signals typically come from Visual3D and look like:
        RTH_AngMom_wrt_LAB_X, Y, Z
        LTH_AngMom_wrt_LAB_X, Y, Z
        ...
        FullBody_AngMom

--cog <file>
    CSV file containing segment center-of-mass positions (x,y,z)
    for every segment in the same order as the angular momentum file.
    Example columns:
        RTH_COG_X, Y, Z
        LTH_COG_X, Y, Z
        ...

--vel <file>
    CSV file containing segment center-of-mass velocities (x,y,z)
    for every segment. Should correspond exactly to the COG file.

--mass <float>
    Total subject body mass in kilograms.
    This is required because segment masses are computed using
    De Leva’s anthropometric mass fractions.

--sex <male|female>
    Specifies whether male or female De Leva mass fractions are used.
    This affects the distribution of segment masses.

--out <file>  (optional)
    Output CSV file path.
    If omitted, the script automatically creates a filename such as:
        <input>_AngMom_wrt_BodyCOM.csv
    in the same directory as the input angular momentum file.
"""

import numpy as np
import pandas as pd
import argparse
import os
from typing import List, Dict

# ------------------------------
# Label mapping
# ------------------------------
NAME_TO_LABEL = {
    "Head": "RHE",
    "Torso": "RTX",
    "Pelvis": "RPV",
    "R_UA": "RAR",
    "R_FA": "RFA",
    "R_Hand": "RHA",
    "L_UA": "LAR",
    "L_FA": "LFA",
    "L_Hand": "LHA",
    "R_Thigh": "RTH",
    "R_Shank": "RSK",
    "R_Foot": "RFT",
    "R_Toes": "RTO",
    "L_Thigh": "LTH",
    "L_Shank": "LSK",
    "L_Foot": "LFT",
    "L_Toes": "LTO",
}

# ------------------------------
# De Leva mass fractions
# ------------------------------

_TRUNK_DEN = 0.142 + 0.355
_PELVIS_RATIO = 0.142 / _TRUNK_DEN
_TORSO_RATIO  = 0.355 / _TRUNK_DEN

def get_deleva_mass_frac(sex: str) -> Dict[str, float]:
    if sex.lower() == "male":
        trunk = 0.4346
        return {
            "Head": 0.0694,
            "Pelvis": trunk * _PELVIS_RATIO,
            "Torso":  trunk * _TORSO_RATIO,
            "R_UA": 0.0271, "L_UA": 0.0271,
            "R_FA": 0.0162, "L_FA": 0.0162,
            "R_Hand": 0.0061, "L_Hand": 0.0061,
            "R_Thigh": 0.1416, "L_Thigh": 0.1416,
            "R_Shank": 0.0433, "L_Shank": 0.0433,
            "R_Foot": 0.0137, "L_Foot": 0.0137,
            "R_Toes": 0.0, "L_Toes": 0.0,
        }

    trunk = 0.4257  # female
    return {
        "Head": 0.0668,
        "Pelvis": trunk * _PELVIS_RATIO,
        "Torso":  trunk * _TORSO_RATIO,
        "R_UA": 0.0255, "L_UA": 0.0255,
        "R_FA": 0.0138, "L_FA": 0.0138,
        "R_Hand": 0.0056, "L_Hand": 0.0056,
        "R_Thigh": 0.1478, "L_Thigh": 0.1478,
        "R_Shank": 0.0481, "L_Shank": 0.0481,
        "R_Foot": 0.0129, "L_Foot": 0.0129,
        "R_Toes": 0.0, "L_Toes": 0.0,
    }

# ------------------------------
# CSV block helpers
# ------------------------------

def find_data_start(df: pd.DataFrame) -> int:
    first_col = df.columns[0]
    for i, v in enumerate(df[first_col].values):
        try:
            if int(float(v)) == 1:
                return i
        except:
            pass
    return 5

def extract_blocks(df: pd.DataFrame, start_row: int) -> List[dict]:
    cols = df.columns[1:]
    blocks = []
    for i in range(0, len(cols), 3):
        cX, cY, cZ = cols[i:i+3]
        data = np.vstack([
            pd.to_numeric(df[cX].iloc[start_row:], errors='coerce').to_numpy(),
            pd.to_numeric(df[cY].iloc[start_row:], errors='coerce').to_numpy(),
            pd.to_numeric(df[cZ].iloc[start_row:], errors='coerce').to_numpy(),
        ]).T
        blocks.append({
            "name": str(df[cX].iloc[0]),
            "type": str(df[cX].iloc[1]),
            "label": str(df[cX].iloc[2]),
            "axes": (df[cX].iloc[3], df[cY].iloc[3], df[cZ].iloc[3]),
            "cols": (cX, cY, cZ),
            "data": data
        })
    return blocks

def group_by_trials(ang_blocks: List[dict]):
    groups, cur = [], []
    for b in ang_blocks:
        cur.append(b)
        if b["name"] == "FullBody_AngMom":
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups

def chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# ------------------------------
# Core computation
# ------------------------------

def compute_for_all_groups(ang_groups, cog_groups, vel_groups, start_row, ang_df,
                           total_mass_kg: float, mass_frac: Dict[str, float]):


    out = ang_df.copy()


    T = len(ang_df) - start_row


    col_names = ang_df.columns[1:]

    # A list to store all computed output blocks
    computed_blocks = []

    # ----------------------------------------------------------
    # Loop over all trials/groups of segments
    # ----------------------------------------------------------
    for g_idx, ang_group in enumerate(ang_groups):

        # Matching groups for COG and velocity
        cog_group = cog_groups[g_idx]
        vel_group = vel_groups[g_idx]

    
        label_map = {cb["label"]: (cb["data"], vb["data"])
                     for cb, vb in zip(cog_group, vel_group)}

        # ------------------------------------------------------
        # 1. Compute segment masses m_i (De Leva fraction * total mass)
        # ------------------------------------------------------
        m_i = {}
        for b in ang_group:
            seg = b["name"].split("_AngMom")[0]
            if seg != "FullBody":
                m_i[seg] = mass_frac.get(seg, 0.0) * total_mass_kg

        # ------------------------------------------------------
        # 2. Compute whole-body COM position rG 
    
        # ------------------------------------------------------
        rG = np.zeros((T,3))
        totM = 0.0

        for b in ang_group:
            seg = b["name"].split("_AngMom")[0]
            if seg == "FullBody":
                continue

            lab = NAME_TO_LABEL.get(seg)
            if lab not in label_map:
                continue

            # r_i = segment COM position
            # v_i = segment COM velocity
            r_i, v_i = label_map[lab]
            r_i = r_i[:T]
            v_i = v_i[:T]

            # Accumulate weighted COG positions
            rG += m_i[seg] * r_i
            totM += m_i[seg]


        rG /= totM

        # ------------------------------------------------------
        # 3. Initialize full-body angular momentum about COM
        # ------------------------------------------------------
        H_full = np.zeros((T,3))

        # ------------------------------------------------------
        # 4. Convert each segment’s angular momentum
        #    from LAB origin → Body COM
        
        # ------------------------------------------------------
        for b in ang_group:
            nm = b["name"]

            # Skip the “FullBody_AngMom” entry (will be reconstructed)
            if nm == "FullBody_AngMom":
                continue

            seg = nm.split("_AngMom")[0]

         
            I_w = b["data"][:T]

            lab = NAME_TO_LABEL.get(seg)

        
            if lab is None:
                raise ValueError(
                    f"Segment '{seg}' not found in NAME_TO_LABEL mapping. "
                    f"Check NAME_TO_LABEL or your signal names."
                )

        
            if lab not in label_map:
                raise ValueError(
                    f"Missing COM/Velocity for segment '{seg}' (expected label '{lab}'). "
                    f"Check your COG/vel CSV exports."
                )

           
            r_i, v_i = label_map[lab]
            r_i = r_i[:T]
            v_i = v_i[:T]
            mi = m_i[seg]
            R = r_i - rG
          

            H_G = I_w + np.cross(R,mi*v_i)



            new_name = nm.replace("AngMom_wrt_LAB", "AngMom_wrt_BodyCOM")

            computed_blocks.append((new_name, H_G))

            
            H_full += H_G

    
        computed_blocks.append(("FullBody_AngMom_wrt_BodyCOM", H_full))

    # ----------------------------------------------------------
    # 5. Write results back into the output dataframe
    # ----------------------------------------------------------
    for k, (nm, data) in enumerate(computed_blocks):
        cX, cY, cZ = col_names[3*k:3*k+3]

        # First row: the signal name
        out.loc[0, cX] = nm
        out.loc[0, cY] = nm
        out.loc[0, cZ] = nm

        # Data rows
        out.loc[start_row:, cX] = data[:,0]
        out.loc[start_row:, cY] = data[:,1]
        out.loc[start_row:, cZ] = data[:,2]

    return out

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ang", required=True)
    ap.add_argument("--cog", required=True)
    ap.add_argument("--vel", required=True)
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--sex", required=True, choices=["male","female"])
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    # Auto output path if --out omitted
    if args.out is None:
        base = os.path.dirname(args.ang)
        name = os.path.basename(args.ang).replace("_AngMoms_wrt_LAB", "_AngMom_wrt_BodyCOM")
        args.out = os.path.join(base, name)

    mass_frac = get_deleva_mass_frac(args.sex)

    ang_df = pd.read_csv(args.ang)
    cog_df = pd.read_csv(args.cog)
    vel_df = pd.read_csv(args.vel)

    start_row = find_data_start(ang_df)

    ang_blocks = extract_blocks(ang_df, start_row)
    cog_blocks = extract_blocks(cog_df, start_row)
    vel_blocks = extract_blocks(vel_df, start_row)

    ang_groups = group_by_trials(ang_blocks)
    group_len = len(ang_groups[0])

    cog_groups = chunk(cog_blocks, group_len)
    vel_groups = chunk(vel_blocks, group_len)

    out_df = compute_for_all_groups(
        ang_groups, cog_groups, vel_groups,
        start_row, ang_df,
        total_mass_kg=args.mass,
        mass_frac=mass_frac
    )

    out_df.to_csv(args.out, index=False)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
