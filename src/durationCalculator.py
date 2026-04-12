import numpy as np
 
data = {
    "E1": {
        "teep": {
            "lift": [456,1082,1458,1870,2328,2723,3136,3578,4025,4469,4896],
            "impact": [545,1133,1507,1912,2377,2769,3182,3628,4071,4515,4940],
            "foot_down": [610,1176,1553,1956,2424,2815,3229,3674,4119,4569,5011],
        },
        "roundhouse": {
            "lift": [703,1075,1542,2072,2667,3132,3642,4153,4521],
            "impact": [734,1105,1576,2106,2700,3163,3675,4185,4548],
            "foot_down": [806,1176,1635,2174,2753,3235,3741,4238,4604],
        },
        "elbow":{
            "start": [320,740,1010,1300,1620,1969,2385,2745,3075,3460,],
            "end": [640,903,1166,1520,1800,2150,2570,2900,3235,3660,],
        },
        "uppercut":{
             "start": [475,855,1275,1625,2015,2350,2770,3190,3485,3840],
             "end": [685,1027,1440,1847,2225,2575,2990,3380,3700,4035]
        }
    },

    "E2": {
        "teep": {
            "lift": [340,739,1122,1496,1882,2282,2714,3151,3578,4058,4476],
            "impact": [393,790,1167,1540,1930,2331,2766,3195,3620,4105,4524],
            "foot_down": [446,839,1220,1603,1990,2386,2821,3258,3695,4166,4587],
        },
        "elbow":{
            "start": [410,699,1015,1300,1610,1870,2150,2488,2890,3160,3470,3810,4100,4388],
            "end": [620,909,1200,1500,1823,2060,2380,2675,3053,3386,3371,3705,4369,4580],
        },
        "uppercut":{
             "start": [400,740,990,1250,1560,1850,2155,2415,2710,3065,],
             "end": [600,915,1190,1500,1770,2050,2350,2645,2920,3265]
        }
    },

    "E3": {
        "teep": {
            "lift": [206,552,891,1249,1576,1923,2288,2660,3012,3431],
            "impact": [240,588,929,1286,1618,1962,2325,2702,3052,3476],
            "foot_down": [314,661,1001,1355,1690,2031,2401,2772,3176,3540],
        },
        "roundhouse": {
            "lift": [477,972,1367,1723,2120,2503,2890,3279,3647,4022],
            "impact": [510,1007,1398,1756,2157,2534,2923,3312,3678,4056],
            "foot_down": [581,1066,1454,1828,2215,2602,2992,3375,3742,4131],
        },
        "elbow":{
            "start": [125,510,900,1300,1697,2090,2510,2930,3342,3740],
            "end": [260,680,1080,1469,1865,2250,2700,3100,3550,3940],
        },
        "uppercut":{
             "start": [115,540,930,1290,1665,2025,2435,2800,3200,3600],
             "end": [345,720,1080,1460,1875,2220,2595,2975,3390,3700]
        }
    },

    "N1": {
        "roundhouse": {
            "lift": [519,989,1245,1563,1855,2335,2621,2937,3230,3507],
            "impact": [560,1024,1280,1595,1887,2369,2656,2969,3262,3540],
            "foot_down": [760,1104,1372,1667,1959,2453,2762,3038,3331,3617],
        },
        "elbow":{
            "start": [290,515,800,1055,1345,1645,1923,2160,2470,2745],
            "end": [495,745,960,1265,1525,1844,2130,2370,2660,2950],
        },
        "uppercut":{
             "start": [185,445,735,1020,1326,1604,1893,2170,2444,2725],
             "end": [380,640,930,1230,1495,1798,2060,2350,2635,2920]
        }
    },

    "N2": {
        "teep": {
            "lift": [217,466,740,983,1247,1519,1779,2063,2411,2679],
            "impact": [269,512,790,1028,1290,1556,1818,2100,2442,2714],
            "foot_down": [337,591,871,1121,1403,1640,1887,2162,2516,2795],
        },
        "roundhouse": {
            "lift": [259,541,871,1130,1412,1751,2065,2431,2718,2972],
            "impact": [295,575,905,1165,1445,1789,2098,2469,2753,3009],
            "foot_down": [395,770,1070,1200,1550,1872,2185,2566,2845,3100],
        },
        "elbow":{
            "start": [298,580,800,1040,1275,1500,1700,1900,2160,2415],
            "end": [475,750,999,1200,1455,1660,1860,2070,2300,2555],
        },
        "uppercut":{
             "start": [190,430,640,860,1055,1260,1460,1680,1900,2090],
             "end": [390,610,850,1035,1258,1450,1640,1875,2060,2275]
        }
    },

    "N3": {
        "teep": {
            "lift": [215,489,771,1040,1349,1588,2058,2377,2966,3317],
            "impact": [264,535,810,1080,1395,1638,2100,2423,3017,3362],
            "foot_down": [313,575,860,1146,1446,1701,2166,2655,3083,3405],
        },
        "roundhouse": {
            "lift": [235,642,922,1192,1433,1868,2111,2340,2572,2784],
            "impact": [271,686,971,1234,1479,1909,2158,2389,2621,2837],
            "foot_down": [533,748,1045,1309,1553,1976,2210,2455,2685,2909],
        },
        "elbow":{
            "start": [245,500,703,910,1110,1300,1530,1735,1970,2185],
            "end": [390,595,848,1035,1240,1440,1650,1870,2089,2310],
        },
        "uppercut":{
             "start": [135,360,560,750,977,1200,1425,1665,1900,2100],
             "end": [335,477,700,894,1140,1350,1583,1809,2038,2260]
        }
    },

    "N4": {
        "teep": {
            "lift": [415,754,1038,1322,1608,1892,2230,2520,2799,3300],
            "impact": [458,795,1080,1366,1649,1935,2277,2563,2842,3345],
            "foot_down": [504,845,1136,1424,1712,2005,2332,2621,2943,3400],
        },
        "roundhouse": {
            "lift": [251,505,773,1157,1391,1628,1855,2100,2376,2616],
            "impact": [291,541,807,1194,1430,1664,1892,2132,2416,2650],
            "foot_down": [362,603,868,1260,1496,1726,1964,2214,2485,2720],
        },
        "elbow":{
            "start": [185,455,670,900,1125,1350,1625,1860,2065,2270],
            "end": [350,610,826,1045,1286,1516,1800,2000,2215,2400],
        },
        "uppercut":{
             "start": [182,460,730,965,1190,1412,1657,1880,2126,2345,],
             "end": [375,635,870,1140,1350,1586,1820,2040,2275,2510,]
        }
    }
}
# ── helpers ──────────────────────────────────────────────────────────────────
 
def compute_durations(start_list, end_list):
    """
    Pair each start with its corresponding end and return a list of durations.
    If one list is longer than the other, extra values are ignored with a warning.
    """
    n_pairs = min(len(start_list), len(end_list))
    if len(start_list) != len(end_list):
        print(f"  ⚠  Length mismatch: {len(start_list)} starts vs {len(end_list)} ends "
              f"→ using first {n_pairs} pairs")
    durations = [end_list[i]/120*1000 - start_list[i]/120*1000 for i in range(n_pairs)]
    return durations
 
 
# ── main calculation ──────────────────────────────────────────────────────────
 
results = {}          # per-participant, per-technique durations
 
for participant, techniques in data.items():
    results[participant] = {}
    for technique in ("elbow", "uppercut"):
        if technique not in techniques:
            continue
 
        starts = techniques[technique]["start"]
        ends   = techniques[technique]["end"]
 
        durations = compute_durations(starts, ends)
        results[participant][technique] = durations
 
 
# ── report ────────────────────────────────────────────────────────────────────
 
print("=" * 65)
print("  DURATION ANALYSIS  (all values in ms)")
print("=" * 65)
 
for participant, techniques in results.items():
    print(f"\n{'─'*65}")
    print(f"  Participant: {participant}")
    print(f"{'─'*65}")
    for technique, durations in techniques.items():
        arr  = np.array(durations)
        mean = np.mean(arr)
        std  = np.std(arr, ddof=1)
        print(f"  {technique.upper()}")
        print(f"    Durations : {durations}")
        print(f"    Mean      : {mean:.2f} ms")
        print(f"    Std Dev   : {std:.2f} ms")
 
# ── per-subject summary table ─────────────────────────────────────────────────
 
print(f"\n{'='*65}")
print("  PER-SUBJECT SUMMARY")
print(f"{'='*65}")
print(f"  {'Subject':<10} {'Technique':<12} {'N':>4} {'Mean (ms)':>12} {'SD (ms)':>10}")
print(f"  {'─'*10} {'─'*12} {'─'*4} {'─'*12} {'─'*10}")
 
for participant, techniques in results.items():
    for technique, durations in techniques.items():
        arr  = np.array(durations)
        mean = np.mean(arr)
        std  = np.std(arr, ddof=1)
        print(f"  {participant:<10} {technique:<12} {len(durations):>4} {mean:>12.2f} {std:>10.2f}")
 
# ── group-level stats (E and N separately, then overall) ─────────────────────
 
group_durations = {
    "E": {"elbow": [], "uppercut": []},
    "N": {"elbow": [], "uppercut": []},
}
 
for participant, techniques in results.items():
    group = participant[0]          # "E" or "N"
    for technique, durations in techniques.items():
        group_durations[group][technique].extend(durations)
 
print(f"\n{'='*65}")
print("  GROUP STATS")
print(f"{'='*65}")
 
for group_label, techniques in group_durations.items():
    print(f"\n  Group {group_label}")
    print(f"  {'Technique':<12} {'N':>4} {'Mean (ms)':>12} {'SD (ms)':>10}")
    print(f"  {'─'*12} {'─'*4} {'─'*12} {'─'*10}")
    for technique, durations in techniques.items():
        arr  = np.array(durations)
        mean = np.mean(arr)
        std  = np.std(arr, ddof=1)
        print(f"  {technique:<12} {len(durations):>4} {mean:>12.2f} {std:>10.2f}")
 
print()
 