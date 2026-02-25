import Slicer
import os
import Scaler
import Averager
import Dataloader
import matplotlib.pyplot as plt

# Structured dataset: subject -> movement -> frame types

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
        }
    },

    "E2": {
        "teep": {
            "lift": [340,739,1122,1496,1882,2282,2714,3151,3578,4058,4476],
            "impact": [393,790,1167,1540,1930,2331,2766,3195,3620,4105,4524],
            "foot_down": [446,839,1220,1603,1990,2386,2821,3258,3695,4166,4587],
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
        }
    },

    "N1": {
        "teep": {
            "lift": [434,790,1174,1566,2368,2839,3250,3894,4304,4764],
            "impact": [470,840,1209,1604,2413,2880,3291,3939,4344,4810],
            "foot_down": [555,916,1292,1896,2478,2937,3368,4009,4400,4867],
        },
        "roundhouse": {
            "lift": [519,989,1245,1563,1855,2335,2621,2937,3230,3507],
            "impact": [560,1024,1280,1595,1887,2369,2656,2969,3262,3540],
            "foot_down": [760,1104,1372,1667,1959,2453,2762,3038,3331,3617],
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
        }
    }
}

subjects = ["E1", "E2", "E3", "N1", "N2", "N3", "N4"]
movements = ["roundhouse", "teep"]

for subject in subjects:
        for movement in movements:
            if subject == "E2" and movement == "roundhouse":
                continue
            
            trialPath ="/" + subject + "/" + movement
            dataPath = "calculatedAngMomStuff" + trialPath
            SlicedResultsPath = "processed_AngMomData/" +trialPath + "/sliced"
            scaledResultPath = "processed_AngMomData/" + trialPath + "/scaled"

            segmentLiftFrames = data[subject][movement]["lift"]
            segmentImpactFrames = data[subject][movement]["impact"]
            segmentFootDownFrames = data[subject][movement]["foot_down"]

            # Frame numbers for each segment phase boundary
            
            segmentBeginFrame = Slicer.calcBeginnframe(segmentLiftFrames)

            for file in os.listdir(dataPath):
                Slicer.sliceData(os.path.join(dataPath, file), SlicedResultsPath, segmentBeginFrame)

            grfPath = os.path.join(SlicedResultsPath + "/theta")
            Segments = Slicer.findTeepSegments(
                    segmentBeginFrame,
                    grfPath,
                    segmentLiftFrames,
                    segmentImpactFrames,
                    segmentFootDownFrames,

                )

            for directory in sorted(os.listdir(SlicedResultsPath)):
                
                print(directory)
                Scaler.scaleDirectoryToFourPhases(os.path.join(SlicedResultsPath, directory), Segments, scaledResultPath, directory)
                
            #Averager.run(scaled_results_dir=scaledResultPath, output_dir="AveragedResults")



