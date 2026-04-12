from Trajectory_Visualizer import TrajectoryVisualizer
import pandas as pd
import BigLoader

data = BigLoader.loadallspecified(["E1", "E2", "E3"], "elbow")
viz = TrajectoryVisualizer(data)
viz.group_mean(
    joint="R_ANKLE_POSITION",
    participants=["E1", "E2", "E3"],
    group_label="Novice mean",
    mean_color="steelblue",
    trial_color="lightblue",
)
