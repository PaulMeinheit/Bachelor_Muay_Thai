import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class TrajectoryVisualizer:
    """
    3D trajectory visualizer for biomechanics MultiIndex time series data.

    The DataFrame is expected to have a MultiIndex with levels:
        (participant, trial, variable)
    and columns representing time frames.

    Example usage:
        viz = TrajectoryVisualizer(df)
        viz.plot(participant="E1", trial="0", joint="R_WRIST_POSITION")
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.col_names = df.columns.names  # e.g. ['participant', 'trial', 'variable']

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_xyz(self, participant: str, trial: str, joint: str) -> tuple:
        """Extract X, Y, Z time series arrays for a given joint from MultiIndex columns."""
        axes = {}
        for axis in ("X", "Y", "Z"):
            col_key = (participant, trial, f"{joint}_{axis}")
            if col_key not in self.df.columns:
                raise KeyError(
                    f"Could not find '{joint}_{axis}' for participant '{participant}', "
                    f"trial '{trial}'.\n"
                    f"Check available joints with .list_joints(participant, trial)."
                )
            axes[axis] = self.df[col_key].to_numpy(dtype=float)
        return axes["X"], axes["Y"], axes["Z"]
       
    def _make_gradient_segments(self, x, y, z):
        """Build line segments for a colour-gradient trajectory."""
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments



    # ── public interface ───────────────────────────────────────────────────────

    def list_participants(self) -> list:
        """Return all unique participant IDs from columns."""
        return list(self.df.columns.get_level_values(0).unique())

    def list_trials(self, participant: str) -> list:
        """Return all trials for a given participant from columns."""
        mask = self.df.columns.get_level_values(0) == participant
        return list(self.df.columns[mask].get_level_values(1).unique())

    def list_joints(self, participant: str, trial: str) -> list:
        """
        Return all available joints (without _X/_Y/_Z suffix)
        for a given participant and trial from columns.
        """
        mask = (
            (self.df.columns.get_level_values(0) == participant) &
            (self.df.columns.get_level_values(1) == trial)
        )
        variables = self.df.columns[mask].get_level_values(2)
        joints = set()
        for v in variables:
            if v.endswith(("_X", "_Y", "_Z")):
                joints.add(v[:-2])  # strip _X / _Y / _Z
        return sorted(joints)

    def plot(
        self,
        participant: str,
        trial: str,
        joint: str,
        color_by_time: bool = True,
        show_start_end: bool = True,
        frame_start: int = None,
        frame_end: int = None,
        figsize: tuple = (10, 7),
        cmap: str = "plasma",
        title: str = None,
        elev: float = 20,
        azim: float = 45,
    ):
        """
        Plot the 3D trajectory of a joint.

        Parameters
        ----------
        participant   : participant ID, e.g. "E1"
        trial         : trial ID, e.g. "0"
        joint         : joint base name WITHOUT axis suffix, e.g. "R_WRIST_POSITION"
        color_by_time : if True, colour the trajectory by time (gradient)
        show_start_end: if True, mark the start (green) and end (red) points
        frame_start   : first frame index to plot (default: all)
        frame_end     : last frame index to plot (default: all)
        figsize       : figure size tuple
        cmap          : matplotlib colormap name for time gradient
        title         : custom plot title
        elev / azim   : initial 3D viewing angle
        """
        x, y, z = self._get_xyz(participant, trial, joint)

        # optional frame slicing
        fs = frame_start if frame_start is not None else 0
        fe = frame_end   if frame_end   is not None else len(x)
        x, y, z = x[fs:fe], y[fs:fe], z[fs:fe]

        # remove NaN frames
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[valid], y[valid], z[valid]

        if len(x) == 0:
            raise ValueError("No valid (non-NaN) data found for the selected range.")

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection="3d")

        if color_by_time:
            segments = self._make_gradient_segments(x, y, z)
            norm     = plt.Normalize(0, len(x) - 1)
            colors   = plt.get_cmap(cmap)(norm(np.arange(len(segments))))
            lc = Line3DCollection(segments, colors=colors, linewidth=2, alpha=0.85)
            ax.add_collection3d(lc)

            # colour bar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
            cbar.set_label("Frame", fontsize=10)
        else:
            ax.plot(x, y, z, linewidth=2, color="steelblue", alpha=0.85)

        if show_start_end:
            ax.scatter(*[x[0]],  *[y[0]],  *[z[0]],
                       color="green", s=60, zorder=5, label="Start")
            ax.scatter(*[x[-1]], *[y[-1]], *[z[-1]],
                       color="red",   s=60, zorder=5, label="End")
            ax.legend(fontsize=9)

        # axis limits with a small margin
        def _lim(arr):
            pad = (arr.max() - arr.min()) * 0.1 or 0.01
            return arr.min() - pad, arr.max() + pad

        ax.set_xlim(*_lim(x))
        ax.set_ylim(*_lim(y))
        ax.set_zlim(*_lim(z))

        ax.set_xlabel("X (mm)", fontsize=10)
        ax.set_ylabel("Y (mm)", fontsize=10)
        ax.set_zlabel("Z (mm)", fontsize=10)

        default_title = f"{joint}  |  {participant}  –  Trial {trial}"
        ax.set_title(title or default_title, fontsize=12, fontweight="bold", pad=12)

        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()

    def compare(
        self,
        joint: str,
        participants: list,
        figsize: tuple = (12, 8),
        elev: float = 20,
        azim: float = 45,
    ):
        """
        Overlay multiple participant/trial trajectories for the same joint.

        Parameters
        ----------
        joint      : joint base name, e.g. "R_WRIST_POSITION"
        selections : list of (participant, trial) tuples,
                     e.g. [("E1","0"), ("E2","0"), ("N1","0")]
        """
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection="3d")
        cmap = plt.get_cmap("tab10")

        for i, (participant, trial) in enumerate(selections):
            try:
                x, y, z = self._get_xyz(participant, trial, joint)
                valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
                x, y, z = x[valid], y[valid], z[valid]
                ax.plot(x, y, z, linewidth=2,
                        color=cmap(i), label=f"{participant} – Trial {trial}",
                        alpha=0.85)
                ax.scatter(x[0],  y[0],  z[0],  color=cmap(i), s=50, marker="o")
                ax.scatter(x[-1], y[-1], z[-1], color=cmap(i), s=50, marker="^")
            except KeyError as e:
                print(f"Skipping {participant}/{trial}: {e}")

        ax.set_xlabel("X (mm)", fontsize=10)
        ax.set_ylabel("Y (mm)", fontsize=10)
        ax.set_zlabel("Z (mm)", fontsize=10)
        ax.set_title(f"Trajectory comparison – {joint}", fontsize=12,
                     fontweight="bold", pad=12)
        ax.legend(fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()

    def group_mean(
        self,
        joint: str,
        participants: list,
        group_label: str = "Group mean",
        mean_color: str = "black",
        trial_color: str = "steelblue",
        trial_alpha: float = 0.15,
        mean_linewidth: float = 3.0,
        trial_linewidth: float = 1.0,
        show_start_end: bool = True,
        figsize: tuple = (12, 8),
        title: str = None,
        elev: float = 20,
        azim: float = 45,
    ):
        """
        Plot individual trial trajectories as faint lines and their mean
        trajectory as a bold line.

        Parameters
        ----------
        joint         : joint base name WITHOUT axis suffix, e.g. "R_WRIST_POSITION"
        participants  : list of participant IDs, e.g. ["E1", "E2", "E3"]
                        All trials for each participant are included automatically.
        group_label   : legend label for the mean line
        mean_color    : colour of the mean trajectory
        trial_color   : colour of the individual faint trial lines
        trial_alpha   : opacity of the individual trial lines (0–1)
        mean_linewidth: line width of the mean trajectory
        trial_linewidth: line width of the individual trial lines
        show_start_end: mark start (green dot) and end (red dot) on the mean line
        figsize       : figure size tuple
        title         : custom plot title
        elev / azim   : initial 3D viewing angle
        """
        # expand participants -> all their trials automatically
        selections = []
        for participant in participants:
            for trial in self.list_trials(participant):
                selections.append((participant, trial))

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection="3d")

        all_x, all_y, all_z = [], [], []
        first_trial_plotted = False

        for participant, trial in selections:
            try:
                x, y, z = self._get_xyz(participant, trial, joint)

                # remove NaNs
                valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
                x, y, z = x[valid], y[valid], z[valid]

                if len(x) < 2:
                    print(f"  Skipping {participant}/{trial}: too few valid frames.")
                    continue

                all_x.append(x)
                all_y.append(y)
                all_z.append(z)

                # plot faint individual trial line
                label = "Individual trials" if not first_trial_plotted else None
                ax.plot(x, y, z,
                        color=trial_color,
                        linewidth=trial_linewidth,
                        alpha=trial_alpha,
                        label=label)
                first_trial_plotted = True

            except KeyError as e:
                print(f"  Skipping {participant}/{trial}: {e}")

        if len(all_x) == 0:
            raise ValueError("No valid trials found — cannot compute mean trajectory.")

        # compute mean trajectory
        mx = np.mean(all_x, axis=0)
        my = np.mean(all_y, axis=0)
        mz = np.mean(all_z, axis=0)

        ax.plot(mx, my, mz,
                color=mean_color,
                linewidth=mean_linewidth,
                alpha=1.0,
                label=f"{group_label}  (n={len(all_x)})",
                zorder=10)

        if show_start_end:
            ax.scatter(mx[0],  my[0],  mz[0],
                       color="green", s=80, zorder=11, label="Mean start")
            ax.scatter(mx[-1], my[-1], mz[-1],
                       color="red",   s=80, zorder=11, label="Mean end")

        # axis limits from mean trajectory with margin
        def _lim(arr):
            pad = (arr.max() - arr.min()) * 0.1 or 0.01
            return arr.min() - pad, arr.max() + pad

        ax.set_xlim(*_lim(mx))
        ax.set_ylim(*_lim(my))
        ax.set_zlim(*_lim(mz))

        ax.set_xlabel("X (mm)", fontsize=10)
        ax.set_ylabel("Y (mm)", fontsize=10)
        ax.set_zlabel("Z (mm)", fontsize=10)

        default_title = f"Mean trajectory – {joint}  ({group_label})"
        ax.set_title(title or default_title, fontsize=12, fontweight="bold", pad=12)

        ax.legend(fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()