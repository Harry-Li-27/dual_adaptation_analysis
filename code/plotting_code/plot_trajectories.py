import json
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_trajectory(movement_traj_str: str) -> np.ndarray:
    """
    Parse movement_trajectory JSON into an (N, 2) array of x,y in screen coords.

    Assumes movement_traj_str is a JSON-encoded list of dicts with keys "x", "y", "t".
    Returns empty array if result is not a non-empty list.
    """
    result = json.loads(movement_traj_str)

    # Unwrap nested JSON strings, if any
    max_iter = 5
    iter_count = 0
    while isinstance(result, str) and iter_count < max_iter:
        result = json.loads(result)
        iter_count = iter_count + 1

    if not isinstance(result, list) or len(result) == 0:
        return np.zeros((0, 2), dtype=float)

    xs = np.array([pt["x"] for pt in result], dtype=float)
    ys = np.array([pt["y"] for pt in result], dtype=float)

    coords = np.stack([xs, ys], axis=1)
    return coords


def workspace_center_screen_coords(workspace: int) -> tuple:
    """
    Same workspace-to-center mapping as util._extract_angle_from_trajectory.

    Returns (center_x, center_y) in screen-normalized coordinates.
    """
    workspace_int = int(workspace)

    if workspace_int == 0:
        center_x = 0.25
    elif workspace_int == 2:
        center_x = 0.75
    else:
        center_x = 0.5

    center_y = 0.5
    return center_x, center_y


def to_workspace_relative(coords: np.ndarray, workspace: int) -> np.ndarray:
    """
    Convert screen coordinates (x,y in [0,1]) to workspace-centered coordinates (dx, dy),
    matching the convention in util._extract_angle_from_trajectory:

        dx = x - center_x
        dy = center_y - y   (so positive dy is 'up' on the screen)

    Returns an (N, 2) array of (dx, dy).
    """
    center_x, center_y = workspace_center_screen_coords(workspace)

    xs = coords[:, 0]
    ys = coords[:, 1]

    dx = xs - center_x
    dy = center_y - ys

    rel = np.stack([dx, dy], axis=1)
    return rel


def plot_trajectories_for_trials(fname: str, trial_nums: List[int]) -> None:
    """
    Load a CSV at `fname`, select rows whose trialNum is in trial_nums,
    and create subplots of each trial's trajectory.

    - Uses movement_trajectory screen coords.
    - Converts to workspace-centered coords so the workspace center is (0,0).
    - Shows target at radius 0.33 using the target_angle column (deg, 0° along +x).
    """
    df = pd.read_csv(fname)

    if "trialNum" not in df.columns:
        df["trialNum"] = np.arange(1, len(df) + 1, dtype=int)

    mask = df["trialNum"].isin(trial_nums)
    selected = df.loc[mask]

    if selected.empty:
        print("No matching trials found for the given trial numbers.")
        return

    has_target_angle = "target_angle" in df.columns

    n_trials = len(selected)
    ncols = min(n_trials, 4)
    nrows = int(np.ceil(n_trials / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows))

    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    else:
        axes_list = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes_list, selected.iterrows()):
        traj_str = row["movement_trajectory"]
        coords_screen = parse_trajectory(traj_str)

        trial_num = int(row["trialNum"])
        if "workspace" in row.index:
            workspace_val = int(row["workspace"])
        else:
            workspace_val = 1

        if coords_screen.shape[0] == 0:
            ax.set_title("trial {} (no trajectory)".format(trial_num))
            ax.set_aspect("equal", adjustable="box")
            ax.axhline(0.0, linewidth=0.5)
            ax.axvline(0.0, linewidth=0.5)
            continue

        coords_rel = to_workspace_relative(coords_screen, workspace_val)
        xs_rel = coords_rel[:, 0]
        ys_rel = coords_rel[:, 1]

        ax.plot(xs_rel, ys_rel, "-")

        ax.scatter(0.0, 0.0, s=30.0)

        target_x = None
        target_y = None
        if has_target_angle:
            target_angle_deg = float(row["target_angle"])
            theta_rad = np.deg2rad(target_angle_deg)
            radius = 0.33

            target_x = radius * np.cos(theta_rad)
            target_y = radius * np.sin(theta_rad)

            ax.scatter(target_x, target_y, s=40.0)

        all_x = xs_rel
        all_y = ys_rel
        if target_x is not None and target_y is not None:
            all_x = np.concatenate([all_x, np.array([target_x])])
            all_y = np.concatenate([all_y, np.array([target_y])])

        max_abs_x = float(np.max(np.abs(all_x)))
        max_abs_y = float(np.max(np.abs(all_y)))
        max_extent = max(max_abs_x, max_abs_y, 0.33)

        margin = 0.02
        extent = max_extent + margin

        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)

        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0.0, linewidth=0.5)
        ax.axvline(0.0, linewidth=0.5)

        ax.set_xlabel("x (relative to workspace center)")
        ax.set_ylabel("y (relative to workspace center)")
        title_str = "trial {} (workspace {})".format(trial_num, workspace_val)
        if has_target_angle:
            title_str = title_str + ", rot {:.1f}°".format(target_angle_deg)
        ax.set_title(title_str)

    for k in range(len(selected), len(axes_list)):
        axes_list[k].axis("off")

    fig.suptitle("Trajectories in workspace-centered coordinates", y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Edit these two lines when you run the script:
    fname = "processed_data/no_bad_mt_data_150_day1/S-experiment_trial_v3_t1-Group1-ID2.csv"
    interesting_trials = [427, 428, 432]

    plot_trajectories_for_trials(fname, interesting_trials)
