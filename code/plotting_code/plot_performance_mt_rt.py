import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # can be removed if you want
import json

# mapping = {
#     "7": "1", "8": "2", "1": "3", "3": "4",
#     "5": "5", "2": "6", "4": "7", "6": "8", "9": "9"
# }

mapping = {"1": "1", "2": "2"}


def plot_combined(
    csv_file: str,
    save_path: str = None,
    break_indices=None,
    error_check_idx_start: int = 0,
    error_check_idx_end: int = 100,
    participant_id: str = None,
    group_tag: str = None,
    removed: bool = None,
    avg_ranges=None,
    avg_on: str = "error",
) -> None:
    """
    Reads a CSV and generates a combined figure with three subplots:
      1) Angle performance (perturbation, raw angle_diff)
      2) Movement time (mt)
      3) Reaction time (rt)
    All share the common x-axis of trialNum, with shaded no-feedback segments,
    break indicators, and stats annotations.
    """
    df = pd.read_csv(csv_file)
    if len(df) == 0:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(32, 18))
        fig.suptitle(
            f"Group: {group_tag}, ID: {participant_id}, Removed: {removed}",
            fontsize=18,
            y=0.98,
        )
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return

    df["datetime"] = pd.to_datetime(df["currentDate"], format="%m/%d/%Y %H:%M.%S.%f")
    df["time_diff"] = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds()

    baseline_block = None
    if "baseline_mean" in df.columns:
        try:
            baseline_vals = pd.to_numeric(df["baseline_mean"], errors="coerce").dropna()
            if len(baseline_vals) > 0:
                baseline_shift_value = -float(baseline_vals.iloc[0])
                baseline_block = f"Baseline shift applied: {baseline_shift_value:.2f}°"
        except Exception:
            baseline_block = None

    selected_error = (
        df["angle_diff"][error_check_idx_start:error_check_idx_end]
        - df["rotation"][error_check_idx_start:error_check_idx_end]
    )
    std_sel = selected_error.std()
    rmse_sel = np.sqrt((selected_error ** 2).mean())
    mean_sel = selected_error.mean()
    selected_error_no = selected_error[
        abs(selected_error - mean_sel) <= 2 * std_sel
    ]
    rmse_sel_no = np.sqrt((selected_error_no ** 2).mean())

    all_error = df["angle_diff"] - df["rotation"]
    std_all = all_error.std()
    rmse_all = np.sqrt((all_error ** 2).mean())
    mean_all = all_error.mean()
    all_error_no = all_error[abs(all_error - mean_all) <= 2 * std_all]
    rmse_all_no = np.sqrt((all_error_no ** 2).mean())

    def _range_mean(series, start, end):
        n = len(series)
        s = max(0, int(start))
        e = min(n, int(end))
        if s >= e:
            return np.nan
        return float(series.iloc[s:e].mean())

    def _range_median(series, start, end):
        n = len(series)
        s = max(0, int(start))
        e = min(n, int(end))
        if s >= e:
            return np.nan
        return float(series.iloc[s:e].median())

    avg_block = None
    if avg_ranges:
        target_series = all_error if avg_on == "error" else df["angle_diff"]
        label = "angle_diff − rotation" if avg_on == "error" else "angle_diff"
        lines_txt = []
        for i, (s, e) in enumerate(avg_ranges, start=1):
            m = _range_mean(target_series, s, e)
            lines_txt.append(f"Range {i} ({s}-{e}) average: {m:.2f}°")
        avg_block = "Averages (" + label + "):\n" + "\n".join(lines_txt)

    df["segment"] = (df["trial_type"] != df["trial_type"].shift()).cumsum()

    if break_indices is None:
        break_indices = []

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(32, 18))
    fig.suptitle(
        f"Group: {group_tag}, ID: {participant_id}, Removed: {removed}",
        fontsize=18,
        y=0.98,
    )

    for ax in axes:
        for _, group in df.groupby("segment"):
            if group["trial_type"].iloc[0] == "no_fb":
                x0 = group["trialNum"].min() - 0.5
                x1 = group["trialNum"].max() + 0.5
                ax.axvspan(x0, x1, color="lightgray", alpha=0.5)
        for idx in break_indices:
            if idx < df["trialNum"].max():
                ax.axvline(
                    x=idx + 0.5,
                    color="green",
                    alpha=0.5,
                    linewidth=2,
                    label="Break" if idx == break_indices[0] else "",
                )

    tn_min = int(np.floor(df["trialNum"].min() / 10.0) * 10)
    tn_max = int(np.ceil(df["trialNum"].max() / 10.0) * 10)
    xticks = np.arange(tn_min, tn_max + 1, 10)
    for ax in axes:
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.grid(True, which="both", axis="both", alpha=0.3)

    ax0 = axes[0]
    if avg_ranges:
        for (s, e) in avg_ranges:
            s_idx = max(0, min(len(df) - 1, int(s)))
            e_idx = max(0, min(len(df) - 1, int(e) - 1))
            if s_idx <= e_idx:
                x0 = df["trialNum"].iloc[s_idx] - 0.5
                x1 = df["trialNum"].iloc[e_idx] + 0.5
                ax0.axvspan(x0, x1, alpha=0.2, zorder=0)

    ax0.plot(
        df["trialNum"],
        df["rotation"],
        label="Perturbation",
        color="black",
        linewidth=2,
    )

    ax0.scatter(
        df["trialNum"],
        df["angle_diff"],
        label="Participant's action",
        color="blue",
        s=50,
    )

    ax0.set_ylabel("Angle (°)")
    ax0.set_title("Angle Performance")
    ax0.set_ylim(-60, 150)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    mt_min, mt_max = 10, 700
    mt_clipped = df["mt"].clip(lower=mt_min, upper=mt_max)
    ax1.scatter(df["trialNum"], mt_clipped, label="Movement Time", s=50)
    ax1.set_ylim(0, 710)
    ax1.set_ylabel("Movement Time (ms)")
    ax1.set_title("Movement Time")
    ax1.legend(loc="upper right")

    ax2 = axes[2]
    rt_min, rt_max = 10, 700
    rt_clipped = df["rt"].clip(lower=rt_min, upper=rt_max)
    ax2.scatter(df["trialNum"], rt_clipped, label="Reaction Time", s=50)
    ax2.set_ylim(0, 710)
    ax2.set_ylabel("Reaction Time (ms)")
    ax2.set_xlabel("Trial Number")
    ax2.set_title("Reaction Time")
    ax2.legend(loc="upper right")

    plt.subplots_adjust(
        hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05
    )
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    from tqdm import tqdm

    mode = "raw_data_without_outlier"
    days = [1, 2]

    for day in days:
        mode_day = f"{mode}_day{day}"
        data_dir = os.path.join("processed_data", mode_day)

        decision_json_path = os.path.join(
            "plotting_helper_files",
            f"Remov_deci_all_blocks_{mode_day}.json",
        )

        with open(decision_json_path, "r") as f:
            removal = json.load(f)

        if day == 1:
            break_indices = [330]
        else:
            break_indices = [210]

        for fname in tqdm(os.listdir(data_dir), desc=f"Day {day}"):
            if not fname.endswith(".csv"):
                continue

            pid = os.path.splitext(fname)[0]

            id_only = next(
                (part.replace("ID", "") for part in pid.split("-") if part.startswith("ID")),
                pid,
            )

            info = removal.get(pid, {})
            removed = not info.get("decision", False)

            grp_old = next(
                (part.replace("Group", "") for part in pid.split("-") if part.startswith("Group")),
                None,
            )
            grp_tag = mapping.get(grp_old, grp_old)

            group_dir = f"group_{grp_tag}" if grp_tag is not None else "group_unknown"
            out_dir = os.path.join("result", "combined", mode_day, group_dir)
            os.makedirs(out_dir, exist_ok=True)

            in_file = os.path.join(data_dir, fname)
            out_file = os.path.join(out_dir, f"{pid}_combined_performance.png")

            plot_combined(
                in_file,
                out_file,
                break_indices=break_indices,
                participant_id=id_only,
                group_tag=grp_tag,
                removed=removed,
            )


if __name__ == "__main__":
    main()
