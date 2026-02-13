import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mapping = {"1": "1", "2": "2"}


def plot_combined(csv_file: str, save_path: str = None, break_indices=None, participant_id: str = None, group_tag: str = None, removed: bool = None) -> None:
    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(32, 18))
    fig.suptitle(f"Group: {group_tag}, ID: {participant_id}, Removed: {removed}", fontsize=18, y=0.98)

    if len(df) == 0:
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return

    x_col = "cycleNum"
    x_label = "Cycle Number"

    if break_indices is None:
        break_indices = []

    if "trial_type" in df.columns:
        df["segment"] = (df["trial_type"] != df["trial_type"].shift()).cumsum()
    else:
        df["trial_type"] = "unknown"
        df["segment"] = 1

    for ax in axes:
        for _, group in df.groupby("segment"):
            if group["trial_type"].iloc[0] == "no_fb":
                x0 = group[x_col].min() - 0.5
                x1 = group[x_col].max() + 0.5
                ax.axvspan(x0, x1, color="lightgray", alpha=0.5)
        for idx in break_indices:
            if idx < df[x_col].max():
                ax.axvline(x=idx + 0.5, color="green", alpha=0.5, linewidth=2, label="Break" if idx == break_indices[0] else "")

    x_min = int(np.floor(df[x_col].min() / 10.0) * 10)
    x_max = int(np.ceil(df[x_col].max() / 10.0) * 10)
    xticks = np.arange(x_min, x_max + 1, 10)
    for ax in axes:
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.grid(True, which="both", axis="both", alpha=0.3)

    ax0 = axes[0]
    ax0.plot(df[x_col], df["rotation"], label="Perturbation", color="black", linewidth=2)
    ax0.scatter(df[x_col], df["angle_diff"], label="Participant's action", color="blue", s=50)
    ax0.set_ylabel("Angle (°)")
    ax0.set_title("Angle Performance (Cycle-averaged)")
    ax0.set_ylim(-60, 150)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    mt_min, mt_max = 10, 700
    mt_clipped = df["mt"].clip(lower=mt_min, upper=mt_max)
    ax1.scatter(df[x_col], mt_clipped, label="Movement Time", s=50)
    ax1.set_ylim(0, 710)
    ax1.set_ylabel("Movement Time (ms)")
    ax1.set_title("Movement Time (Cycle-averaged)")
    ax1.legend(loc="upper right")

    ax2 = axes[2]
    rt_min, rt_max = 10, 700
    rt_clipped = df["rt"].clip(lower=rt_min, upper=rt_max)
    ax2.scatter(df[x_col], rt_clipped, label="Reaction Time", s=50)
    ax2.set_ylim(0, 710)
    ax2.set_ylabel("Reaction Time (ms)")
    ax2.set_xlabel(x_label)
    ax2.set_title("Reaction Time (Cycle-averaged)")
    ax2.legend(loc="upper right")

    plt.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main() -> None:
    from tqdm import tqdm

    mode = "no_bad_mt_data_150"
    # mode = "raw_data_without_outlier"
    days = [1, 2]

    for day in days:
        mode_day = f"{mode}_day{day}"
        data_dir = os.path.join("processed_data", f"{mode}_day{day}_cycle")

        decision_json_path = os.path.join("plotting_helper_files", f"Remov_deci_all_blocks_{mode_day}.json")
        with open(decision_json_path, "r") as f:
            removal = json.load(f)

        break_trial_indices = [630 - 30] if day == 1 else [214 - 30]
        break_cycle_indices = [int(idx / 3) for idx in break_trial_indices]

        for fname in tqdm(os.listdir(data_dir), desc=f"Day {day}"):
            if not fname.endswith(".csv"):
                continue

            pid = os.path.splitext(fname)[0]
            id_only = next((part.replace("ID", "") for part in pid.split("-") if part.startswith("ID")), pid)

            info = removal.get(pid, {})
            removed = not info.get("decision", False)

            grp_old = next((part.replace("Group", "") for part in pid.split("-") if part.startswith("Group")), None)
            grp_tag = mapping.get(grp_old, grp_old)

            group_dir = f"group_{grp_tag}" if grp_tag is not None else "group_unknown"
            out_dir = os.path.join("result", "combined_cycle", mode_day, group_dir)
            os.makedirs(out_dir, exist_ok=True)

            in_file = os.path.join(data_dir, fname)
            out_file = os.path.join(out_dir, f"{pid}_combined_cycle_performance.png")

            plot_combined(in_file, out_file, break_indices=break_cycle_indices, participant_id=id_only, group_tag=grp_tag, removed=removed)


if __name__ == "__main__":
    main()
