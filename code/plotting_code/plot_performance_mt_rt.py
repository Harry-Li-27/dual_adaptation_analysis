import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import json

# mapping = {
#     "7": "1", "8": "2", "1": "3", "3": "4",
#     "5": "5", "2": "6", "4": "7", "6": "8", "9": "9"
# }

mapping = {"1":"1", "2":"2"}


def plot_combined(
    csv_file: str,
    save_path: str = None,
    break_indices: list = [99, 451, 803],
    error_check_idx_start: int = 0,
    error_check_idx_end: int = 100,
    participant_id: str = None,
    group_tag: str = None,
    removed: bool = None,
    # NEW: pass two (or more) index ranges like [(start1, end1), (start2, end2)]
    avg_ranges = None,
    # NEW: choose what to average: "error" => angle_diff - rotation, "angle" => angle_diff
    avg_on: str = "error"
) -> None:
    """
    Reads a CSV and generates a combined figure with three subplots:
      1) Angle performance (perturbation, raw & smoothed angle_diff)
      2) Movement time (mt)
      3) Reaction time (rt)
    All share the common x-axis of trialNum, with shaded no-feedback segments,
    break indicators, and stats annotations.
    """
    # Note that for no outlier removed stuff, there will be outlier and affect std a lot
    # Load data
    df = pd.read_csv(csv_file)
    if len(df) == 0:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(32, 18))
        fig.suptitle(
            f"Group: {group_tag}, ID: {participant_id}, Removed: {removed}",
            fontsize=18, y=0.98
        )
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return


    # Prepare time and compute angle difference
    df['datetime'] = pd.to_datetime(df['currentDate'], format='%m/%d/%Y %H:%M.%S.%f')
    df['time_diff'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()

    # INSERT (after line 60): prepare baseline-shift text if available
    baseline_block = None
    if 'baseline_mean' in df.columns:
        # best-effort parse and take the first non-NaN
        try:
            baseline_vals = pd.to_numeric(df['baseline_mean'], errors='coerce').dropna()
            if len(baseline_vals) > 0:
                baseline_shift_value = -float(baseline_vals.iloc[0])
                baseline_block = f"Baseline shift applied: {baseline_shift_value:.2f}°"
        except Exception:
            baseline_block = None
    # --- Statistics Calculation for Performance---
    # Selected range stats
    selected_error = df['angle_diff'][error_check_idx_start:error_check_idx_end] - df['rotation'][error_check_idx_start:error_check_idx_end]
    std_sel = selected_error.std()
    rmse_sel = np.sqrt((selected_error ** 2).mean())
    mean_sel = selected_error.mean()
    selected_error_no = selected_error[abs(selected_error - mean_sel) <= 2 * std_sel]
    rmse_sel_no = np.sqrt((selected_error_no ** 2).mean())
    selected_stats_text = (
        f"Selected ({error_check_idx_start}-{error_check_idx_end}):\n"
        f"STD: {std_sel:.2f}\n"
        f"RMSE: {rmse_sel:.2f}\n"
        f"RMSE w/o outlier: {rmse_sel_no:.2f}"
    )

    # # --- Statistics Calculation for Performance---
    # # Pre-perturbation window: 20 trials immediately before the first non-zero 'rotation'
    # nonzero_idx = np.flatnonzero(df['rotation'].to_numpy() != 0)
    # if len(nonzero_idx) > 0:
    #     first_nz = int(nonzero_idx[0])
    #     sel_start = max(0, first_nz - 25)
    #     sel_end_exclusive = first_nz  # window is [sel_start, first_nz)
    # else:
    #     # Fallback: if no non-zero perturbation exists, use the first 20 trials (or all if shorter)
    #     sel_start = 0
    #     sel_end_exclusive = min(25, len(df))
    # Compute error within the selected window using iloc (position-based)
    # selected_error = (df['angle_diff'].iloc[sel_start:sel_end_exclusive]
    #                 - df['rotation'].iloc[sel_start:sel_end_exclusive])
    # std_sel = float(selected_error.std())
    # rmse_sel = float(np.sqrt((selected_error ** 2).mean()))
    # mean_sel = float(selected_error.mean())
    # selected_error_no = selected_error[abs(selected_error - mean_sel) <= 2.0 * std_sel]
    # rmse_sel_no = float(np.sqrt((selected_error_no ** 2).mean()))

    # # Update the annotation text to reflect the new window
    # selected_stats_text = (
    #     f"Pre-perturbation (idx {sel_start}-{max(sel_start, sel_end_exclusive-1)}):\n"
    #     f"STD: {std_sel:.2f}\n"
    #     f"RMSE: {rmse_sel:.2f}\n"
    #     f"RMSE w/o outlier: {rmse_sel_no:.2f}"
    # )

    # All-trials stats
    all_error = df['angle_diff'] - df['rotation']
    std_all = all_error.std()
    rmse_all = np.sqrt((all_error ** 2).mean())
    mean_all = all_error.mean()
    all_error_no = all_error[abs(all_error - mean_all) <= 2 * std_all]
    rmse_all_no = np.sqrt((all_error_no ** 2).mean())
    all_stats_text = (
        f"All Trials:\n"
        f"STD: {std_all:.2f}\n"
        f"RMSE: {rmse_all:.2f}\n"
        f"RMSE w/o outlier: {rmse_all_no:.2f}"
    )
    # --- End of Stats Calculation ---

    # NEW: helper to compute a robust mean over index slices
    def _range_mean(series, start, end):
        n = len(series)
        s = max(0, int(start))
        e = min(n, int(end))
        if s >= e:
            return np.nan
        # Use iloc to slice by index positions
        return float(series.iloc[s:e].mean())
    
    def _range_median(series, start, end):
        n = len(series)
        s = max(0, int(start))
        e = min(n, int(end))
        if s >= e:
            return np.nan
        # Use iloc to slice by index positions
        return float(series.iloc[s:e].median())

    # NEW: prepare averages text for requested ranges
    avg_block = None
    if avg_ranges:
        target_series = (all_error if avg_on == "error" else df['angle_diff'])
        label = "angle_diff − rotation" if avg_on == "error" else "angle_diff"
        lines_txt = []
        for i, (s, e) in enumerate(avg_ranges, start=1):
            m = _range_mean(target_series, s, e)
            lines_txt.append(f"Range {i} ({s}-{e}) average: {m:.2f}°")
        avg_block = "Averages (" + label + "):\n" + "\n".join(lines_txt)

    smoothed = gaussian_filter1d(df['angle_diff'].values, sigma=10)

    # Define segments based on trial_type
    df['segment'] = (df['trial_type'] != df['trial_type'].shift()).cumsum()

    # Set up figure with 3 stacked subplots
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(32, 18))
    fig.suptitle(
        f"Group: {group_tag}, ID: {participant_id}, Removed: {removed}",
        fontsize=18, y=0.98
    )

    # Common shading and breaks
    for ax in axes:
        # Shade 'no_fb' segments
        for _, group in df.groupby('segment'):
            if group['trial_type'].iloc[0] == 'no_fb':
                x0 = group['trialNum'].min() - 0.5
                x1 = group['trialNum'].max() + 0.5
                ax.axvspan(x0, x1, color='lightgray', alpha=0.5)
        # Draw break lines
        for idx in break_indices:
            if idx < df['trialNum'].max():
                ax.axvline(x=idx + 0.5, color='green', alpha=0.5, linewidth=2,
                           label='Break' if idx == break_indices[0] else '')

    tn_min = int(np.floor(df['trialNum'].min() / 10.0) * 10)
    tn_max = int(np.ceil(df['trialNum'].max() / 10.0) * 10)
    xticks = np.arange(tn_min, tn_max + 1, 10)
    for ax in axes:
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', which='both', labelbottom=True)
        ax.grid(True, which='both', axis='both', alpha=0.3)

    # Top subplot: performance
    ax0 = axes[0]
    # NEW: highlight avg_ranges on the angle subplot
    if avg_ranges:
        for (s, e) in avg_ranges:
            # clamp indices safely
            s_idx = max(0, min(len(df) - 1, int(s)))
            e_idx = max(0, min(len(df) - 1, int(e) - 1))
            if s_idx <= e_idx:
                x0 = df['trialNum'].iloc[s_idx] - 0.5
                x1 = df['trialNum'].iloc[e_idx] + 0.5
                ax0.axvspan(x0, x1, alpha=0.2, zorder=0)
    # Perturbation trace
    first_online = True
    for _, group in df.groupby('segment'):
        if group['trial_type'].iloc[0] == 'online_fb':
            ax0.plot(group['trialNum'], group['rotation'],
                     label='Perturbation' if first_online else '',
                     color='black', linewidth=2)
            first_online = False
    # Raw & smoothed participant action
    ax0.scatter(df['trialNum'], df['angle_diff'], label="Participant's action", color="blue", s=50)
    ax0.plot(df['trialNum'], smoothed, label="Smoothed action", linewidth=2, color='red')
    
    # Adjust margin and add stats text on ax0
    # ax0.text(0.05, 0.75, selected_stats_text, transform=ax0.transAxes, fontsize=14,
    #          bbox=dict(facecolor='white', alpha=0.5))
    # ax0.text(0.05, 0.45, all_stats_text, transform=ax0.transAxes, fontsize=14,
    #          bbox=dict(facecolor='white', alpha=0.5))

    # NEW: add the averages box under the existing stats
    # if avg_block:
    #     ax0.text(
    #         0.05, 0.25, avg_block,
    #         transform=ax0.transAxes, fontsize=14,
    #         bbox=dict(facecolor='white', alpha=0.5)
    #     )
    # if baseline_block:
    #     ax0.text(
    #         0.05, 0.05, baseline_block,
    #         transform=ax0.transAxes, fontsize=14,
    #         bbox=dict(facecolor='white', alpha=0.5)
    #     )
    ax0.set_ylabel('Angle (°)')
    ax0.set_title('Angle Performance')
    ax0.set_ylim(-60, 150)
    ax0.legend(loc='upper right')

    # Middle subplot: movement time with clipping at edges
    ax1 = axes[1]
    mt_min, mt_max = 10, 700
    mt_clipped = df['mt'].clip(lower=mt_min, upper=mt_max)
    # Old: ax1.scatter(df['trialNum'], df['mt'], label='Movement Time', s=50)
    ax1.scatter(df['trialNum'], mt_clipped, label='Movement Time', s=50)
    ax1.set_ylim(0, 710)
    perc_below = (df['mt'] < mt_min).mean() * 100
    perc_above = (df['mt'] > mt_max).mean() * 100
    # ax1.text(0.05, 0.8, f"% below {mt_min}: {perc_below:.1f}%\n% above {mt_max}: {perc_above:.1f}%", transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.5), fontsize=14)
    ax1.set_ylabel('Movement Time (ms)')
    ax1.set_title('Movement Time')
    ax1.legend(loc='upper right')

    # Bottom subplot: reaction time with clipping at edges
    ax2 = axes[2]
    rt_min, rt_max = 10, 700
    rt_clipped = df['rt'].clip(lower=rt_min, upper=rt_max)
    # Old: ax2.scatter(df['trialNum'], df['rt'], label='Reaction Time', s=50)
    ax2.scatter(df['trialNum'], rt_clipped, label='Reaction Time', s=50)
    ax2.set_ylim(0, 710)
    perc_below_rt = (df['rt'] < rt_min).mean() * 100
    perc_above_rt = (df['rt'] > rt_max).mean() * 100
    # ax2.text(0.05, 0.8, f"% below {rt_min}: {perc_below_rt:.1f}%\n% above {rt_max}: {perc_above_rt:.1f}%", transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.5), fontsize=14)
    ax2.set_ylabel('Reaction Time (ms)')
    ax2.set_xlabel('Trial Number')
    ax2.set_title('Reaction Time')
    ax2.legend(loc='upper right')

    # Adjust layout and save
    plt.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    from tqdm import tqdm

    data_type = "raw_data_without_outlier"
    data_dir = f'processed_data/{data_type}'

    # Path to JSON with removal decisions
    all_blocks_check = "all_blocks"

    decision_json_path = f"plotting_helper_files/Remov_deci_{all_blocks_check}_{data_type}.json"

    with open(decision_json_path, 'r') as f:
        removal = json.load(f)

    break_idxs = []
    allowed_groups = ["1", "2"]

    for fname in tqdm(os.listdir(data_dir)):
        if not fname.endswith('.csv'):
            continue
        pid = os.path.splitext(fname)[0]
        # grab “ID123” → “123”
        id_only = next(
            (p.replace('ID','') for p in pid.split('-') if p.startswith('ID')),
            pid
        )
        info = removal.get(pid, {})
        removed = not info.get('decision', False)
        grp_old = next((part.replace('Group','') for part in pid.split('-') if part.startswith('Group')), None)
        if allowed_groups is not None and grp_old not in allowed_groups:
            continue
        grp_tag = mapping.get(grp_old, grp_old)

        group_dir = f'group_{grp_tag}' if grp_tag is not None else 'group_unknown'
        out_dir = os.path.join('result', 'combined', data_type, group_dir)
        os.makedirs(out_dir, exist_ok=True)
        in_file = os.path.join(data_dir, fname)
        out_file = os.path.join(out_dir, f'{pid}_combined_performance.png')
        plot_combined(
            in_file, out_file, break_idxs,
            participant_id=id_only,
            group_tag=grp_tag,
            removed=removed,
            avg_ranges=[],
            # NEW: choose what to average: "error" or "angle"
            avg_on="angle",
        )

if __name__ == '__main__':
    main()