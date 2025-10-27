import os
import re
import json
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel


def load_and_preprocess(file_path):
    """
    Load a single participant CSV and return a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def check_full_trials(df, full_trial_num):
    """
    Check if the number of trials in df equals full_trial_num.
    Returns (bool, reason_str). 
    ``bool`` is True if count matches, False otherwise.
    """
    total_trials = len(df)
    if total_trials != full_trial_num:
        reason = (
            f"incomplete trials: total_trials={total_trials} "
            f"!= full_trial_num={full_trial_num}"
        )
        return False, reason
    return True, ""

def check_adaptation_level(df, start_idx, end_idx, testing_range, pick_portion=None, mean_threshold=5):
    """
    Check if the mean adaptation over the specified trials exceeds a threshold.
    Returns (bool, reason_str).
    """
    adaptations = df.iloc[start_idx:end_idx]["angle_diff"].values

    if pick_portion is not None:
        if not (0 < pick_portion <= 1):
            raise ValueError(f"pick_portion must be between 0 and 1, got {pick_portion}")
        k = int(np.ceil(len(adaptations) * pick_portion))
        adaptations = np.sort(adaptations)[-k:]

    mean_adapt = np.mean(adaptations)
    if mean_threshold > 0:
        if mean_adapt < mean_threshold:
            reason = f"no learning shown in {testing_range}, mean adaptation is {mean_adapt:.2f} < threshold {mean_threshold:.2f}"
            return False, reason
    else:
        if mean_adapt > mean_threshold:
            reason = f"no learning shown in {testing_range}, mean adaptation is {mean_adapt:.2f} > threshold {mean_threshold:.2f}"
            return False, reason
    return True, f"in {testing_range}: mean adaptation is {mean_adapt:.2f}"

def check_block_average(df, start_idx, end_idx, testing_range, mean_threshold=5):
    """
    Compute mean adaptation over [start_idx:end_idx], filtering to online feedback trials.
    Uses check_adaptation_level for thresholding logic so positive/negative thresholds work as before.
    """
    block_df = df.iloc[start_idx:end_idx]
    block_df = block_df[block_df["trial_type"] == "online_fb"]

    if len(block_df) == 0:
        return False, f"no online_fb trials in {testing_range}"

    # Reuse existing logic by delegating to check_adaptation_level on the filtered block.
    return check_adaptation_level(
        block_df,
        0,
        len(block_df),
        testing_range,
        pick_portion=None,
        mean_threshold=mean_threshold
    )

def check_block_median(df, start_idx, end_idx, testing_range, median_threshold=5):
    block_df = df.iloc[start_idx:end_idx]
    block_df = block_df[block_df["trial_type"] == "online_fb"]

    if len(block_df) == 0:
        return False, f"no online_fb trials in {testing_range}"

    adaptations = block_df["angle_diff"].values
    median_adapt = float(np.median(adaptations))

    if median_threshold > 0:
        if median_adapt < median_threshold:
            return False, (
                f"no learning shown in {testing_range}, "
                f"median adaptation is {median_adapt:.2f} < threshold {median_threshold:.2f}"
            )
    else:
        if median_adapt > median_threshold:
            return False, (
                f"no learning shown in {testing_range}, "
                f"median adaptation is {median_adapt:.2f} > threshold {median_threshold:.2f}"
            )
    return True, f"in {testing_range}: median adaptation is {median_adapt:.2f}"

def check_mt_proportion(df, mt_threshold, mt_fraction_threshold):
    """
    Check the proportion of trials where 'mt' > mt_threshold.
    If proportion > mt_fraction_threshold, return (False, reason).
    Otherwise, return (True, "").
    Assumes df has an 'mt' column.
    """
    mt_vals = df["mt"].values
    if len(mt_vals) == 0:
        # If no 'mt' values present, skip this check (treat as passed)
        return True, ""
    proportion = np.mean(mt_vals > mt_threshold)
    if proportion > mt_fraction_threshold:
        reason = (
            f"mt proportion {proportion:.2f} > {mt_fraction_threshold:.2f} "
            f"(mt_threshold={mt_threshold})"
        )
        return False, reason
    return True, ""

def extract_paired_groups(df, start1, end1, start2, end2):
    """
    Extract two numpy arrays of RTs from specified row-index ranges.
    """
    block1 = df.iloc[start1:end1]["rt"].values
    block2 = df.iloc[start2:end2]["rt"].values
    return block1, block2

def extract_triplet_groups(df, b1_start, b1_end, b2_start, b2_end, b3_start, b3_end):
    """
    Extract three numpy arrays of RTs from three blocks.
    """
    block1 = df.iloc[b1_start:b1_end]["rt"].values
    block2 = df.iloc[b2_start:b2_end]["rt"].values
    block3 = df.iloc[b3_start:b3_end]["rt"].values
    return block1, block2, block3

def run_paired_ttest(block1, block2):
    t_stat, p_value = ttest_rel(block1, block2)
    return t_stat, p_value

def compute_cohens_d(block1, block2):
    diffs = block1 - block2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    if std_diff == 0:
        return 0.0
    return mean_diff / std_diff

def _preperturb_window(df, window=40):
    """
    Return (sel_start, sel_end_exclusive) for the 'window' trials immediately
    before the first index where rotation becomes non-zero. If no such index,
    return the first 'window' trials (clipped by length).
    """
    # rot = df['rotation'][30:].to_numpy()
    # nonzero_idx = np.flatnonzero(rot != 0)
    # if len(nonzero_idx) > 0:
    #     first_nz = int(nonzero_idx[0])
    #     sel_start = max(0, first_nz - window)
    #     sel_end_exclusive = first_nz
    # else:
    #     sel_start = 0
    #     sel_end_exclusive = min(window, len(df))
    # return sel_start, sel_end_exclusive

    if len(df['rotation'] > 30):
        nonzero_idx = np.flatnonzero(df['rotation'][30:].to_numpy() != 0)
        if len(nonzero_idx) > 0:
            first_nz = int(nonzero_idx[0])+30
            sel_start = max(0, first_nz - window)
            sel_end_exclusive = first_nz  # window is [sel_start, first_nz)
        else:
            # Fallback: if no non-zero perturbation exists, use the first 20 trials (or all if shorter)
            sel_start = 0
            sel_end_exclusive = min(window, len(df))
    else:
        # Fallback: if no non-zero perturbation exists, use the first 20 trials (or all if shorter)
        sel_start = 0
        sel_end_exclusive = min(window, len(df))
    return sel_start, sel_end_exclusive


def analyze_all_participants(
    data_folder,
    block1_start,
    block1_end,
    block2_start,
    block2_end,
    block3_start,
    block3_end,
    alpha=1e-4,
    d_threshold=0.8,
    full_trial_num=800,
    mt_threshold=400,
    mt_fraction_threshold=0.35,
    test_asymptote=True,
    test_initial_rate=False,
    test_average=False,
    test_mt=False,
    test_rt_p=False,
    test_rt_cohend=False,
    test_rmse=False,
    test_rmse_all=False,
    rmse_threshold=9.5,
    # NEW: baseline-shift removal
    test_baseline_shift=False,
    baseline_shift_threshold=9.0,
    pick_portion=None,
    test_median=False,
    test_preperturb_std=False,
    preperturb_window=40,
    preperturb_std_threshold=3.5,
):
    results = {}

    for filename in os.listdir(data_folder):
        if not filename.lower().endswith(".csv"):
            continue

        participant_id = os.path.splitext(filename)[0]
        file_path = os.path.join(data_folder, filename)

        # Extract group number from filename (e.g., 'Group1')
        match = re.search(r'Group(\d+)', filename)
        group_num = int(match.group(1)) if match else None

        try:
            block_checks = {
                1: [(30, "block 1"), (20, "block 2"), (30, "block 3")],
                2: [(30, "block 1"), (-20, "block 2"), (30, "block 3")],
                3: [(10, "block 1"), (20, "block 2"), (10, "block 3")],
                4: [(10, "block 1"), (-20, "block 2"), (10, "block 3")],
                5: [(None, "block 1"), (20, "block 2"), (10, "block 3")],
                6: [(None, "block 1"), (-20, "block 2"), (10, "block 3")],
                7: [(30, "block 1"), (None, "block 2"), (30, "block 3")],
                8: [(10, "block 1"), (None, "block 2"), (10, "block 3")],
                9: [(20, "block 1"), (20, "block 2"), (20, "block 3")],
            }
            # block_checks = {
            #     1: [(30, "block 1")],
            #     2: [(30, "block 1")],
            #     3: [(10, "block 1")],
            #     4: [(10, "block 1")],
            #     5: [(20, "block 2")],
            #     6: [(-20, "block 2")],
            #     7: [(30, "block 1")],
            #     8: [(10, "block 1")],
            #     9: [(20, "block 1")],
            # }
            df = load_and_preprocess(file_path)
            reason = "all good"
            # --- 1) Full-trials check ---
            # ok_trials, reason_trials = check_full_trials(df, full_trial_num)
            # if not ok_trials:
            #     results[participant_id] = {"decision": False, "reason": reason_trials}
            #     continue

            if test_rmse:
                # compute selected‐range errors
                selected_error = (
                    df['angle_diff'][0:100]  # or your original error_check_idx_start:error_check_idx_end
                    - df['rotation'][0:100]
                )
                std_sel = selected_error.std()
                sel_no_out = selected_error[abs(selected_error - selected_error.mean()) <= 2 * std_sel]
                rmse_sel_no = np.sqrt((sel_no_out ** 2).mean())
                if rmse_sel_no > rmse_threshold:
                    results[participant_id] = {
                        "decision": False,
                        "reason": f"RMSE w/o outliers {rmse_sel_no:.2f} > threshold {rmse_threshold:.2f}"
                    }
                    continue

            if test_rmse_all:
                # --- NEW: all-trials RMSE (w/o outliers) vs. largest perturbation ---
                all_error = df['angle_diff'] - df['rotation']
                mean_all = all_error.mean()
                std_all = all_error.std()
                all_error_no = all_error[abs(all_error - mean_all) <= 3 * std_all]
                rmse_all = float(np.sqrt((all_error ** 2).mean()))

                # Guard against pathological filtering
                if len(all_error_no) == 0:
                    rmse_all_no = float('inf')
                else:
                    rmse_all_no = float(np.sqrt((all_error_no ** 2).mean()))

                largest_perturb = float(np.nanmax(np.abs(df['rotation'].values)))
                abs_rot = np.abs(pd.to_numeric(df['rotation'], errors='coerce').to_numpy(dtype=float))
                mask = abs_rot > 0
                smallest_perturb = float(np.min(abs_rot[mask])) if np.any(mask) else float('nan')
                average_perturb = float(np.mean(abs_rot[mask])) if np.any(mask) else float('nan')
                # print(participant_id, rmse_all_no, largest_perturb)
                if rmse_all_no > largest_perturb:
                    results[participant_id] = {
                        "decision": False,
                        "reason": (
                            f"All-trials RMSE w/o outliers {rmse_all_no:.2f} "
                            f"> largest perturbation {largest_perturb:.2f}"
                        ),
                    }
                    continue

            # --- 1.5) Remove unlearned participant (asymptote check) ---
            if test_asymptote:
                checks = block_checks.get(group_num, [])
                failed = False
                for check_val, block_name in checks:
                    if check_val is None:
                        continue  # skip this block
                    trials = abs(check_val)
                    threshold = check_val / 4
                    # determine start/end indices based on block_name
                    if block_name == "block 1":
                        end_idx = block1_end
                    elif block_name == "block 2":
                        end_idx = block2_end
                    else:
                        end_idx = block3_end
                    start_idx = end_idx - 30
                    testing_range = f"ending trials in {block_name}"
                    decision, reason = check_adaptation_level(
                        df, start_idx, end_idx, testing_range, pick_portion, mean_threshold=threshold
                    )
                    if not decision:
                        results[participant_id] = {"decision": False, "reason": reason}
                        failed = True
                        break
                if failed:
                    continue

            if test_baseline_shift:
                baseline_val = float('nan')
                if 'baseline_mean' in df.columns:
                    try:
                        _bvals = pd.to_numeric(df['baseline_mean'], errors='coerce').dropna()
                        if len(_bvals) > 0:
                            baseline_val = float(_bvals.iloc[0])
                    except Exception:
                        baseline_val = float('nan')

                # Only enforce if we have a valid value
                if np.isfinite(baseline_val) and abs(baseline_val) > float(baseline_shift_threshold):
                    results[participant_id] = {
                        "decision": False,
                        "reason": (
                            f"baseline shift {baseline_val:.2f} > "
                            f"threshold {float(baseline_shift_threshold):.2f}"
                        ),
                    }
                    continue

            if test_average:
                checks = block_checks.get(group_num, [])
                failed = False
                for check_val, block_name in checks:
                    if check_val is None:
                        continue  # skip this block

                    threshold = check_val / 4  # keep the same threshold rule
                    if block_name == "block 1":
                        start_idx, end_idx = block1_start, block1_end
                    elif block_name == "block 2":
                        start_idx, end_idx = block2_start, block2_end
                    else:
                        start_idx, end_idx = block3_start, block3_end

                    testing_range = f"average adaptation in {block_name} (online_fb only)"
                    decision, reason = check_block_average(
                        df, start_idx, end_idx, testing_range, mean_threshold=threshold
                    )
                    if not decision:
                        results[participant_id] = {"decision": False, "reason": reason}
                        failed = True
                        break
                if failed:
                    continue

            if test_preperturb_std and (preperturb_std_threshold is not None):
                sel_start, sel_end_exclusive = _preperturb_window(df, window=preperturb_window)
                # Selected error is angle_diff - rotation over the pre-perturbation window
                selected_error = (
                    df['angle_diff'].iloc[sel_start:sel_end_exclusive]
                    - df['rotation'].iloc[sel_start:sel_end_exclusive]
                )
                std_sel = float(selected_error.std())

                print(filename, std_sel, sel_start, sel_end_exclusive)
                if np.isnan(std_sel) or (sel_end_exclusive - sel_start) == 0:
                    results[participant_id] = {
                        "decision": False,
                        "reason": "pre-perturbation window empty or NaN STD"
                    }
                    continue

                if std_sel > float(preperturb_std_threshold):
                    results[participant_id] = {
                        "decision": False,
                        "reason": (
                            f"pre-perturbation STD {std_sel:.2f} > threshold {float(preperturb_std_threshold):.2f} "
                            f"(idx {sel_start}-{sel_end_exclusive-1})"
                        )
                    }
                    continue

            if test_median:
                checks = block_checks.get(group_num, [])
                failed = False
                for check_val, block_name in checks:
                    if check_val is None:
                        continue  # skip this block

                    threshold = check_val / 4  # keep the same threshold rule
                    if block_name == "block 1":
                        start_idx, end_idx = block1_start, block1_end
                    elif block_name == "block 2":
                        start_idx, end_idx = block2_start, block2_end
                    else:
                        start_idx, end_idx = block3_start, block3_end

                    testing_range = f"average adaptation in {block_name} (online_fb only)"
                    decision, reason = check_block_median(
                        df, start_idx, end_idx, testing_range, median_threshold=threshold
                    )
                    if not decision:
                        results[participant_id] = {"decision": False, "reason": reason}
                        failed = True
                        break
                if failed:
                    continue

            if test_initial_rate:
                # Similar block-specific logic can be applied here if needed
                start_offset = 15
                if group_num in [1, 2, 3, 4, 7, 8, 9]:
                    start_idx = block1_start + start_offset
                    end_idx = block1_start + 2 * start_offset
                    testing_range = "starting trials in block 1"
                else:
                    start_idx = block2_start + start_offset
                    end_idx = block2_start + 2 * start_offset
                    testing_range = "starting trials in block 2"

                decision, reason = check_adaptation_level(
                    df, start_idx, end_idx, testing_range, pick_portion
                )
                if not decision:
                    results[participant_id] = {"decision": False, "reason": reason}
                    continue

            # --- 2) MT-proportion check (if enabled) ---
            if test_mt:
                ok_mt, reason_mt = check_mt_proportion(df, mt_threshold, mt_fraction_threshold)
                if not ok_mt:
                    results[participant_id] = {"decision": False, "reason": reason_mt}
                    continue

            # --- 3) RT-based checks (paired t-test / Cohen's d) ---

            if test_rt_p or test_rt_cohend:
                total_trials = len(df)
                if block2_end > total_trials or block1_end > total_trials:
                    reason = (
                        f"not enough trials: total_trials={total_trials}, "
                        f"block2_end={block2_end}"
                    )
                    results[participant_id] = {"decision": False, "reason": reason}
                    continue
                # Extract RT groups (block1 & block2)
                block1, block2 = extract_paired_groups(
                    df, block1_start, block1_end, block2_start, block2_end
                )
                # Optionally extract block3 for future analyses
                # block1, block2, block3 = extract_triplet_groups(
                #     df, block1_start, block1_end, block2_start, block2_end, block3_start, block3_end
                # )

                t_stat, p_value = run_paired_ttest(block1, block2)
                d_value = compute_cohens_d(block1, block2)

            rt_reasons = []
            if test_rt_p and (p_value < alpha):
                rt_reasons.append(f"p_value={p_value:.2e} < alpha={alpha}")
            if test_rt_cohend and (abs(d_value) > d_threshold):
                rt_reasons.append(f"|d|={abs(d_value):.2f} > d_threshold={d_threshold}")

            if rt_reasons:
                results[participant_id] = {"decision": False, "reason": " and ".join(rt_reasons)}
            else:
                results[participant_id] = {"decision": True, "reason": reason}

        except pd.errors.EmptyDataError:
            results[participant_id] = {"decision": False, "reason": "empty or corrupted file"}
        except Exception as e:
            results[participant_id] = {"decision": False, "reason": f"error: {e}"}

    return results


def save_results_to_json(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    mode = "raw_data_without_outlier"
    data_folder = f"processed_data/{mode}"
    result_folder = "plotting_helper_files"
    os.makedirs(result_folder, exist_ok=True)

    # --- Define block indices (zero-indexed, end-exclusive) ---
    block1_start, block1_end = 100, 210
    block2_start, block2_end = 380, 490
    block3_start, block3_end = 660, 770

    # --- Thresholds ---
    alpha = 1e-4
    d_threshold = 0.8
    full_trial_num = 770
    mt_threshold = 400
    mt_fraction_threshold = 0.35

    # --- Enable/disable checks ---
    test_mt = False
    test_rt_p = False
    test_rt_cohend = False
    test_asymptote = False
    test_initial_rate = False
    test_rmse = False
    test_rmse_all = False
    pick_portion = None
    test_average = False
    test_median = False
    test_preperturb_std = False
    test_baseline_shift = False
    baseline_shift_threshold = 10.0
    preperturb_window = 30
    preperturb_std_threshold = 3.7

    #work: 30, 3.7; 25, 3.5
    #kinda work (use test_rmse_all): directly change it to True
    #kinda work (use test_rmse): rmse wo 9.5

    results = analyze_all_participants(
        data_folder,
        block1_start,
        block1_end,
        block2_start,
        block2_end,
        block3_start,
        block3_end,
        alpha=alpha,
        d_threshold=d_threshold,
        full_trial_num=full_trial_num,
        mt_threshold=mt_threshold,
        mt_fraction_threshold=mt_fraction_threshold,
        test_asymptote=test_asymptote,
        test_initial_rate=test_initial_rate,
                test_average=test_average,
        test_mt=test_mt,
        test_rt_p=test_rt_p,
        test_rt_cohend=test_rt_cohend,
        test_rmse=test_rmse,
        test_rmse_all=test_rmse_all,
        # NEW: baseline-shift removal
        test_baseline_shift=test_baseline_shift,
        baseline_shift_threshold=baseline_shift_threshold,
        pick_portion=pick_portion,
        test_median=test_median,
        test_preperturb_std=test_preperturb_std,
        preperturb_window=preperturb_window,
        preperturb_std_threshold=preperturb_std_threshold
    )

    output_path = os.path.join(result_folder, f"Remov_deci_all_blocks_{mode}.json")
    save_results_to_json(results, output_path)
    print(f"Results saved to {output_path}")