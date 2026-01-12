import os
import json
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a single participant CSV and return a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def check_full_trials(df: pd.DataFrame, full_trial_num: int) -> Tuple[bool, str]:
    """
    Check if the number of trials in df equals full_trial_num.

    Returns
    -------
    ok : bool
        True if the number of rows in df == full_trial_num, otherwise False.
    reason : str
        A short string explaining why it failed (empty if ok).
    """
    total_trials = len(df)
    if total_trials != full_trial_num:
        reason = (
            f"incomplete trials: total_trials={total_trials} "
            f"!= full_trial_num={full_trial_num}"
        )
        return False, reason
    return True, ""


def check_rmse_selected_range(
    df: pd.DataFrame,
    rmse_threshold: float,
    start_idx: int = 0,
    end_idx: int = 100,
    error_col: str = "angle_diff",
    model_col: str = "rotation",
) -> Tuple[bool, str]:
    """
    Compute RMSE over [start_idx:end_idx] between df[error_col] and df[model_col],
    remove 2-sigma outliers, and compare to rmse_threshold.

    Returns
    -------
    ok : bool
        True if RMSE <= threshold, otherwise False.
    reason : str
        Explanation string.
    """
    n_rows = len(df)
    if n_rows == 0:
        return False, "RMSE check: empty dataframe"

    # Make sure the slice is valid even if df has fewer rows than end_idx
    actual_end_idx = min(end_idx, n_rows)
    if start_idx >= actual_end_idx:
        return False, (
            f"RMSE check: invalid slice start_idx={start_idx}, "
            f"end_idx={end_idx}, n_rows={n_rows}"
        )

    if error_col not in df.columns or model_col not in df.columns:
        return False, (
            f"RMSE check: missing columns (need '{error_col}' and '{model_col}')"
        )

    selected_err = df[error_col].iloc[start_idx:actual_end_idx] - df[model_col].iloc[
        start_idx:actual_end_idx
    ]

    if selected_err.empty:
        return False, "RMSE check: no data in selected range"

    # 2-sigma outlier removal
    mean_val = selected_err.mean()
    std_val = selected_err.std()

    if std_val == 0:
        rmse_val = float(np.sqrt((selected_err ** 2).mean()))
        if rmse_val > rmse_threshold:
            return (
                False,
                f"RMSE (no outliers, std=0) {rmse_val:.2f} > threshold {rmse_threshold:.2f}",
            )
        return True, f"pass RMSE (std=0): {rmse_val:.2f} <= {rmse_threshold:.2f}"

    mask_no_out = (selected_err - mean_val).abs() <= (2.0 * std_val)
    sel_no_out = selected_err[mask_no_out]

    if sel_no_out.empty:
        return False, "RMSE check: all points removed as outliers"

    rmse_val = float(np.sqrt((sel_no_out ** 2).mean()))

    if rmse_val > rmse_threshold:
        return (
            False,
            f"RMSE w/o outliers {rmse_val:.2f} > threshold {rmse_threshold:.2f}",
        )

    return True, f"pass RMSE: {rmse_val:.2f} <= {rmse_threshold:.2f}"


def analyze_all_participants(
    data_folder: str,
    full_trial_num: int,
    rmse_threshold: float,
    test_full_trials: bool = True,
    test_rmse: bool = True,
    rmse_start_idx: int = 0,
    rmse_end_idx: int = 100,
    error_col: str = "angle_diff",
    model_col: str = "rotation",
) -> Dict[str, Dict[str, object]]:

    results: Dict[str, Dict[str, object]] = {}

    for filename in os.listdir(data_folder):
        if not filename.lower().endswith(".csv"):
            continue

        participant_id = os.path.splitext(filename)[0]
        file_path = os.path.join(data_folder, filename)

        # Optional: extract group number if the filename contains 'GroupX'
        match = re.search(r"Group(\d+)", filename)
        _group_num = int(match.group(1)) if match else None
        # _group_num is unused here; kept in case you want group-based logic later

        df = load_csv(file_path)

        reasons = []

        # 1) Full-trials check
        if test_full_trials:
            ok_trials, reason_trials = check_full_trials(df, full_trial_num)
            if not ok_trials:
                results[participant_id] = {
                    "decision": False,
                    "reason": reason_trials,
                }
                continue
            reasons.append("pass full_trials")

        # 2) RMSE check (selected range)
        if test_rmse:
            ok_rmse, reason_rmse = check_rmse_selected_range(
                df,
                rmse_threshold=rmse_threshold,
                start_idx=rmse_start_idx,
                end_idx=rmse_end_idx,
                error_col=error_col,
                model_col=model_col,
            )
            if not ok_rmse:
                results[participant_id] = {
                    "decision": False,
                    "reason": reason_rmse,
                }
                continue
            reasons.append(reason_rmse)

        # If no tests are enabled, we still accept the participant by default
        if not test_full_trials and not test_rmse:
            reasons.append("no tests enabled; accepted by default")

        reason_str = "; ".join(reasons) if reasons else "accepted"
        results[participant_id] = {
            "decision": True,
            "reason": reason_str,
        }

    return results


def save_results_to_json(results: Dict[str, Dict[str, object]], output_path: str) -> None:
    """
    Save results dict as JSON to output_path.
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Where to save the JSON of keep/remove decisions (shared for both days)
    result_folder = "plotting_helper_files"
    mode = "no_bad_mt_data_150"
    os.makedirs(result_folder, exist_ok=True)

    # ---------------------------
    # Day 1 configuration
    # ---------------------------
    mode_day1 = f"{mode}_day1"
    data_folder_day1 = f"processed_data/{mode_day1}"

    # Full trial count for Day 1
    full_trial_num_day1 = 480

    # RMSE parameters for Day 1
    rmse_threshold_day1 = 9.5
    rmse_start_idx_day1 = 0
    rmse_end_idx_day1 = 100
    error_col_day1 = "angle_diff"
    model_col_day1 = "rotation"

    # Enable/disable checks for Day 1
    test_full_trials_day1 = True
    test_rmse_day1 = True

    results_day1 = analyze_all_participants(
        data_folder=data_folder_day1,
        full_trial_num=full_trial_num_day1,
        rmse_threshold=rmse_threshold_day1,
        test_full_trials=test_full_trials_day1,
        test_rmse=test_rmse_day1,
        rmse_start_idx=rmse_start_idx_day1,
        rmse_end_idx=rmse_end_idx_day1,
        error_col=error_col_day1,
        model_col=model_col_day1,
    )

    output_path_day1 = os.path.join(
        result_folder,
        f"Remov_deci_all_blocks_{mode_day1}.json",
    )
    save_results_to_json(results_day1, output_path_day1)
    print(f"Results for Day 1 saved to {output_path_day1}")

    # ---------------------------
    # Day 2 configuration
    # ---------------------------
    mode_day2 = f"{mode}_day2"
    data_folder_day2 = f"processed_data/{mode_day2}"

    # Full trial count for Day 2
    full_trial_num_day2 = 420

    # RMSE parameters for Day 2
    rmse_threshold_day2 = 9.5
    rmse_start_idx_day2 = 0
    rmse_end_idx_day2 = 100
    error_col_day2 = "angle_diff"
    model_col_day2 = "rotation"

    # Enable/disable checks for Day 2
    test_full_trials_day2 = True
    test_rmse_day2 = False

    results_day2 = analyze_all_participants(
        data_folder=data_folder_day2,
        full_trial_num=full_trial_num_day2,
        rmse_threshold=rmse_threshold_day2,
        test_full_trials=test_full_trials_day2,
        test_rmse=test_rmse_day2,
        rmse_start_idx=rmse_start_idx_day2,
        rmse_end_idx=rmse_end_idx_day2,
        error_col=error_col_day2,
        model_col=model_col_day2,
    )

    output_path_day2 = os.path.join(
        result_folder,
        f"Remov_deci_all_blocks_{mode_day2}.json",
    )
    save_results_to_json(results_day2, output_path_day2)
    print(f"Results for Day 2 saved to {output_path_day2}")
