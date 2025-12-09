import os
from tqdm import tqdm
import pandas as pd
import util
import numpy as np


def process_single_file(
    input_path: str,
    output_path: str,
    remove_first_n: int,
    use_adjust_baseline: bool,
    baseline_start_idx: int,
    baseline_end_idx: int,
) -> None:
    """
    Read a CSV from input_path, compute angle_diff, optionally drop
    the first N trials, run outlier removal, optionally apply baseline
    adjustment, and save the result to output_path.
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(input_path)

    # Compute and add the 'angle_diff' column
    df = util.compute_angle_diff(df)

    # Optional: drop first N trials (e.g., warmup/baseline)
    if remove_first_n > 0:
        df = df.iloc[remove_first_n:].reset_index(drop=True)

    # If nothing left, just save the empty frame
    if df.empty:
        df.to_csv(output_path, index=False)
        return

    # Outlier removal on angle_diff via Hampel filter
    percent_outliers = util.detect_and_replace_outliers_hampel(
        df,
        window_size=9,
        n_sigmas=3.0,
        replace_with="median",
    )
    print(f"[INFO] {os.path.basename(input_path)}: {percent_outliers:.2f}% outliers replaced")

    # Optional baseline adjustment on angle_diff
    if use_adjust_baseline:
        df = util.adjust_angle_diff_baseline(
            df,
            baseline_start_idx,
            baseline_end_idx,
        )

    # Rebuild trialNum after any trimming
    df["trialNum"] = np.arange(1, len(df) + 1, dtype=int)

    # Save the modified DataFrame to the output path
    df.to_csv(output_path, index=False)


def process_all_csvs(
    data_folder: str,
    result_folder: str,
    remove_first_n: int,
    use_adjust_baseline: bool,
    baseline_start_idx: int,
    baseline_end_idx: int,
) -> None:
    """
    Iterate through all .csv files in data_folder, process each one,
    and save to result_folder with the same filename.
    """
    os.makedirs(result_folder, exist_ok=True)

    for fname in tqdm(os.listdir(data_folder)):
        if not fname.endswith(".csv"):
            continue

        input_path = os.path.join(data_folder, fname)
        output_path = os.path.join(result_folder, fname)

        process_single_file(
            input_path=input_path,
            output_path=output_path,
            remove_first_n=remove_first_n,
            use_adjust_baseline=use_adjust_baseline,
            baseline_start_idx=baseline_start_idx,
            baseline_end_idx=baseline_end_idx,
        )




if __name__ == "__main__":

    # Configuration per “day”
    DAY_PROCESSING_CONFIG = {
        "day1": {
            "data_folder": "data/trials_day1",
            "result_folder": "processed_data/raw_data_without_outlier_day1",
            "remove_first_n": 30,       # drop first 30 trials on day1
            "use_adjust_baseline": False,
            "baseline_start_idx": 0,
            "baseline_end_idx": 120,
        },
        "day2": {
            "data_folder": "data/trials_day2",
            "result_folder": "processed_data/raw_data_without_outlier_day2",
            "remove_first_n": 0,        # keep all trials on day2
            "use_adjust_baseline": False,
            "baseline_start_idx": 0,
            "baseline_end_idx": 0,
        },
    }

    for day_name, cfg in DAY_PROCESSING_CONFIG.items():
        print(f"[INFO] Processing {day_name} from {cfg['data_folder']} -> {cfg['result_folder']}")
        process_all_csvs(
            data_folder=cfg["data_folder"],
            result_folder=cfg["result_folder"],
            remove_first_n=cfg["remove_first_n"],
            use_adjust_baseline=cfg["use_adjust_baseline"],
            baseline_start_idx=cfg["baseline_start_idx"],
            baseline_end_idx=cfg["baseline_end_idx"],
        )
