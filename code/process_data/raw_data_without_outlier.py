import os
from tqdm import tqdm
import pandas as pd
import util
import numpy as np


def process_single_file(input_path: str, output_path: str) -> None:
    """
    Read a CSV from input_path, add the 'angle_diff' column using util.compute_angle_diff,
    apply outlier removal using util.detect_and_replace_outliers, and save the modified
    DataFrame to output_path.
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(input_path)

    # Compute and add the 'angle_diff' column
    df = util.compute_angle_diff(df)

    df = df.iloc[30:].reset_index(drop=True)
    if df.empty:
        df.to_csv(output_path, index=False)
        return

    # Updated: apply outlier removal to the 'angle_diff' column
    # This will replace any angle_diff outside [-60, 60] by the mean of its neighbors,
    # and return the percentage of points that were treated as outliers.
    percent_outliers = util.detect_and_replace_outliers_hampel(
        df,
        window_size=9,
        n_sigmas=3.0,
        replace_with="median",
    )
    print(f"[INFO] {os.path.basename(input_path)}: {percent_outliers:.2f}% outliers replaced")
    df = util.adjust_angle_diff_baseline(df, 0, 120)
    df["trialNum"] = np.arange(1, len(df) + 1, dtype=int)

    # Save the modified DataFrame to the output path
    df.to_csv(output_path, index=False)


def process_all_csvs(data_folder: str, result_folder: str) -> None:
    """
    Iterate through all .csv files in data_folder, process each one by adding 'angle_diff',
    removing outliers, and save the result into result_folder with the same filename.
    """
    os.makedirs(result_folder, exist_ok=True)

    for fname in tqdm(os.listdir(data_folder)):
        if not fname.endswith('.csv'):
            continue

        input_path = os.path.join(data_folder, fname)
        output_path = os.path.join(result_folder, fname)

        process_single_file(input_path, output_path)


if __name__ == "__main__":
    data_folder = 'data/trials'
    result_folder = 'processed_data/raw_data_without_outlier'
    process_all_csvs(data_folder, result_folder)
