import os
from tqdm import tqdm
import pandas as pd
import util


def process_single_file(input_path: str, output_path: str) -> None:
    """
    Read a CSV from input_path, add the 'angle_diff' column using util.compute_angle_diff,
    and save the modified DataFrame to output_path.
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(input_path)

    # Compute and add the 'angle_diff' column
    df = util.compute_angle_diff(df)

    # Save the modified DataFrame to the output path
    df.to_csv(output_path, index=False)


def process_all_csvs(data_folder: str, result_folder: str) -> None:
    """
    Iterate through all .csv files in data_folder, process each one by adding 'angle_diff',
    and save the result into result_folder with the same filename.
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
    result_folder = 'processed_data/raw_data'
    process_all_csvs(data_folder, result_folder)
