import os
from tqdm import tqdm
import pandas as pd
import numpy as np

MODE = "mean"


def _assert_constant(block: pd.DataFrame, col: str) -> None:
    values = block[col].unique()
    assert len(values) == 1, f"{col} not constant in cycle: {values}"


def group_trials_to_cycles(df: pd.DataFrame, group_size: int = 3) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    total = len(df)
    if total % group_size != 0:
        raise ValueError(
            f"\ntrial count {total} not divisible by {group_size}\n"
            f"################################################\n"
        )

    rows = []
    for start in range(0, total, group_size):
        block = df.iloc[start:start + group_size].reset_index(drop=True)

        assert block["target_angle"].nunique() == group_size, (
            f"target_angle not unique in cycle: {block['target_angle'].tolist()}"
        )

        for col in ["trial_type", "rotation", "workspace", "day"]:
            _assert_constant(block, col)

        row = {
            "cycleNum": int(start / group_size) + 1,
            "currentDate": block["currentDate"].iloc[0] if "currentDate" in block.columns else np.nan,
            "trial_type": block["trial_type"].iloc[0] if "trial_type" in block.columns else np.nan,
            "rotation": block["rotation"].iloc[0] if "rotation" in block.columns else np.nan,
            "workspace": block["workspace"].iloc[0] if "workspace" in block.columns else np.nan,
            "day": block["day"].iloc[0] if "day" in block.columns else np.nan,
        }

        for col in ["rt", "mt", "search_time", "angle_diff"]:
            if col in block.columns:
                if MODE == "mean":
                    row[col] = pd.to_numeric(block[col], errors="raise").mean()
                elif MODE == "median":
                    row[col] = pd.to_numeric(block[col], errors="raise").median()
            else:
                assert False, f"{col} not in dataframe columns"

        rows.append(row)

    return pd.DataFrame(rows)


def process_single_file(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    try:
        df_cycle = group_trials_to_cycles(df, group_size=3)
    except (ValueError) as exc:
        print(f"\n################################################\n"
              f"[WARN] Skipping {os.path.basename(input_path)}: {exc}")
        return

    drop_cols = [
        "target_angle",
        "hand_fb_angle",
        "reach_feedback",
        "movement_trajectory",
        "valid_hand_fb_angle",
    ]
    for col in drop_cols:
        if col in df_cycle.columns:
            df_cycle = df_cycle.drop(columns=[col])

    df_cycle.to_csv(output_path, index=False)


def process_all_csvs(data_folder: str, result_folder: str) -> None:
    os.makedirs(result_folder, exist_ok=True)

    for fname in tqdm(os.listdir(data_folder)):
        if not fname.endswith(".csv"):
            continue

        input_path = os.path.join(data_folder, fname)
        output_path = os.path.join(result_folder, fname)

        process_single_file(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    mode = "no_bad_mt_data_150"
    # mode = "raw_data_without_outlier"
    days = [1, 2]

    for day in days:
        mode_day = f"{mode}_day{day}"
        data_dir = os.path.join("processed_data", mode_day)
        out_dir = os.path.join("processed_data", f"{mode}_day{day}_cycle")

        print(f"[INFO] Processing cycles for {mode_day} -> {out_dir}")
        process_all_csvs(data_folder=data_dir, result_folder=out_dir)
