import os
import json
from typing import Dict

import numpy as np
import pandas as pd


def load_trial_to_workspace_map(json_path: str) -> Dict[int, int]:
    """
    Load a JSON like group1_day1.json and build a mapping from trialNum to workspace.

    The JSON has keys:
        "trialnum": { "0": 1, "1": 2, ... }
        "workspace": { "0": 1, "1": 1, ... }

    We pair them index-wise so that each trialNum gets its corresponding workspace.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    trialnum_dict = data["trialnum"]
    workspace_dict = data["workspace"]

    trial_to_workspace = {}

    for idx_str, trial_num in trialnum_dict.items():
        trial_int = int(trial_num)
        workspace_val = int(workspace_dict[idx_str])
        trial_to_workspace[trial_int] = workspace_val

    return trial_to_workspace


def add_workspace_to_csv(csv_path: str, trial_to_workspace: Dict[int, int]) -> None:
    """
    Read one CSV, ensure it has a trialNum column, and add a workspace column
    based on trial_to_workspace mapping. The result overwrites the same CSV.
    """
    print("Processing CSV:", csv_path)
    df = pd.read_csv(csv_path)

    if "trialNum" not in df.columns:
        df["trialNum"] = np.arange(1, len(df) + 1, dtype=int)

    df["workspace"] = df["trialNum"].map(trial_to_workspace)

    if df["workspace"].isna().any():
        missing = df.loc[df["workspace"].isna(), "trialNum"].unique()
        print("Warning: some trialNum values have no workspace mapping:", missing)

    df.to_csv(csv_path, index=False)


def add_workspace_to_folder(folder_path: str, json_path: str) -> None:
    """
    Given a folder and a JSON config, read all .csv files in the folder and
    add a workspace column to each, using the trialNum -> workspace mapping
    from the JSON file.
    """
    trial_to_workspace = load_trial_to_workspace_map(json_path)

    for name in os.listdir(folder_path):
        if not name.lower().endswith(".csv"):
            continue
        csv_path = os.path.join(folder_path, name)
        if not os.path.isfile(csv_path):
            continue
        add_workspace_to_csv(csv_path, trial_to_workspace)


if __name__ == "__main__":
    folder_path = "data/trials_day1_wait_modified"
    json_path = "group1_day1.json"

    add_workspace_to_folder(folder_path, json_path)
