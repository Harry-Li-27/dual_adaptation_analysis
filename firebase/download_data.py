import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import json
import os

# Initialize Firebase Admin SDK
cred = credentials.Certificate('firebase/dual-adaptation-firebase-adminsdk-fbsvc-7f56dc27e1.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

# Set the default experiment label.
# If set to a value (e.g., "something"), that value will be used.
# If set to None, then the experiment label will be taken from the subject's "experiment" field.
default_experiment_label = "experiment_v7"  # Change this to None if you want to use subject_info["experiment"]

# Required number of trials per day
REQUIRED_DAY1_TRIALS = 510
REQUIRED_DAY2_TRIALS = 420


def create_subject_file_mapping(subject_info, subject_id, day1_trial_count, day2_trial_count, finished_time_day1, finished_time_day2):
    """
    Build the mapping entry for a subject:
    value includes:
      1. id: the Firestore subject ID
      2. completion_status: describes which days reached required trials
      3. day1_trials: number of trials in Trials_day1
      4. day2_trials: number of trials in Trials_day2
      5. finished_time_day1: last currentDate in day1
      6. finished_time_day2: last currentDate in day2
    """
    failed_attempts = subject_info.get('failed_attempts')

    day1_complete = day1_trial_count == REQUIRED_DAY1_TRIALS
    day2_complete = day2_trial_count == REQUIRED_DAY2_TRIALS

    if day1_complete and day2_complete:
        status = 'both complete'
    elif day1_complete and not day2_complete:
        status = 'day1 complete, day2 not'
    elif day2_complete and not day1_complete:
        status = 'day2 complete, day1 not'
    elif subject_info.get('failed_reason'):
        status = subject_info.get('failed_reason')
    elif failed_attempts:
        status = f"failed {failed_attempts} times in baseline"
    else:
        status = 'unknown'

    return {
        'id': subject_id,
        'completion_status': status,
        'day1_trials': day1_trial_count,
        'day2_trials': day2_trial_count,
        'finished_time_day1': finished_time_day1,
        'finished_time_day2': finished_time_day2,
    }


def fetch_data():
    subjects_collection = db.collection('Subjects')
    trials_day1_collection = db.collection('Trials_day1')
    trials_day2_collection = db.collection('Trials_day2')

    # Folder for JSON files (subject information)
    json_folder = os.path.join('data', 'participant_information')
    os.makedirs(json_folder, exist_ok=True)

    # CSV files for day1 and day2 in separate folders
    csv_folder_day1 = os.path.join('data', "trials_day1")
    os.makedirs(csv_folder_day1, exist_ok=True)

    csv_folder_day2 = os.path.join('data', "trials_day2")
    os.makedirs(csv_folder_day2, exist_ok=True)

    subjects_docs = subjects_collection.stream()

    # Dictionary to keep track of participant numbers per group.
    group_counters = {}
    # Track invalid entries
    invalid_counter = 0
    # Dictionary to map filename prefix to info
    subject_file_mapping = {}

    for subject_doc in subjects_docs:
        subject_id = subject_doc.id
        subject_info = subject_doc.to_dict()
        group_choice = subject_info.get("groupChoice", "default")

        # Determine experiment label: if default_experiment_label is not None, use it;
        # otherwise, attempt to use subject_info's "experiment" field.
        experiment_label = subject_info.get("experimentID", default_experiment_label)

        # Fetch trial data for this subject from day1 and day2 using subject_id
        day1_docs = list(trials_day1_collection.where('id', '==', subject_id).stream())
        day2_docs = list(trials_day2_collection.where('id', '==', subject_id).stream())

        records_day1 = []
        for trial_doc in day1_docs:
            trial_data = trial_doc.to_dict()
            trial_list_length = len(trial_data.get('trialNum', []))
            for i in range(trial_list_length):
                workspace_list = trial_data.get('workspace', [])
                workspace_value = workspace_list[i] if i < len(workspace_list) else None
                record = {
                    'trialNum': trial_data.get('trialNum', [None])[i],
                    'currentDate': trial_data.get('currentDate', [None])[i],
                    'target_angle': trial_data.get('target_angle', [None])[i],
                    'trial_type': trial_data.get('trial_type', [None])[i],
                    'rotation': trial_data.get('rotation', [None])[i],
                    'hand_fb_angle': trial_data.get('hand_fb_angle', [None])[i],
                    'rt': trial_data.get('rt', [None])[i],
                    'mt': trial_data.get('mt', [None])[i],
                    'search_time': trial_data.get('search_time', [None])[i],
                    'reach_feedback': trial_data.get('reach_feedback', [None])[i],
                    'movement_trajectory': json.dumps(trial_data['movement_trajectory'][i])
                        if 'movement_trajectory' in trial_data
                        and trial_data.get('movement_trajectory')
                        and i < len(trial_data['movement_trajectory'])
                        else 'N/A',
                    'workspace': workspace_value,
                    'day': 1,
                }
                records_day1.append(record)

        records_day2 = []
        for trial_doc in day2_docs:
            trial_data = trial_doc.to_dict()
            trial_list_length = len(trial_data.get('trialNum', []))
            for i in range(trial_list_length):
                workspace_list = trial_data.get('workspace', [])
                workspace_value = workspace_list[i] if i < len(workspace_list) else None
                record = {
                    'trialNum': trial_data.get('trialNum', [None])[i],
                    'currentDate': trial_data.get('currentDate', [None])[i],
                    'target_angle': trial_data.get('target_angle', [None])[i],
                    'trial_type': trial_data.get('trial_type', [None])[i],
                    'rotation': trial_data.get('rotation', [None])[i],
                    'hand_fb_angle': trial_data.get('hand_fb_angle', [None])[i],
                    'rt': trial_data.get('rt', [None])[i],
                    'mt': trial_data.get('mt', [None])[i],
                    'search_time': trial_data.get('search_time', [None])[i],
                    'reach_feedback': trial_data.get('reach_feedback', [None])[i],
                    'movement_trajectory': json.dumps(trial_data['movement_trajectory'][i])
                        if 'movement_trajectory' in trial_data
                        and trial_data.get('movement_trajectory')
                        and i < len(trial_data['movement_trajectory'])
                        else 'N/A',
                    'workspace': workspace_value,
                    'day': 2,
                }
                records_day2.append(record)

        records = records_day1 + records_day2

        # If no trial data is found for this subject, record mapping as "no result" and skip processing.
        if not records:
            invalid_counter += 1
            fake_key = f"invalid{invalid_counter}"
            subject_file_mapping[fake_key] = {
                'id': subject_id,
                'completion_status': 'no_trials'
            }
            print(f"Subject {subject_id} has no trial data; recorded as {fake_key}")
            continue

        day1_trial_count = len(records_day1)
        day2_trial_count = len(records_day2)

        finished_time_day1 = ""
        finished_time_day2 = ""

        if records_day1:
            date_values_day1 = [r.get('currentDate') for r in records_day1 if r.get('currentDate') is not None]
            if date_values_day1:
                finished_time_day1 = max(date_values_day1)

        if records_day2:
            date_values_day2 = [r.get('currentDate') for r in records_day2 if r.get('currentDate') is not None]
            if date_values_day2:
                finished_time_day2 = max(date_values_day2)

        # Update participant number for the group.
        if group_choice not in group_counters:
            group_counters[group_choice] = 1
        participant_number = group_counters[group_choice]
        group_counters[group_choice] += 1

        # Create filename prefix using the naming format:
        # "S_{experiment_label}_Group{groupChoice}_{participant_number}"
        filename_prefix = f"S-{experiment_label}-Group{group_choice}-ID{participant_number}"

        # Save trial data for day1 and day2 into separate CSV folders
        if records_day1:
            df_day1 = pd.DataFrame(records_day1)
            csv_file_path_day1 = os.path.join(csv_folder_day1, f"{filename_prefix}.csv")
            df_day1.to_csv(csv_file_path_day1, index=False)
            print(f'Trial data for subject {subject_id} day1 written to CSV at {csv_file_path_day1}')

        if records_day2:
            df_day2 = pd.DataFrame(records_day2)
            csv_file_path_day2 = os.path.join(csv_folder_day2, f"{filename_prefix}.csv")
            df_day2.to_csv(csv_file_path_day2, index=False)
            print(f'Trial data for subject {subject_id} day2 written to CSV at {csv_file_path_day2}')

        # Save subject information to JSON in the target JSON folder
        json_file_path = os.path.join(json_folder, f"{filename_prefix}.json")
        with open(json_file_path, 'w') as f:
            json.dump(subject_info, f)
        print(f'Subject info for subject {subject_id} written to JSON at {json_file_path}')

        # Use filename_prefix as key, map to dict with id, completion_status, day counts, and finished_time
        subject_file_mapping[filename_prefix] = create_subject_file_mapping(
            subject_info,
            subject_id,
            day1_trial_count,
            day2_trial_count,
            finished_time_day1,
            finished_time_day2,
        )

    # Sort mapping by the keys (filename_prefix)
    ordered_mapping = dict(sorted(subject_file_mapping.items(), key=lambda kv: kv[0]))
    # Save the mapping to JSON
    mapping_file_path = os.path.join(json_folder, "subject_file_mapping.json")
    with open(mapping_file_path, 'w') as f:
        json.dump(ordered_mapping, f, indent=4)
    print(f'Subject to filename mapping saved to {mapping_file_path}')


def main():
    fetch_data()


if __name__ == "__main__":
    main()
