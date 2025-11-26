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

def create_subject_file_mapping(subject_info, required_trail_num, subject_id, csv_length):
    """
    Build the mapping entry for a subject:
    value includes:
      1. id: the Firestore subject ID
      2. completion_status: 'completed' if currTrial == last_trial_num, else failed_reason
    """
    curr_trial = subject_info.get('currTrial')
    failed_attempts = subject_info.get('failed_attempts')
    # Determine completion status
    if curr_trial == required_trail_num:
        status = 'completed'
    elif csv_length == required_trail_num:
        status = "completed, but no questionnaire"
    elif subject_info.get('failed_reason'):
        status = subject_info.get('failed_reason')
    elif failed_attempts:
        status = f"failed {failed_attempts} times in baseline"
    else:
        status = 'unknown'
    return {'id': subject_id, 'completion_status': status}


def fetch_data(required_trail_num=800):
    subjects_collection = db.collection('Subjects')
    trials_collection = db.collection('Trials')

    # Folder for JSON files (subject information)
    json_folder = os.path.join('data', 'participant_information')
    os.makedirs(json_folder, exist_ok=True)
    
    # CSV files remain in the original folder
    csv_folder = os.path.join('data', "trials")
    os.makedirs(csv_folder, exist_ok=True)

    subjects_docs = subjects_collection.stream()
    
    # Dictionary to keep track of participant numbers per group.
    group_counters = {}
    # Track invalid entries
    invalid_counter = 0
    # Dictionary to map subjectID to filename prefix (or "no result" if skipped)
    subject_file_mapping = {}

    for subject_doc in subjects_docs:
        subject_id = subject_doc.id
        subject_info = subject_doc.to_dict()
        group_choice = subject_info.get("groupChoice", "default")

        # Determine experiment label: if default_experiment_label is not None, use it;
        # otherwise, attempt to use subject_info's "experiment" field.

        experiment_label = subject_info.get("experimentID", default_experiment_label)

        # Fetch trial data for this subject using subject_id
        trials_docs = trials_collection.where('id', '==', subject_id).stream()
        trial_docs_list = list(trials_docs)

        records = []
        for trial_doc in trial_docs_list:
            trial_data = trial_doc.to_dict()
            trial_list_length = len(trial_data.get('trialNum', []))
            for i in range(trial_list_length):
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
                        if 'movement_trajectory' in trial_data and trial_data.get('movement_trajectory') and i < len(trial_data['movement_trajectory'])
                        else 'N/A'
                }
                records.append(record)

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

        # Update participant number for the group.
        if group_choice not in group_counters:
            group_counters[group_choice] = 1
        participant_number = group_counters[group_choice]
        group_counters[group_choice] += 1

        # Create filename prefix using the naming format:
        # "S_{experiment_label}_Group{groupChoice}_{participant_number}"
        filename_prefix = f"S-{experiment_label}-Group{group_choice}-ID{participant_number}"

        # Save trial data to CSV in the original CSV folder
        df = pd.DataFrame(records)
        csv_file_path = os.path.join(csv_folder, f"{filename_prefix}.csv")
        df.to_csv(csv_file_path, index=False)
        print(f'Trial data for subject {subject_id} written to CSV at {csv_file_path}')

        # Save subject information to JSON in the target JSON folder
        json_file_path = os.path.join(json_folder, f"{filename_prefix}.json")
        with open(json_file_path, 'w') as f:
            json.dump(subject_info, f)
        print(f'Subject info for subject {subject_id} written to JSON at {json_file_path}')

        # Old: subject_filename_map[subject_id] = filename_prefix
        # Updated: use filename_prefix as key, map to dict with id and completion_status
        subject_file_mapping[filename_prefix] = create_subject_file_mapping(subject_info, required_trail_num, subject_id, len(records))

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
