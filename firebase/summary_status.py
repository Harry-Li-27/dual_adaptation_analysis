import json
import os
from collections import defaultdict, Counter

def load_data(filepath):
    """
    Load the JSON data from the given filepath.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def summarize_status(data):
    """
    Given a dict of {key: {..., "completion_status": status}}, return a dict
    mapping each prefix (everything before the last hyphen) to a Counter of statuses.
    """
    summary = defaultdict(Counter)
    for full_key, info in data.items():
        status = info.get("completion_status", "unknown")
        if '-' in full_key:
            prefix = full_key.rsplit('-', 1)[0]
        else:
            prefix = full_key
        summary[prefix][status] += 1

    # Convert defaultdict(Counter) to a normal dict of dicts
    return {prefix: dict(counts) for prefix, counts in summary.items()}

def write_summary_json(summary, output_path):
    """
    Write the summary dict to a JSON file at output_path, with indentation for readability.
    """
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)

def main():
    # Define input JSON file, output folder, and output filename
    input_json = "python_scripts/data/participant_information/subject_file_mapping.json"
    output_folder = "python_scripts/data/participant_information"
    output_filename = "status_summary.json"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load, summarize, and write
    data = load_data(input_json)
    summary = summarize_status(data)
    output_path = os.path.join(output_folder, output_filename)
    write_summary_json(summary, output_path)

    print(f"Summary written to: {output_path}")

if __name__ == "__main__":
    main()
