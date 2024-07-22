import os
import itertools
import argparse

# Function to get all files in the directories
def get_files_and_labels(base_directory, directories):
    files_and_labels = []
    for label, directory in directories.items():
        full_directory = os.path.join(base_directory, directory)
        files = os.listdir(full_directory)
        for file in files:
            if file.endswith(".wav"):
                relative_path = os.path.join(directory, file)
                files_and_labels.append((relative_path, label))
    return files_and_labels

def main(base_directory, output_file):
    # Directories for each class relative to the base directory
    directories = {
        "none": "none/",
        "car": "car/",
        "bus": "bus/",
        "truck": "truck/",
        "motorbike": "motorbike/",
        "person": "person/"
    }

    # Get all files and labels
    files_and_labels = get_files_and_labels(base_directory, directories)

    # Create pairs
    pairs = list(itertools.combinations(files_and_labels, 2))

    # Create accept/reject labels
    trials = []
    for (file1, label1), (file2, label2) in pairs:
        if label1 == label2:
            trials.append(f"1 {file1} {file2}")
        else:
            trials.append(f"0 {file1} {file2}")

    # Save to file
    with open(output_file, "w") as f:
        for trial in trials:
            f.write(trial + "\n")

    print(f"Binary trials file created at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary trials for EER computation.")
    parser.add_argument("--base_directory", type=str, required=True, help="Base directory for the input files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for the binary trials.")

    args = parser.parse_args()
    main(args.base_directory, args.output_file)
