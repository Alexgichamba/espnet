# local/dev_data_prep.py
# Prepares SpoofCeleb development or evaluation data

import argparse
import os
import sys
from pathlib import Path

# Create the parser
parser = argparse.ArgumentParser(
    description="Prepares SpoofCeleb development or eval data"
)

# Add the arguments
parser.add_argument(
    "--spoofceleb_root",
    type=str,
    required=True,
    help="The base directory of the SpoofCeleb data",
)
parser.add_argument(
    "--target_dir", type=str, required=True, help="The target directory"
)
parser.add_argument(
    "--split", type=str, required=True, help="The split (dev or eval)"
)

# Parse the arguments
args = parser.parse_args()

SpoofCeleb_root = Path(args.spoofceleb_root)
target_root = Path(args.target_dir)
split = args.split

assert split in ["dev", "eval"], "Split must be either dev or eval"

# check if the SpoofCeleb directories exist
if not SpoofCeleb_root.exists():
    print("SpoofCeleb directory does not exist")
    sys.exit(1)

# trial file
protocol_file = "sasv_evaluation_evaluation_protocol.csv" if split == "eval" else "sasv_development_evaluation_protocol.csv"
trial_file = SpoofCeleb_root / "protocol" / f"{protocol_file}"
if not trial_file.exists():
    print(f"SpoofCeleb trial file {trial_file} does not exist")
    sys.exit(1)

# subdir mapping: dev==development, eval==evaluation
subdir_mapping = {
    "dev": "development",
    "eval": "evaluation",
}

# trial label mapping: target==0, nontarget==1, spoof==2
trial_label_mapping = {
    "target": 0,
    "nontarget": 1,
    "spoof": 2,
}

# make a set of utterances
utterances = set()

# make trial_label file and wav.scp
trial_label = target_root / "trial_label"
wav_scp = target_root / "wav.scp"
utt2spk = target_root / "utt2spk"
spk2enroll = target_root / "spk2enroll"

with open(trial_file, "r") as f_trial, open(
    trial_label, "w") as f_trial_label, open(
        wav_scp, "w") as f_wav, open(
            utt2spk, "w") as f_utt2spk, open(
                spk2enroll, "w") as f_spk2enroll:
    lines = f_trial.readlines()
    # loop through the lines in the trial file, skipping the header
    for line in lines[1:]:
        # the trial file has the format:
            #  speakerID,flac_file_name,label
        parts = line.strip().split(",")
        speakerID = parts[0]
        flac_file_name = parts[1]
        label = parts[2]

        # the trial_label file has the format
        # speakerID*flac_file_name label
        f_trial_label.write(f"{speakerID}*{flac_file_name} {trial_label_mapping[label]}\n")

        # add the utterance to the set
        utterances.add(flac_file_name)
    # write the wav.scp file
    for utterance in utterances:
        subdir = f"flac/{subdir_mapping[split]}"
        path = SpoofCeleb_root / f"{subdir}" / f"{utterance}"
        # convert / in utterance to _
        utterance = utterance.replace("/", "_")
        f_wav.write(f"{utterance} {path}\n")
        f_utt2spk.write(f"{utterance} {utterance}\n")
        f_spk2enroll.write(f"{utterance} {utterance}\n")