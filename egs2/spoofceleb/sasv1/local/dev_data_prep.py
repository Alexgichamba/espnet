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

args = parser.parse_args()

SpoofCeleb_root = Path(args.spoofceleb_root)
target_root = Path(args.target_dir)
split = args.split

assert split in ["dev", "eval"], "Split must be either dev or eval"

if not SpoofCeleb_root.exists():
    print("SpoofCeleb directory does not exist")
    sys.exit(1)

protocol_file = "sasv_evaluation_evaluation_protocol.csv" if split == "eval" else "sasv_development_evaluation_protocol.csv"
trial_file = SpoofCeleb_root / "protocol" / f"{protocol_file}"
if not trial_file.exists():
    print(f"SpoofCeleb trial file {trial_file} does not exist")
    sys.exit(1)

subdir_mapping = {
    "dev": "development",
    "eval": "evaluation",
}

metadata_file = SpoofCeleb_root / "metadata" / f"{subdir_mapping[split]}.csv"
if not metadata_file.exists():
    print(f"SpoofCeleb metadata file {metadata_file} does not exist")
    sys.exit(1)

trial_label_mapping = {
    "target": 0,
    "nontarget": 1,
    "spoof": 2,
}

# Single set for all utterances
all_utterances = set()
enroll_utterances = set()  # Keep track of enrollment utterances for spk2enroll

trial_label = target_root / "trial_label"
wav_scp = target_root / "wav.scp"
utt2spk = target_root / "utt2spk"
spk2enroll = target_root / "spk2enroll"

# build scp dict from metadata file, and write utt2spk
scp_dict = {}
with open(metadata_file, "r") as f, open(utt2spk, "w") as f_utt2spk, open(wav_scp, "w") as f_wav:
    lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(",")
        flac_file_name = parts[0]
        uttID = flac_file_name.replace("/", "_")
        speakerID = parts[1]
        spoofID = parts[2]
        subdir = f"flac/{subdir_mapping[split]}"
        path = SpoofCeleb_root / subdir / f"{flac_file_name}"
        scp_dict[uttID] = path
        f_utt2spk.write(f"{uttID} {spoofID}_{speakerID}\n")
        f_wav.write(f"{uttID} {path}\n")


with open(trial_file, "r") as f_trial, open(
    trial_label, "w") as f_trial_label:
    
    lines = f_trial.readlines()
    for line in lines:
        parts = line.strip().split(",")
        enroll_utt = parts[0].replace("/", "_")
        test_utt = parts[1].replace("/", "_")
        label = parts[2]

        # Write trial label: enroll_utt*test_utt label
        f_trial_label.write(f"{enroll_utt}*{test_utt} {trial_label_mapping[label]}\n")
        
        # Keep track of enrollment utterances for spk2enroll
        enroll_utterances.add(enroll_utt)


# Write spk2enroll
with open(spk2enroll, "w") as f_spk2enroll:
    for utt in enroll_utterances:
        f_spk2enroll.write(f"{utt} {utt}\n")

# Print statistics
print(f"Number of total utterances: {len(scp_dict)}")
print(f"Number of enrollment utterances: {len(enroll_utterances)}")