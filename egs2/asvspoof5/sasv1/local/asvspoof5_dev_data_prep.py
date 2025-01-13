# local/asvspoof5_dev_data_prep.py
# Prepares ASVspoof5 development or evaluation data

import argparse
import os
import sys
from pathlib import Path

# Create the parser
parser = argparse.ArgumentParser(
    description="Prepares ASVspoof5 development or eval data"
)

# Add the arguments
parser.add_argument(
    "--asvspoof5_root",
    type=str,
    required=True,
    help="The directory of the ASVspoof5 enrolment data",
)
parser.add_argument(
    "--target_dir", type=str, required=True, help="The target directory"
)
parser.add_argument(
    "--split", type=str, required=True, help="The split (dev or eval)"
)

# Parse the arguments
args = parser.parse_args()

ASVSpoof_root = Path(args.asvspoof5_root)
target_root = Path(args.target_dir)
split = args.split

assert split in ["dev", "eval"], "Split must be either dev or eval"

# check if the ASVspoof5 directories exist
if not ASVSpoof_root.exists():
    print("ASVspoof5 enrolment directory does not exist")
    sys.exit(1)

# trial file
trial_file = ASVSpoof_root / f"ASVspoof5.{split}.track_2.trial.tsv"
if not trial_file.exists():
    print("ASVspoof5 trial file does not exist")
    sys.exit(1)

# enrollment file
enrollment_file = ASVSpoof_root / f"ASVspoof5.{split}.track_2.enroll.tsv"
if not enrollment_file.exists():
    print("ASVspoof5 enrollment file does not exist")
    sys.exit(1)

# trial label mapping: target==0, nontarget==1, spoof==2
trial_label_mapping = {
    "target": 0,
    "nontarget": 1,
    "spoof": 2,
}

# subdir mapping: dev==D, eval==E_eval
subdir_mapping = {
    "dev": "D",
    "eval": "E_eval",
}

# make spk2enroll file
spk2enroll = target_root / "spk2enroll"
spkenrollment_dict = {}
with open(enrollment_file, "r") as f_enroll, open(
    spk2enroll, "w") as f_spk2enroll:
    lines = f_enroll.readlines()
    for line in lines:
        # the enrollment file has the format:
            #  speakerID flac_file_name1,flac_file_name2,flac_file_name3...
        parts = line.strip().split()
        speakerID = parts[0]
        flac_files = parts[1]
        # the spk2enroll file has the same format
        f_spk2enroll.write(f"{speakerID} {flac_files}\n")
        # add entry to the dictionary
        spkenrollment_dict[speakerID] = flac_files.split(",")

# make a set of utterances
utterances = set()
# add the values of the spkenrollment_dict to the utterances set
for flac_files in spkenrollment_dict.values():
    for flac_file in flac_files:
        utterances.add(flac_file)
len_enrollment = len(utterances)

# make trial_label file and wav.scp
trial_label = target_root / "trial_label"
wav_scp = target_root / "wav.scp"
utt2spk = target_root / "utt2spk"

with open(trial_file, "r") as f_trial, open(
    trial_label, "w") as f_trial_label, open(
        wav_scp, "w") as f_wav, open(
            utt2spk, "w") as f_utt2spk:
    lines = f_trial.readlines()

    for line in lines:
        # the trial file has the format:
            #  speakerID flac_file_name gender spoofing_class label
        parts = line.strip().split()
        speakerID = parts[0]
        flac_file_name = parts[1]
        label = parts[4]
        # the trial_label file has the format
        # speakerID*flac_file_name label
        f_trial_label.write(f"{speakerID}*{flac_file_name} {trial_label_mapping[label]}\n")

        # add the utterance to the set
        utterances.add(flac_file_name)
    print(f"In split {split}, there are {len(utterances)} unique utterances total")
    print(f"{len(utterances) - len_enrollment} are unique test utterances")
    print(f"{len_enrollment} are unique enrollment utterances")
    # write the wav.scp file
    for utterance in utterances:
        subdir = f"flac_{subdir_mapping[split]}"
        path = ASVSpoof_root / f"{subdir}" / f"{utterance}.flac"
        f_wav.write(f"{utterance} {path}\n")
        f_utt2spk.write(f"{utterance} {utterance}\n")
