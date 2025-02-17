# local/train_data_prep.py
# Makes kaldi-style files for SpoofCeleb training data

import argparse
import os
import sys
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Prepare Kaldi-style files for SpoofCeleb training data from CSV'
    )
    parser.add_argument(
        'spoofceleb_dir',
        type=str,
        help='Directory containing SpoofCeleb dataset'
    )
    parser.add_argument(
        'target_dir',
        type=str,
        help='Output directory for Kaldi-style files'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    SpoofCeleb_root = Path(args.spoofceleb_dir)
    target_root = Path(args.target_dir)

    # Check if the SpoofCeleb directory exists
    if not SpoofCeleb_root.exists():
        print("SpoofCeleb directory does not exist")
        sys.exit(1)

    # Metadata file path
    csv_file = SpoofCeleb_root / "metadata" /"train.csv"
    if not csv_file.exists():
        print(f"SpoofCeleb CSV file {csv_file} does not exist")
        sys.exit(1)

    # Create target directory if it doesn't exist
    target_root.mkdir(parents=True, exist_ok=True)

    # Open all the files
    with open(csv_file, "r") as f_meta, \
         open(target_root / "wav.scp", "w") as f_wav, \
         open(target_root / "utt2spk", "w") as f_utt2spk, \
         open(target_root / "utt2spf", "w") as f_utt2spf:
        
        lines = f_meta.readlines()
        # loop through the lines in the metadata file, skipping the header
        for line in lines[1:]:
            # Split CSV line and unpack values
            parts = line.strip().split(",")
            if len(parts) < 3:
                print(f"Invalid CSV line: {line}")
                
            speaker_id = parts[1]          # SPEAKER_ID
            flac_name = parts[0]           # FLAC_FILE_NAME
            spoof_key = parts[2]           # KEY (spoof/bonafide)
            
            # Construct file paths and IDs
            path = SpoofCeleb_root / "flac" / "train" / f"{flac_name}"
            spoofID = "bonafide" if spoof_key == "a00" else "spoof"
            speakerID = f"{spoofID}_{speaker_id}"
            # convert / in flac_name to _
            flac_name = flac_name.replace("/", "_")
            uttID = f"{speakerID}_{flac_name}"

            # Write wav.scp file
            f_wav.write(f"{uttID} {path}\n")
            # Write the utt2spk file
            f_utt2spk.write(f"{uttID} {speakerID}\n")
            # Write the utt2spf file
            f_utt2spf.write(f"{uttID} {spoofID}\n")

if __name__ == "__main__":
    main()