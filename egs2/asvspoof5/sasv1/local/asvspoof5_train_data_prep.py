# local/asvspoof5_train_data_prep.py
# Makes kaldi-style files for ASVspoof5 training data using TSV format

import argparse
import os
import sys
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Prepare Kaldi-style files for ASVspoof5 training data from TSV'
    )
    parser.add_argument(
        'asvspoof_dir',
        type=str,
        help='Directory containing ASVspoof5 dataset'
    )
    parser.add_argument(
        'target_dir',
        type=str,
        help='Output directory for Kaldi-style files'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    ASVSpoof_root = Path(args.asvspoof_dir)
    target_root = Path(args.target_dir)

    # Check if the ASVspoof5 directory exists
    if not ASVSpoof_root.exists():
        print("ASVspoof5 directory does not exist")
        sys.exit(1)

    # TSV file path
    tsv_file = ASVSpoof_root / "ASVspoof5.train.tsv"
    if not tsv_file.exists():
        print(f"ASVspoof5 TSV file {tsv_file} does not exist")
        sys.exit(1)

    # Create target directory if it doesn't exist
    target_root.mkdir(parents=True, exist_ok=True)

    # Open all the files
    with open(tsv_file, "r") as f_meta, \
         open(target_root / "wav.scp", "w") as f_wav, \
         open(target_root / "utt2spk", "w") as f_utt2spk, \
         open(target_root / "utt2spf", "w") as f_utt2spf:
        
        lines = f_meta.readlines()
        for line in lines:
            # Split TSV line and unpack values
            parts = line.strip().split()
            if len(parts) < 10:
                print(f"Invalid TSV line: {line}")
                
            speaker_id = parts[0]          # SPEAKER_ID
            flac_name = parts[1]           # FLAC_FILE_NAME
            spoof_key = parts[8]           # KEY (spoof/bonafide)
            
            # Construct file paths and IDs
            path = ASVSpoof_root / "flac_T" / f"{flac_name}.flac"
            uttID = f"{spoof_key}_{speaker_id}_{flac_name}"
            speakerID = f"{spoof_key}_{speaker_id}"

            # Write wav.scp file
            f_wav.write(f"{uttID} {path}\n")
            # Write the utt2spk file
            f_utt2spk.write(f"{uttID} {speakerID}\n")
            # Write the utt2spf file
            f_utt2spf.write(f"{uttID} {spoof_key}\n")

if __name__ == "__main__":
    main()