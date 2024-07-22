# data_prep.py

import argparse
import os
import sys

def main(args):
    src = args.input_dir
    dst = args.trg_dir

    spk2utt = {}
    utt2spk = []
    wav_list = []

    # Read the source directory
    # input dir has the following structure:
    # each directory is a speaker
    # each speaker directory has multiple wav files
    
    for spk in os.listdir(src):
        spk_path = os.path.join(src, spk)
        if not os.path.isdir(spk_path):
            continue

        for wav in os.listdir(spk_path):
            if not wav.endswith(".wav"):
                continue

            utt = wav.split(".")[0]
            spk2utt[spk] = spk2utt.get(spk, []) + [utt]
            utt2spk.append((utt, spk))
            wav_list.append((utt, os.path.join(spk_path, wav)))

    with open(os.path.join(dst, "utt2spk"), "w") as f_utt2spk, open(
        os.path.join(dst, "wav.scp"), "w") as f_wav:

        for utt in utt2spk:
            f_utt2spk.write(f"{utt[0]} {utt[1]}\n")

        for utt in wav_list:
            f_wav.write(f"{utt[0]} {utt[1]}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation for vehicle recognition")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="source directory of voxcelebs",
    )
    parser.add_argument(
        "--trg_dir",
        type=str,
        required=True,
        help="destination directory of voxcelebs",
    )
    args = parser.parse_args()

    sys.exit(main(args))