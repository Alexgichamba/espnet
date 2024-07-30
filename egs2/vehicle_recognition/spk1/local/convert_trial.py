import argparse
import os
import sys


def main(args):
    with open(args.trial_file, "r") as f:
        lines_trial_org = f.readlines()
    with open(args.scp, "r") as f:
        lines_scp = f.readlines()

    scp_dict = dict()
    for scp in lines_scp:
        utt_id, path = scp.strip().split(" ")
        scp_dict[utt_id] = path

    with open(os.path.join(args.out_dir, "trial.scp"), "w") as f_trial, open(
        os.path.join(args.out_dir, "trial2.scp"), "w"
    ) as f_trial2, open(os.path.join(args.out_dir, "trial_label"), "w") as f_label:
        for tr in lines_trial_org:
            trial_key, label = tr.strip().split(" ")
            utt1, utt2 = trial_key.split("*")
            # remove the parent directory from the path of utterances
            utt1 = utt1.split("/")[-1]
            utt2 = utt2.split("/")[-1]
            # remove the .wav extension
            utt1 = utt1.replace(".wav", "")
            utt2 = utt2.replace(".wav", "")
            trial_key = "*".join([utt1, utt2])
            # trial.scp will have the key + the first utterance path
            f_trial.write(f"{trial_key} {scp_dict[utt1]}\n")
            # trial2.scp will have the key + the second utterance path
            f_trial2.write(f"{trial_key} {scp_dict[utt2]}\n")
            # trial_label will have the key + the label
            f_label.write(f"{trial_key} {label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trial mapper")
    parser.add_argument(
        "--trial_file",
        type=str,
        required=True,
        help="path of the original trial file",
    )
    parser.add_argument(
        "--scp",
        type=str,
        required=True,
        help="path of wav.scp file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="destination directory of processed trial and label files",
    )
    args = parser.parse_args()

    sys.exit(main(args))