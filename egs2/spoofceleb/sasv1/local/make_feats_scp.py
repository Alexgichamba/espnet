# make_feats_scp.py

# make_feats_scp.py

# This script reads the input directory and creates the feats.scp file which is of form:
# <utt_id> <absolute_path_to_npy_file>


import argparse
import os

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    return parser.parse_args()


def main():
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # generate feats.scp file
    with open(os.path.join(output_dir, "feats.scp"), "w") as f_feats:
        for root, _, files in os.walk(input_dir):
            for file in tqdm(files):
                if file.endswith(".npy"):
                    utt_id = file.split(".")[0]
                    f_feats.write(
                        f"{utt_id} {os.path.abspath(os.path.join(root, file))}\n"
                    )
    print("feats.scp file created successfully!")


if __name__ == "__main__":
    main()