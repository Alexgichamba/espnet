import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import argparse


def load_embeddings(embd_dir: str) -> dict:
    embd_dic = OrderedDict(np.load(embd_dir))
    embd_dic2 = {}
    for k, v in embd_dic.items():
        if len(v.shape) == 1:
            v = v[None, :]
        embd_dic2[k] = torch.nn.functional.normalize(
            torch.from_numpy(v), p=2, dim=1
        ).numpy()

    return embd_dic2


def main(args):
    embd_dir = args.embeddings
    trial_label = args.trial_label
    out_file = args.output

    embd_dic = load_embeddings(embd_dir)
    with open(trial_label, "r") as f:
        lines = f.readlines()
    trial_ids = [line.strip().split(" ")[0] for line in lines]
    labels = [int(line.strip().split(" ")[1]) for line in lines]

    enrolls = [trial.split("*")[0] for trial in trial_ids]
    tests = [trial.split("*")[1] for trial in trial_ids]
    assert len(enrolls) == len(tests) == len(labels)

    scores = []
    for e, t in zip(enrolls, tests):
        enroll = torch.from_numpy(embd_dic[e])
        test = torch.from_numpy(embd_dic[t])
        if len(enroll.size()) == 1:
            enroll = enroll.unsqueeze(0)
            test = enroll.unsqueeze(0)
        score = F.cosine_similarity(enroll, test, dim=1)
        scores.append(score.item())

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    with open(out_file, "w") as f:
        for enrol, test, score, label in zip(enrolls, tests, scores, labels):
            f.write(f"{enrol} {test} {score} {label}\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute scores for SASV trials")
    
    parser.add_argument("--embeddings", 
                         help="Path to the embeddings .npz file")
    
    parser.add_argument("--trial_label", 
                        help="Path to the trial label file")
    
    parser.add_argument("--output", 
                        help="Path to the output scores file")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
