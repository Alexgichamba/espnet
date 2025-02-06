import sys
from typing import List, Tuple
from dataclasses import dataclass
import argparse
import numpy as np

from espnet2.utils.a_dcf import calculate_a_dcf

@dataclass
class SASVCostModel:
    Pspf: float = 0.05
    Pnontrg: float = 0.05
    Ptrg: float = 0.9
    Cmiss: float = 1
    Cfa_asv: float = 10
    Cfa_cm: float = 20

def main(scorefile, out_dir):
    scores = []
    labels = []
    # open the score file and read the scores and labels.
    with open(scorefile, "r") as f:
        lines = f.readlines()
        for line in lines:
            enrol, test, score, label = line.strip().split()
            scores.append(float(score))
            labels.append(int(label))

    # calculate statistics in target and nontarget classes.
    n_trials = len(scores)
    scores_trg = []
    scores_nontrg = []
    scores_spf = []
    for _s, _l in zip(scores, labels):
        if _l == 0:
            scores_trg.append(_s)
        elif _l == 1:
            scores_nontrg.append(_s)
        elif _l == 2:
            scores_spf.append(_s)
        else:
            raise ValueError(f"{_l}, {type(_l)}")

    trg_mean = float(np.mean(scores_trg))
    trg_std = float(np.std(scores_trg))
    nontrg_mean = float(np.std(scores_nontrg))
    nontrg_std = float(np.std(scores_nontrg))
    spf_mean = float(np.mean(scores_spf))
    spf_std = float(np.std(scores_spf))

    results = calculate_a_dcf(
        sasv_score_txt=scorefile,
        cost_model=SASVCostModel(),
    )
    a_DCF = results["min_a_dcf"]
    threshold = results["min_a_dcf_thresh"]

    with open(out_dir, "w") as f:
        f.write(f"num trials: {n_trials}\n")
        f.write(f"trg_mean: {trg_mean}, trg_std: {trg_std}\n")
        f.write(f"nontrg_mean: {nontrg_mean}, nontrg_std: {nontrg_std}\n")
        f.write(f"spf_mean: {spf_mean}, spf_std: {spf_std}\n")
        f.write(f"min a-DCF: {a_DCF}, threshold: {threshold}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute scores for SASV trials")
    
    parser.add_argument("--scorefile", 
                         help="Path to the score file")
    
    parser.add_argument("--out_dir", 
                        help="Path to the output directory")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.scorefile, args.out_dir)
