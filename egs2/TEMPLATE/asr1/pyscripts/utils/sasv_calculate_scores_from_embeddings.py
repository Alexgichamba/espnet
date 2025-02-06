import os
import sys
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import argparse


def load_embeddings(embd_dir: str, device: str):
    print(f"Loading embeddings from {embd_dir}")
    embd_dic = OrderedDict(np.load(embd_dir))
    embd_dic2 = {}
    for k, v in tqdm(embd_dic.items(), desc="Reading and normalizing embeddings..."):
        if len(v.shape) == 1:
            v = v[None, :]
        embd_dic2[k] = torch.nn.functional.normalize(
            torch.from_numpy(v), p=2, dim=1
        ).to(device)

    return embd_dic2


def main(args):
    embd_dir = args.embeddings
    trial_label = args.trial_label
    out_file = args.output
    device = args.device

    # Load embeddings
    embd_dic = load_embeddings(embd_dir, device)

    # Read trial labels
    with open(trial_label, "r") as f:
        lines = f.readlines()
    
    # Prepare trial information
    trial_pairs = []
    for line in lines:
        parts = line.strip().split()
        trial_id, label = parts[0], int(parts[1])
        enroll, test = trial_id.split("*")
        trial_pairs.append((enroll, test, label))

    # Sort trials for efficient processing
    sorted_trials = sorted(trial_pairs, key=lambda x: (x[0], x[1]))

    # Process trials
    scores = []
    labels = []
    current_enroll = None
    current_enroll_embd = None

    for enroll, test, label in tqdm(sorted_trials, desc="Computing scores..."):
        # Compute enrollment embedding only when it changes
        if enroll != current_enroll:
            current_enroll = enroll
            current_enroll_embd = embd_dic[enroll]
            current_enroll_embd = current_enroll_embd.unsqueeze(0).mean(dim=1)

        # Get test embedding
        test_embd = embd_dic[test]
        test_embd = test_embd.unsqueeze(0).mean(dim=1)

        # Compute cosine similarity
        score = F.cosine_similarity(current_enroll_embd, test_embd, dim=1).item()
        
        scores.append(score)
        labels.append(label)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Write results
    with open(out_file, "w") as f:
        for (enroll, test, score, label) in zip(
            [x[0] for x in sorted_trials], 
            [x[1] for x in sorted_trials], 
            scores, 
            labels
        ):
            f.write(f"{enroll} {test} {score} {label}\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute scores for SASV trials")
    
    parser.add_argument("--embeddings", 
                         help="Path to the embeddings .npz file")
    
    parser.add_argument("--trial_label", 
                        help="Path to the trial label file")
    
    parser.add_argument("--output", 
                        help="Path to the output scores file")
    
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (default: %(default)s)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
