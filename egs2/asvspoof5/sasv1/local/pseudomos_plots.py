# plot_score_distribution.py
# This score_file takes in a score_file containing SASV scores
# and plots the distribution of the scores.
# The output is a histogram of the scores. Each class is
# represented by a different color.

import sys
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt

def plot_distribution(mos_score_file, metadata_file, utt_field_id, class_field_id):
    # Build class dictionary
    class_dict = {}
    with open(metadata_file, "r") as f:
        for line in f:
            line = line.strip().split()
            class_dict[line[utt_field_id]] = line[class_field_id]

    # Build score dictionary
    score_dict = {}
    with open(mos_score_file, "r") as f:
        for line in f:
            line = line.strip().split()
            score_dict[line[0]] = float(line[1])

    # find the set of classes
    classes = set(class_dict.values())
    clases = sorted(classes)

    # Assign unique hatch patterns and colors
    hatch_patterns = ['/', '\\', '|', '-', '+', 'x', 'o']
    colors = plt.cm.tab10.colors  # A palette with 10 distinct colors
    hatch_dict = {c: hatch_patterns[i % len(hatch_patterns)] for i, c in enumerate(classes)}
    color_dict = {c: colors[i % len(colors)] for i, c in enumerate(classes)}

    # Plot the classwise-score distribution with different colors and textures for each class
    mean_dict = {}
    std_dict = {}
    for c in classes:
        #plot only A14 and A12
        # if c == "A14":
        #     continue
        scores = [score_dict[utt] for utt in class_dict if class_dict[utt] == c]
        plt.hist(
            scores,
            bins=100,
            alpha=0.7,
            label=c,
            color=color_dict[c],
            hatch=hatch_dict[c],
            edgecolor='black'
        )

        # compute classwise mean and std
        mean = sum(scores) / len(scores)
        std = (sum([(s - mean) ** 2 for s in scores]) / len(scores)) ** 0.5
        mean_dict[c] = mean
        std_dict[c] = std
    
    # print mean and std sorted by highest mean
    for c in sorted(mean_dict, key=mean_dict.get, reverse=True):
        print(f"{c}: mean={mean_dict[c]}, std={std_dict[c]}")

    # save plot
    plt.xlabel("Pseudo-MOS")
    plt.ylabel("Number of utterances")
    # sort legend by alphabetical order
    plt.legend(title="Class", loc="upper left")
    plt.savefig("pmos_distribution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file", help="Path to the file with pseudo-mos scores")
    parser.add_argument("--metadata_file", help="Path to the metadata file")
    parser.add_argument("--utt_field_id", type=int, help="Field id for the utterance id")
    parser.add_argument("--class_field_id", type=int, help="Field id for the class")
    args = parser.parse_args()
    plot_distribution(args.score_file, args.metadata_file, args.utt_field_id, args.class_field_id)