
import csv
from espnet2.utils.a_dcf import calculate_a_dcf
from dataclasses import dataclass
import sys
import os
from typing import List
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import fsolve
import numpy as np

@dataclass
class SASVCostModel:
    "Class describing SASV-DCF's relevant costs"
    Pspf: float = 0.05
    Pnontrg: float = 0.0095
    Ptrg: float = 0.9405
    Cmiss: float = 1
    Cfa_asv: float = 10
    Cfa_cm: float = 10

# File paths
input_file = 'exp/spk_train_ska_1_100_raw/inference/dev_raw_trial_scores'
metadata_file = '../asvspoof5_data/ASVspoof5.dev.trial.txt'
output_file = 'output_with_labels.txt'


# calculate a_dcf
adcf_results = calculate_a_dcf(
    output_file, cost_model=CostModel()
)

# check which trials were false accepted
def find_false_accepts(adcf_results):
    # Get the threshold from a_dcf results
    min_a_dcf_thresh = adcf_results["min_a_dcf_thresh"]

    # Load the data from the output file
    data = np.genfromtxt(output_file, dtype=str, delimiter=" ")
    scores = data[:, 2].astype(np.float64)
    keys = data[:, 3].astype(np.int32)

    false_accepts = []

    # Iterate through the scores and keys to find false accepts
    for i, (score, key) in enumerate(zip(scores, keys)):
        # False accept happens when non-target or spoof trial (key 1 or 2) is accepted (score >= threshold)
        if (key == 1 or key == 2) and score >= min_a_dcf_thresh:
            identifier = data[i, 0] + '*' + data[i, 1]
            false_accepts.append((identifier, score, key))

    return false_accepts

# Call the function to find false accepts
false_accepts = find_false_accepts(adcf_results)

# Print the false accept trials
print(f"Number of false accepts: {len(false_accepts)}")


# Load MOS scores from the file
def load_mos_scores(mos_file):
    mos_scores = {}
    with open(mos_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            identifier = parts[0]
            score = float(parts[1])
            mos_scores[identifier] = score
    return mos_scores

# Fit a Gaussian Mixture Model to the MOS scores
def fit_gmm(mos_values, n_components=2):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(mos_values.reshape(-1, 1))
    return gmm

# Function to compute the threshold between the two distributions
def find_threshold(gmm):
    # Extract parameters for the two Gaussian components
    mean1, mean2 = gmm.means_[0][0], gmm.means_[1][0]
    variance1, variance2 = gmm.covariances_[0][0][0], gmm.covariances_[1][0][0]
    weight1, weight2 = gmm.weights_[0], gmm.weights_[1]

    # Define the two Gaussian functions
    def gaussian(x, mean, variance, weight):
        return weight * (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean)**2 / (2 * variance))

    # Function representing the difference between the two Gaussians
    def equation(x):
        return gaussian(x, mean1, variance1, weight1) - gaussian(x, mean2, variance2, weight2)

    # Use fsolve to find the root (intersection point) of the equation
    threshold = fsolve(equation, x0=(mean1 + mean2) / 2)[0]
    return threshold

# Plot the MOS score distribution and the GMM components with threshold
def plot_gmm_mos_distribution_with_threshold(mos_values, gmm, threshold, save_path):
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram of MOS scores
    plt.hist(mos_values, bins=30, alpha=0.7, color='blue', label='MOS Scores', density=True)

    # Create a range of values to plot the GMM components
    x = np.linspace(min(mos_values), max(mos_values), 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    
    # Plot the overall GMM
    plt.plot(x, pdf, '-r', label='Gaussian Mixture Model')

    # Plot each Gaussian component
    for i in range(gmm.n_components):
        mean = gmm.means_[i][0]
        variance = np.sqrt(gmm.covariances_[i][0][0])
        weight = gmm.weights_[i]
        plt.plot(x, weight * (1/(variance * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * variance**2)),
                 label=f'Component {i+1} (mean={mean:.2f}, var={variance:.2f})')

    # Plot the threshold
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f}')

    plt.title('MOS Distribution with GMM Components and Threshold')
    plt.xlabel('MOS')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it

# Path to the MOS file and the output plot file
mos_file = "pseudomos/utt2pmos"
output_plot_file = "mos_score_distribution_gmm_with_threshold.png"

# Load the MOS scores
mos_scores = load_mos_scores(mos_file)
mos_values = np.array(list(mos_scores.values()))

# Fit a GMM to the MOS scores
gmm = fit_gmm(mos_values, n_components=2)

# Compute the threshold between the two distributions
threshold = find_threshold(gmm)

# make a list of uttids where the mos score is below the threshold
def get_low_mos_uttids(mos_scores, threshold):
    low_mos_uttids = []
    for uttid, score in mos_scores.items():
        if score < threshold:
            low_mos_uttids.append(uttid)
    return low_mos_uttids

# Get the list of utterances with low MOS scores
low_mos_uttids = get_low_mos_uttids(mos_scores, threshold)
print(f"Number of utterances with MOS score below the threshold: {len(low_mos_uttids)}")

# Plot the MOS score distribution with GMM components and the threshold
plot_gmm_mos_distribution_with_threshold(mos_values, gmm, threshold, output_plot_file)

print(f"Threshold between the two distributions: {threshold:.2f}")
print(f"Plot saved to {output_plot_file}")

# Step 2: Compute the mean scores for trg, nontrg, and spf categories
def compute_means(ids, keys, scores):
    trg_scores = scores[keys == 0]
    nontrg_scores = scores[keys == 1]
    spf_scores = scores[keys == 2]
    
    trg_mean = trg_scores.mean()
    nontrg_mean = nontrg_scores.mean()
    spf_mean = spf_scores.mean()
    
    return trg_mean, nontrg_mean, spf_mean

# Step 3: Rescore the accepted trials based on the low MOS list
def rescore_accepted_trials(uttids, identifiers, scores, keys, low_mos_uttids, trg_mean, spf_mean, threshold):
    rescored_data = []
    adjustment = trg_mean - spf_mean

    # keep track of the number of trials rescored
    num_rescored = 0
    for i, (uttid, identifier, score, key) in enumerate(zip(uttids, identifiers, scores, keys)):
        # Ensure we compare uttids after stripping any whitespace or formatting inconsistencies
        if score >= threshold and uttid.strip() in low_mos_uttids:
            # Adjust the score by subtracting the difference (trg_mean - spf_mean)
            new_score = score - adjustment
            rescored_data.append((identifier, new_score, key))
            num_rescored += 1
        else:
            rescored_data.append((identifier, score, key))

    print(f"Number of trials rescored: {num_rescored}")
    
    return rescored_data

# Step 4: Save the rescored file
def save_rescored_file(rescored_data, output_file):
    with open(output_file, 'w') as file:
        for identifier, score, key in rescored_data:
            spk, utt = identifier.split('*')
            file.write(f"{spk} {utt} {score:.6f} {key}\n")

# Step 5: Recompute the a_dcf using the rescored data
def recompute_a_dcf(rescored_file, cost_model):
    # Reuse your existing a_dcf function from previous steps
    return calculate_a_dcf(rescored_file, cost_model=cost_model)

# Load data
mos_scores = load_mos_scores(mos_file)
data = np.genfromtxt(output_file, dtype=str, delimiter=" ")
identifiers = np.char.add(data[:, 0], np.char.add('*', data[:, 1]))
scores = data[:, 2].astype(np.float64)
keys = data[:, 3].astype(np.int32)

# Compute trg_mean, nontrg_mean, spf_mean
trg_mean, nontrg_mean, spf_mean = compute_means(identifiers, keys, scores)
print(f"Target mean: {trg_mean:.2f}, Non-target mean: {nontrg_mean:.2f}, Spoof mean: {spf_mean:.2f}")

# Rescore accepted trials
uttids = data[:, 1]
rescored_data = rescore_accepted_trials(uttids, identifiers, scores, keys, low_mos_uttids, trg_mean, spf_mean, threshold)

output_rescored_file = 'output_rescored.txt'
# Save rescored file
save_rescored_file(rescored_data, output_rescored_file)
print(f"Rescored file saved to {output_rescored_file}")

# Recompute a_dcf
adcf_results = calculate_a_dcf(output_rescored_file, cost_model=CostModel())
print(f"New a-DCF after rescoring: {adcf_results['min_a_dcf']:.5f}, with threshold: {adcf_results['min_a_dcf_thresh']:.5f}")