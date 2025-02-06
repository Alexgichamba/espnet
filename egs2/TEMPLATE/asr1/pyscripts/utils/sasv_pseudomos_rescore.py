import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import fsolve
from espnet2.utils.a_dcf import calculate_a_dcf

@dataclass
class SASVCostModel:
    "Class describing SASV-DCF's relevant costs"
    Pspf: float = 0.05
    Pnontrg: float = 0.0095
    Ptrg: float = 0.9405
    Cmiss: float = 1
    Cfa_asv: float = 10
    Cfa_cm: float = 10

class MOSAnalyzer:
    def __init__(self, valid_mos_file: str, test_mos_file: str):
        self.valid_mos_scores = self._load_mos_scores(valid_mos_file)
        self.valid_mos_values = np.array(list(self.valid_mos_scores.values()))
        self.gmm = self._fit_gmm(self.valid_mos_values)
        self.threshold = self._find_threshold()

        self.test_mos_scores = self._load_mos_scores(test_mos_file)
        self.test_mos_values = np.array(list(self.test_mos_scores.values()))
        self.low_mos_uttids = self._get_low_mos_uttids()

    def _load_mos_scores(self, mos_file: str) -> Dict[str, float]:
        """Load MOS scores from file"""
        mos_scores = {}
        with open(mos_file, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                identifier = parts[0]
                score = float(parts[1])
                mos_scores[identifier] = score
        return mos_scores

    def _fit_gmm(self, mos_values: np.ndarray, n_components: int = 2) -> GaussianMixture:
        """Fit a Gaussian Mixture Model to the MOS scores"""
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(mos_values.reshape(-1, 1))
        return gmm

    def _find_threshold(self) -> float:
        """Compute threshold between GMM components"""
        mean1, mean2 = self.gmm.means_[0][0], self.gmm.means_[1][0]
        variance1, variance2 = self.gmm.covariances_[0][0][0], self.gmm.covariances_[1][0][0]
        weight1, weight2 = self.gmm.weights_[0], self.gmm.weights_[1]

        def gaussian(x, mean, variance, weight):
            return weight * (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean)**2 / (2 * variance))

        def equation(x):
            return gaussian(x, mean1, variance1, weight1) - gaussian(x, mean2, variance2, weight2)

        return fsolve(equation, x0=(mean1 + mean2) / 2)[0]

    def _get_low_mos_uttids(self) -> List[str]:
        """Get list of utterance IDs with MOS scores below threshold"""
        return [uttid for uttid, score in self.test_mos_scores.items() if score < self.threshold]

    def plot_distribution(self, output_plot_file: str):
        """Plot MOS score distribution with GMM components"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.valid_mos_values, bins=30, alpha=0.7, color='blue', label='MOS Scores', density=True)

        x = np.linspace(min(self.valid_mos_values), max(self.valid_mos_values), 1000)
        logprob = self.gmm.score_samples(x.reshape(-1, 1))
        pdf = np.exp(logprob)
        plt.plot(x, pdf, '-r', label='Gaussian Mixture Model')

        for i in range(self.gmm.n_components):
            mean = self.gmm.means_[i][0]
            variance = np.sqrt(self.gmm.covariances_[i][0][0])
            weight = self.gmm.weights_[i]
            plt.plot(x, weight * (1/(variance * np.sqrt(2 * np.pi))) * 
                    np.exp(-(x - mean)**2 / (2 * variance**2)),
                    label=f'Component {i+1} (mean={mean:.2f}, var={variance:.2f})')

        plt.axvline(self.threshold, color='green', linestyle='--', 
                   label=f'Threshold = {self.threshold:.2f}')
        plt.title('MOS Distribution with GMM Components and Threshold')
        plt.xlabel('MOS')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_plot_file)
        plt.close()

class ScoreProcessor:
    def __init__(self, test_file: str, valid_file: str):
        self.test_data = np.genfromtxt(test_file, dtype=str, delimiter=" ")
        self.valid_data = np.genfromtxt(valid_file, dtype=str, delimiter=" ")
        # file is of format: spk utt score key
        self.test_identifiers = np.char.add(self.test_data[:, 0], np.char.add('*', self.test_data[:, 1]))
        self.test_uttids = self.test_data[:, 1]

        self.test_scores = self.test_data[:, 2].astype(np.float64)
        self.valid_scores = self.valid_data[:, 2].astype(np.float64)

        self.test_keys = self.test_data[:, 3].astype(np.int32)
        self.valid_keys = self.valid_data[:, 3].astype(np.int32)
        
        self.means = self._compute_means()

    def _compute_means(self) -> Tuple[float, float, float]:
        """Compute mean scores for target, non-target, and spoof categories"""
        trg_scores = self.valid_scores[self.valid_keys    == 0]
        nontrg_scores = self.valid_scores[self.valid_keys == 1]
        spf_scores = self.valid_scores[self.valid_keys    == 2]
        return (trg_scores.mean(), nontrg_scores.mean(), spf_scores.mean())

    def rescore_trials(self, low_mos_uttids: List[str], threshold: float) -> List[Tuple]:
        """Rescore trials based on MOS analysis"""
        trg_mean, _, spf_mean = self.means
        adjustment = trg_mean - spf_mean
        rescored_data = []
        num_rescored = 0

        for i, (uttid, identifier, score, key) in enumerate(zip(self.test_uttids, self.test_identifiers, self.test_scores, self.test_keys)):
            if score >= threshold and uttid.strip() in low_mos_uttids:
                new_score = score - adjustment
                rescored_data.append((identifier, new_score, key))
                num_rescored += 1
            else:
                rescored_data.append((identifier, score, key))

        print(f"Number of trials rescored: {num_rescored}")
        return rescored_data

def save_scores(rescored_data: List[Tuple], output_file: str):
    """Save rescored data to file"""
    with open(output_file, 'w') as file:
        for identifier, score, key in rescored_data:
            spk, utt = identifier.split('*')
            file.write(f"{spk} {utt} {score:.6f} {key}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='SASV Score Processing with MOS Analysis')
    parser.add_argument('--valid_scorefile', required=True, 
                      help='Input file containing trial scores for the validation set')
    parser.add_argument('--test_scorefile', required=True,
                        help='Input file containing trial scores for the test set')
    parser.add_argument('--valid_pseudomos', required=True,
                      help='File containing MOS scores for the validation set')
    parser.add_argument('--test_pseudomos', required=True,
                        help='File containing MOS scores for the test set')
    parser.add_argument('--output_dir', required=True,
                      help='Path to save the rescored trial scores for the test set')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize MOS analyzer
    mos_analyzer = MOSAnalyzer(args.valid_pseudomos, args.test_pseudomos)
    print(f"Threshold between distributions for validation: {mos_analyzer.threshold:.2f}")
    print(f"Number of utterances with MOS score below threshold in the test set: {len(mos_analyzer.low_mos_uttids)}")
    
    # Plot MOS distribution
    plot_path = os.path.join(args.output_dir, 'mos_distribution.png')
    mos_analyzer.plot_distribution(plot_path)
    print(f"Plot saved to {plot_path}")

    # Compute initial a-DCF
    adcf_results = calculate_a_dcf(args.valid_scorefile, cost_model=SASVCostModel())
    print(f"Valid a-DCF: {adcf_results['min_a_dcf']:.5f}, "
          f"with valid threshold: {adcf_results['min_a_dcf_thresh']:.5f}")
    accept_threshold = adcf_results['min_a_dcf_thresh']
    
    # Process scores
    processor = ScoreProcessor(args.test_scorefile, args.valid_scorefile)
    
    # Rescore trials and save results
    rescored_data = processor.rescore_trials(mos_analyzer.low_mos_uttids, 
                                           threshold=accept_threshold)
    output_file = os.path.join(args.output_dir, 'mos_rescored_trials.txt')
    save_scores(rescored_data, output_file)
    print(f"Rescored file saved to {output_file}")
    
    # Calculate final a-DCF
    adcf_results = calculate_a_dcf(output_file, cost_model=SASVCostModel())
    print(f"New a-DCF after rescoring: {adcf_results['min_a_dcf']:.5f}, "
          f"with threshold: {adcf_results['min_a_dcf_thresh']:.5f}")

if __name__ == '__main__':
    main()