import random
import argparse

def generate_trials(utt2spk_file: str, output_file: str, num_trials: int = 1000):
    # Read the utt2spk file
    utt2spk = {}
    with open(utt2spk_file, 'r') as file:
        for line in file:
            utt, spk = line.strip().split()
            utt2spk[utt] = spk

    # Separate utterances by speaker
    speakers = {}
    for utt, spk in utt2spk.items():
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(utt)
    
    # Prepare for trials
    genuine_trials = []
    imposter_trials = []

    # Generate genuine trials
    for utts in speakers.values():
        if len(utts) > 1:
            for i in range(len(utts)):
                for j in range(i + 1, len(utts)):
                    genuine_trials.append((utts[i], utts[j], 1))
                    if len(genuine_trials) >= num_trials // 2:
                        break
                if len(genuine_trials) >= num_trials // 2:
                    break
        if len(genuine_trials) >= num_trials // 2:
            break

    # Generate imposter trials
    all_utterances = list(utt2spk.keys())
    while len(imposter_trials) < num_trials // 2:
        utt1 = random.choice(all_utterances)
        spk1 = utt2spk[utt1]
        spk2 = random.choice([spk for spk in speakers.keys() if spk != spk1])
        utt2 = random.choice(speakers[spk2])
        imposter_trials.append((utt1, utt2, 0))

    # Combine and shuffle trials
    trials = genuine_trials + imposter_trials
    random.shuffle(trials)
    
    # Write trials to output file
    with open(output_file, 'w') as file:
        for utt1, utt2, label in trials:
            file.write(f"{utt1}*{utt2} {label}\n")
    
    print(f"Generated {num_trials} trials with {num_trials // 2} genuine and {num_trials // 2} imposter trials. Saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate trials from utt2spk file.")
    parser.add_argument('--utt2spk_file', type=str, help='Path to the utt2spk file')
    parser.add_argument('--output_file', type=str, help='Path to the output file for trials')
    parser.add_argument('--num_trials', type=int, default=1000, help='Number of trials to generate')

    args = parser.parse_args()
    
    generate_trials(args.utt2spk_file, args.output_file, args.num_trials)
