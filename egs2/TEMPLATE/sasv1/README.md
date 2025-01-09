# Spoofing Aware Speaker Representations

This is the sasv1 recipe template for ESPnet2.
ESPnet-SASV extends [ESPnet-SPK](https://www.isca-archive.org/interspeech_2024/jung24c_interspeech.pdf) (Jung et al. 2024) to support the training and extraction of speaker embeddings that are robust to spoofing.

## Table of Contents

* [Recipe flow](#recipe-flow)
  * [1\. Data preparation](#1-data-preparation)
  * [2\. Speed perturbation](#2-speed-perturbation)
  * [3\. Wav format](#3-wav-format)
  * [4\. Statistics collection](#4-spk-statistics-collection)
  * [5\. Spk pre-training](#5-spk-pretraining)
  * [6\. SASV training](#6-sasv-training)
  * [7\. SASV embedding extraction](#7-sasv-embedding-extraction)
  * [8\. Score calculation](#8-score-calculation)
  * [9\. Metric calculation](#9-metric-calculation)
  * [10\-11\. (Optional) Pack results for upload](#10-11-optional-pack-results-for-upload)
* [How to run](#how-to-run)
  * [LibriSpeech training](#librispeech-training)
* [Related works](#related-works)

## Recipe flow

sasv1 recipe consists of 11 stages.

### 1. Data preparation

Data preparation stage.

#### ESPnet format:

It calls `local/data.sh` to create Kaldi-style data directories in `data/` for training, validation, and evaluation sets. It's the same as `asr1` tasks.

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)

### 2. Speed perturbation
Generate train data with different speed offline, as a form of augmentation.

### 3. Wav format

Format the wave files in `wav.scp` to a single format (wav / flac / kaldi_ark).

### 4. Spk statistics collection

Statistics calculation stage.
It collects the shape information of input and output texts for Spk training.
Currently, it's close to a dummy because we set all utterances to have equal
duration in the training phase.

### 5. Spk pretraining

Optional speaker model training stage.
You can change the training setting via `--spk_config` and `--spk_args` options.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 6. SASV training

Spk model training stage.
You can change the pre-training setting via `--spk_config` and `--spk_args` options.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)


### 7. SASV embedding extraction
Extracts spoofing-aware speaker embeddings for inference.
Speaker embeddings belonging to the evaluation set are extracted.
If `score_norm=true` and/or `qmf_func=true`, cohort set(s) for score normalization and/or quality measure function is also extracted.

### 8. Score calculation
Calculates speaker similarity scores for an evaluation protocol (i.e., a set of trials).
One scalar score is calcuated for each trial.

This stage includes score normalization if set with `--score_norm=true`.
This stage includes score normalization if set with `--qmf_func=true`.

### 9. Metric calculation
Calculates minimum agnostic Decision Cost Function ([a-DCF](https://arxiv.org/abs/2403.01355))

### 10-11. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to Huggingface.
If you want to run this stage, you need to register your account in Huggingface.

## How to run

### ASVspoof5 Training
Here, we show the procedure to run the recipe using `egs2/asvspoof5/sasv1`.

Move to the recipe directory.
```sh
$ cd egs2/asvspoof5/sasv1
```

Modify `ASVSPOOF5` variable in `db.sh` if you want to change the download directory.
```sh
$ vim db.sh
```

Modify `cmd.sh` and `conf/*.conf` if you want to use the job scheduler.
See the detail in [using job scheduling system](https://espnet.github.io/espnet/parallelization.html).
```sh
$ vim cmd.sh
```

Run `run.sh`, which conducts all of the stages explained above.
```sh
$ ./run.sh
```

## Related works
```
@inproceedings{jung24c_interspeech,
  title     = {ESPnet-SPK: full pipeline speaker embedding toolkit with reproducible recipes, self-supervised front-ends, and off-the-shelf models},
  author    = {Jee-weon Jung and Wangyou Zhang and Jiatong Shi and Zakaria Aldeneh and Takuya Higuchi and Alex Gichamba and Barry-John Theobald and Ahmed {Hussen Abdelaziz} and Shinji Watanabe},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4278--4282},
  doi       = {10.21437/Interspeech.2024-1345},
  issn      = {2958-1796},
}
```
