#!/usr/bin/env python3

from espnet2.tasks.sasv import SASVTask


def get_parser():
    parser = SASVTask.get_parser()
    return parser


def main(cmd=None):
    r"""Spoofing Aware Speaker embedding extractor training.

    Trained model can be used for spoofing-aware
    speaker verification, open set speaker identification, and also as
    embeddings for various other tasks including speaker diarization.

    Example:
        % python sasv_train.py --print_config --optim adadelta \
                > conf/train_spk.yaml
        % python sasv_train.py --config conf/train_diar.yaml
    """
    SASVTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
