# Team 33 IDLS24 project

## Baseline

Architecture: [SKA-TDNN](https://arxiv.org/abs/2204.01005)  

Hyperparams and training configs at [conf/train_ska_mel.yaml](conf/train_ska_mel.yaml)

Results on Vox1-O, after training on VoxCeleb1-dev

| EER (%) | minDCF|
|---------|-------|
|2.665| 0.191 |

Model and training logs available on [HuggingFace](https://huggingface.co/alexgichamba/idls24_team33_baseline)

Note, this is a reproducible end-to-end recipe. After installing ESPnet, data download, preprocessing, training and inference can be done by simply running the run.sh script