#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Remember to update the following variables based on the stage
mode=spk
audio_format=flac
skip_spk_pretrain=false

spk_config=conf/spk_pretrain_SKATDNN_mel.yaml
sasv_config=conf/sasv_train_SKATDNN_mel.yaml

spk_train_set=voxceleb1_dev
spk_valid_set=voxceleb1_test

sasv_train_set=asvspoof5_train
sasv_valid_set=asvspoof5_dev
test_sets=asvspoof5_eval

skip_train=false

feats_type="raw" # or raw_copy

ngpu=1
nj=8
speed_perturb_factors="0.9 1.0 1.1"
inference_model=valid.a_dcf.best.pth

./sasv.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --cohort_set ${cohort_set} \
    --test_sets ${test_sets} \
    --skip_train ${skip_train} \
    --ngpu ${ngpu} \
    --nj ${nj} \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --audio_format ${audio_format} \
    --inference_model ${inference_model} \
    --mode ${mode} \
    "$@"