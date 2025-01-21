#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Remember to update the following variables based on the stage
mode=sasv
audio_format=flac
skip_spk_pretrain=true

spk_config=conf/pretrain_SKATDNN_mel.yaml
sasv_config=conf/train_SKATDNN_mel.yaml

spk_train_set=voxceleb1_dev
spk_valid_set=voxceleb1_test

sasv_train_set=asvspoof5_train
sasv_valid_set=asvspoof5_dev
test_sets=asvspoof5_eval
eval_valid_set=true
skip_train=false

feats_type="raw" # or raw_copy

ngpu=1
nj=4
speed_perturb_factors=
inference_model=valid.a_dcf.best.pth

./sasv.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --spk_train_set ${spk_train_set} \
    --spk_valid_set ${spk_valid_set} \
    --sasv_config ${sasv_config} \
    --sasv_train_set ${sasv_train_set} \
    --sasv_valid_set ${sasv_valid_set} \
    --test_sets ${test_sets} \
    --eval_valid_set ${eval_valid_set} \
    --skip_train ${skip_train} \
    --ngpu ${ngpu} \
    --nj ${nj} \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --inference_model ${inference_model} \
    --mode ${mode} \
    --audio_format ${audio_format} \
    --skip_spk_pretrain ${skip_spk_pretrain} \
    "$@"