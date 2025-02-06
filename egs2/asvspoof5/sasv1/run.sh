#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Remember to update the following variables based on the stage
mode=sasv
audio_format=flac
skip_spk_pretrain=true
pretrained_model=10epoch.pth
ignore_init_mismatch=true

spk_config=conf/pretrain_SKATDNN_mel.yaml
sasv_config=conf/bs64_train_SKATDNN_mel.yaml
inference_config=conf/decode.yaml

inference_model=../../9epoch.pth
use_pseudomos=true

spk_train_set=voxceleb2_dev
spk_valid_set=voxceleb1_test

sasv_train_set=asvspoof5_train
sasv_valid_set=asvspoof5_dev

test_sets=asvspoof5_dev

eval_valid_set=true
skip_train=false

feats_type="raw" # or raw_copy

ngpu=1
nj=4
speed_perturb_factors=

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
    --pretrained_model ${pretrained_model} \
    --ignore_init_mismatch ${ignore_init_mismatch} \
    --inference_config ${inference_config} \
    --inference_model ${inference_model} \
    --use_pseudomos ${use_pseudomos} \
    "$@"
