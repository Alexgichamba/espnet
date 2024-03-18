#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_ska_mel.yaml

train_set="dev_vox1"
valid_set="test_vox1"
cohort_set="test_vox1"
test_sets="test_vox1"
skip_train=false

feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --cohort_set ${cohort_set} \
    --test_sets ${test_sets} \
    --skip_train ${skip_train} \
    "$@"