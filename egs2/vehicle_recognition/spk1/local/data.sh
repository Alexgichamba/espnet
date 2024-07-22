#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=100
n_proc=8

data_dir_prefix= # root dir to save datasets.

trg_dir=data

. utils/parse_options.sh
. db.sh
. path.sh
. cmd.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



if [ -z ${data_dir_prefix} ]; then
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/vehicle_recognition"
    data_dir_prefix=${MAIN_ROOT}/egs2/vehicle_recognition
else
    log "Root dir set to ${VOXCELEB}"
    data_dir_prefix=${VOXCELEB}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Make kaldi style files"

    mkdir -p ${trg_dir}/train
    mkdir -p ${trg_dir}/test

    # convert dual channel to mono
    python3 local/make_single_channel.py \
        --input_dir ${data_dir_prefix}/classes \
        --output_dir ${data_dir_prefix}/classes_mono

    python3 local/data_prep.py \
        --input_dir ${data_dir_prefix}/classes_mono/train \
        --trg_dir ${trg_dir}/train

    for f in wav.scp utt2spk ; do
        sort ${trg_dir}/train/${f} -o ${trg_dir}/train/${f}
    done
    utils/utt2spk_to_spk2utt.pl ${trg_dir}/train/utt2spk > "${trg_dir}/train/spk2utt"

    python3 local/data_prep.py \
        --input_dir ${data_dir_prefix}/classes_mono/test \
        --trg_dir ${trg_dir}/test

    for f in wav.scp utt2spk ; do
        sort ${trg_dir}/test/${f} -o ${trg_dir}/test/${f}
    done
    utils/utt2spk_to_spk2utt.pl ${trg_dir}/test/utt2spk > "${trg_dir}/test/spk2utt"

    # make trials
    python3 local/make_trials.py \
        --base_directory ${data_dir_prefix}/classes_mono/test \
        --output_file ${data_dir_prefix}/trials.txt

    # convert trials
    python3 local/convert_trial.py \
        --trial_file ${data_dir_prefix}/trials.txt \
        --scp ${trg_dir}/test/wav.scp \
        --out_dir ${trg_dir}/test

    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Download Musan and RIR_NOISES for augmentation."

    # if trg_dir does not exist, create it.
    mkdir -p ${trg_dir}

    if [ ! -f ${data_dir_prefix}/rirs_noises.zip ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/28/rirs_noises.zip
    else
        log "RIRS_NOISES exists. Skip download."
    fi

    if [ ! -f ${data_dir_prefix}/musan.tar.gz ]; then
        wget -P ${data_dir_prefix} -c http://www.openslr.org/resources/17/musan.tar.gz
    else
        log "Musan exists. Skip download."
    fi

    if [ -d ${data_dir_prefix}/RIRS_NOISES ]; then
        log "Skip extracting RIRS_NOISES"
    else
        log "Extracting RIR augmentation data."
        unzip -q ${data_dir_prefix}/rirs_noises.zip -d ${data_dir_prefix}
    fi

    if [ -d ${data_dir_prefix}/musan ]; then
        log "Skip extracting Musan"
    else
        log "Extracting Musan noise augmentation data."
        tar -zxvf ${data_dir_prefix}/musan.tar.gz -C ${data_dir_prefix}
    fi

    # make scp files
    for x in music noise speech; do
        find ${data_dir_prefix}/musan/${x} -iname "*.wav" > ${trg_dir}/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    # Similar setup to Kaldi and VoxCeleb_trainer.
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" > ${trg_dir}/rirs.scp
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >> ${trg_dir}/rirs.scp
    log "Stage 2, DONE."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"