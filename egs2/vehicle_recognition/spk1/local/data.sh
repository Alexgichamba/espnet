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

# Define directories for IDMT_Traffic
AUDIO_DIR_duo="${data_dir_prefix}/IDMT_Traffic/audio"
AUDIO_DIR_mono="${data_dir_prefix}/IDMT_Traffic/audio_mono"
ANNOTATION_DIR="${data_dir_prefix}/IDMT_Traffic/annotation"
TRAIN_DIR="${data_dir_prefix}/IDMT_Traffic/train"
TEST_DIR="${data_dir_prefix}/IDMT_Traffic/test"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Make train and test dirs"

    # make all audio single channel
    # if audio_mono dir does not exist, create it.
    if [ ! -d "${AUDIO_DIR_mono}" ]; then
        log "Creating audio_mono dir"
        mkdir -p ${AUDIO_DIR_mono}
        # convert all audio files to mono
        python3 local/make_single_channel.py --input_dir ${AUDIO_DIR_duo}  --output_dir ${AUDIO_DIR_mono}
    else
        log "Audio_mono dir exists. Skip creating."
    fi

    # if train dir does not exist
    if [ ! -d "${TRAIN_DIR}" ]; then
        log "Creating train dir"
        mkdir -p ${TRAIN_DIR}
        # Create symbolic links for the audio files listed in the annotation files
        while IFS= read -r line; do
            ln -s "${AUDIO_DIR_mono}/${line}" "${TRAIN_DIR}/${line}"
        done < "${ANNOTATION_DIR}/eusipco_2021_train.txt"
    else
        log "Train dir exists. Skip creating."
    fi

    # if test dir does not exist
    if [ ! -d "${TEST_DIR}" ]; then
        log "Creating test dir"
        mkdir -p ${TEST_DIR}
        # Create symbolic links for the audio files listed in the annotation files
        while IFS= read -r line; do
            ln -s "${AUDIO_DIR_mono}/${line}" "${TEST_DIR}/${line}"
        done < "${ANNOTATION_DIR}/eusipco_2021_test.txt"
    else
        log "Test dir exists. Skip creating."
    fi

    # print the number of files in train and test dirs
    log "Number of files in train dir: $(ls -1 ${TRAIN_DIR} | wc -l)"
    log "Number of files in test dir: $(ls -1 ${TEST_DIR} | wc -l)"

    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Make kaldi style data directories."

    # if trg_dir does not exist, create it.
    mkdir -p ${trg_dir}

    # train
    python3 local/idmt_data_prep.py --audio_dir ${TRAIN_DIR} --trg_dir "${trg_dir}/train" --metadata_file ${ANNOTATION_DIR}/eusipco_2021_train.txt
    # sort files
    for f in utt2spk wav.scp; do
        sort ${trg_dir}/train/${f} -o ${trg_dir}/train/${f}
    done
    # make spk2utt from utt2spk
    utils/utt2spk_to_spk2utt.pl ${trg_dir}/train/utt2spk > ${trg_dir}/train/spk2utt
    # validate data dir
    utils/validate_data_dir.sh --no-feats --no-text ${trg_dir}/train

    # test
    python3 local/idmt_data_prep.py --audio_dir ${TEST_DIR} --trg_dir "${trg_dir}/test" --metadata_file ${ANNOTATION_DIR}/eusipco_2021_test.txt
    # sort files
    for f in utt2spk wav.scp; do
        sort ${trg_dir}/test/${f} -o ${trg_dir}/test/${f}
    done
    # make spk2utt from utt2spk
    utils/utt2spk_to_spk2utt.pl ${trg_dir}/test/utt2spk > ${trg_dir}/test/spk2utt
    # validate data dir
    utils/validate_data_dir.sh --no-feats --no-text ${trg_dir}/test

    # make trials
    if [ ! -f "${trg_dir}/test/trials.txt" ]; then
        python3 local/make_trials.py --utt2spk_file ${trg_dir}/test/utt2spk --output_file ${trg_dir}/test/trials.txt --num_trials 1000
    else
        log "Trials file exists. Skip creating."
    fi

    # make trial.scp,trial2.scp, and trial_label
    python3 local/convert_trial.py --trial_file ${trg_dir}/test/trials.txt --scp ${trg_dir}/test/wav.scp --out_dir ${trg_dir}/test

    log "Stage 2, DONE."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Download Musan and RIR_NOISES for augmentation."

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
    log "Stage 3, DONE."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"