#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000

data_dir_prefix= # root dir to save datasets
trg_dir=data


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z ${data_dir_prefix} ]; then
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/idlproj"
    data_dir_prefix=${MAIN_ROOT}/egs2/idlproj
else
    log "Root dir set to ${ASVSpoof_LA}"
    data_dir_prefix=${ASVSpoof_LA}
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation for pretraining on VoxCeleb1"

    if [ ! -f "${data_dir_prefix}/vox1_dev_wav.zip" ]; then
        local/download_voxcelebs.sh
    else
        log "vox1_dev_wav.zip exists. Skip download."
    fi

    if [ ! -d "${data_dir_prefix}/voxceleb1" ]; then
        mkdir -p "${data_dir_prefix}/voxceleb1/dev"
        mkdir -p "${data_dir_prefix}/voxceleb1/test"
        unzip "${data_dir_prefix}/vox1_dev_wav.zip" -d "${data_dir_prefix}/voxceleb1/dev"
        unzip "${data_dir_prefix}/vox1_test_wav.zip" -d "${data_dir_prefix}/voxceleb1/test"
    else
        log "voxceleb1 exists. Skip unzip."
    fi

    # download Vox1-O eval protocol
    if [ ! -f "${data_dir_prefix}/veri_test2.txt" ]; then
        log "Download Vox1-O cleaned eval protocol."
        wget -P ${data_dir_prefix} https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
    else
       log "Skip downloading Vox1-O cleaned eval protocol."
    fi

    if [ ! -d "${trg_dir}/test_vox1" ]; then
        log "Making Kaldi style files and making trials"

        mkdir -p "${trg_dir}/test_vox1"
        mkdir -p "${trg_dir}/dev_vox1"
        
        python local/voxcelebs_data_prep.py --src "${data_dir_prefix}/voxceleb1/test/wav" --dst "${trg_dir}/test_vox1"
        python local/voxcelebs_data_prep.py --src "${data_dir_prefix}/voxceleb1/dev/wav" --dst "${trg_dir}/dev_vox1"

        for f in wav.scp utt2spk spk2utt; do
            sort ${trg_dir}/test_vox1/${f} -o ${trg_dir}/test_vox1/${f}
            sort ${trg_dir}/dev_vox1/${f} -o ${trg_dir}/dev_vox1/${f}
        done

    
    else
        log "vox1 kaldi style files exists. Skipping."

    fi

    # make test trial compatible with ESPnet.
    python local/convert_trial.py --trial ${data_dir_prefix}/veri_test2.txt --scp ${trg_dir}/test_vox1/wav.scp --out ${trg_dir}/test_vox1


    log "Stage 1, DONE."

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Download Musan and RIR_NOISES for augmentation."

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
    log "Making scp files for musan"
    for x in music noise speech; do
        find ${data_dir_prefix}/musan/${x} -iname "*.wav" > ${trg_dir}/musan_${x}.scp
    done

    # Use small and medium rooms, leaving out largerooms.
    # Similar setup to Kaldi and VoxCeleb_trainer.
    log "Making scp files for RIRS_NOISES"
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/mediumroom -iname "*.wav" > ${trg_dir}/rirs.scp
    find ${data_dir_prefix}/RIRS_NOISES/simulated_rirs/smallroom -iname "*.wav" >> ${trg_dir}/rirs.scp
    log "Stage 2, DONE."
fi

log "Successfully finished. [elapsed=${SECONDS}s]"