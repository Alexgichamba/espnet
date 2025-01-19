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
    log "Root dir for dataset not defined, setting to ${MAIN_ROOT}/egs2/asvspoof5"
    data_dir_prefix=${MAIN_ROOT}/egs2/asvspoof5
else
    log "Root dir set to ${ASVSPOOF5}"
    data_dir_prefix=${ASVSPOOF5}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: ASVspoof5 Training Data Preparation"

    if [ ! -d "${trg_dir}/asvspoof5_train" ]; then
        log "Making Kaldi style files for train"
        mkdir -p "${trg_dir}/asvspoof5_train"
        python3 local/asvspoof5_train_data_prep.py "${data_dir_prefix}/asvspoof5_data" "${trg_dir}/asvspoof5_train"
        for f in wav.scp utt2spk utt2spf; do
            sort ${trg_dir}/asvspoof5_train/${f} -o ${trg_dir}/asvspoof5_train/${f}
        done
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/asvspoof5_train/utt2spk > "${trg_dir}/asvspoof5_train/spk2utt"
        utils/utt2spk_to_spk2utt.pl ${trg_dir}/asvspoof5_train/utt2spf > "${trg_dir}/asvspoof5_train/spf2utt"
        utils/validate_data_dir.sh --no-feats --no-text "${trg_dir}/asvspoof5_train" || exit 1
    else
        log "${trg_dir}/asvspoof5_train exists. Skip making Kaldi style files for train"
    fi

    log "Stage 1, DONE."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: ASVspoof 5 Development and Evaluation Data Preparation"

    for split in dev eval; do
        if [ ! -d "${trg_dir}/asvspoof5_${split}" ]; then
            log "Making Kaldi style files for ${split}"
            mkdir -p "${trg_dir}/asvspoof5_${split}"
            python3 local/asvspoof5_dev_data_prep.py --asvspoof5_root "${data_dir_prefix}/asvspoof5_data" --target_dir "${trg_dir}/asvspoof5_${split}" --split ${split}
            for f in wav.scp spk2enroll trial_label utt2spk; do
                sort "${trg_dir}/asvspoof5_${split}/${f}" -o "${trg_dir}/asvspoof5_${split}/${f}"
            done
            utils/utt2spk_to_spk2utt.pl "${trg_dir}/asvspoof5_${split}/utt2spk" > "${trg_dir}/asvspoof5_${split}/spk2utt"
            utils/validate_data_dir.sh --no-feats --no-text --no-spk-sort "${trg_dir}/asvspoof5_${split}" || exit 1
        else
            log "${trg_dir}/asvspoof5_${split} exists. Skip making Kaldi style files for asvspoof5_${split}"
        fi
    done

    log "Stage 2, DONE."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Download Musan and RIR_NOISES for augmentation."

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
    log "Stage 3, DONE."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: VoxCelebs Data Preparation"

    for dataset in voxceleb1 voxceleb2; do
        for split in dev test; do
            if [ ! -d "${trg_dir}/${dataset}_${split}" ]; then
                log "Making Kaldi style files for ${dataset}_${split}"
                mkdir -p "${trg_dir}/${dataset}_${split}"
                python3 local/voxceleb_data_prep.py --src "${data_dir_prefix}/${dataset}/${split}" --dst "${trg_dir}/${dataset}_${split}"
                for f in wav.scp utt2spk; do
                    sort "${trg_dir}/${dataset}_${split}/${f}" -o "${trg_dir}/${dataset}_${split}/${f}"
                done
                utils/utt2spk_to_spk2utt.pl "${trg_dir}/${dataset}_${split}/utt2spk" > "${trg_dir}/${dataset}_${split}/spk2utt"
                utils/validate_data_dir.sh --no-feats --no-text "${trg_dir}/${dataset}_${split}" || exit 1
            else
                log "${trg_dir}/${dataset}_${split} exists. Skip making Kaldi style files for ${dataset}_${split}"
            fi
        done
    done

    log "Stage 4, DONE."
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: VoxCeleb Evaluation Data Preparation"

    for dataset in voxceleb1; do
        if [ ! -f "${trg_dir}/voxceleb1_test/trial_label" ]; then
            log "Making trial files for voxceleb1_test"
            mkdir -p "${trg_dir}/voxceleb1_test"
            python3 local/voxceleb_trial_prep.py --scp "${trg_dir}/voxceleb1_test/wav.scp" --dst "${trg_dir}/voxceleb1_test" --trial "${data_dir_prefix}/veri_test2.txt"
            for f in wav.scp utt2spk spk2enroll trial_label; do
                sort "${trg_dir}/voxceleb1_test/${f}" -o "${trg_dir}/voxceleb1_test/${f}"
            done
            utils/utt2spk_to_spk2utt.pl "${trg_dir}/voxceleb1_test/utt2spk" > "${trg_dir}/voxceleb1_test/spk2utt"
            utils/validate_data_dir.sh --no-feats --no-text "${trg_dir}/voxceleb1_test" || exit 1
        else
            log "${trg_dir}/voxceleb1_test exists. Skip making trial files for voxceleb1_test"
        fi
    done

    log "Stage 5, DONE."
fi


log "Successfully finished. [elapsed=${SECONDS}s]"