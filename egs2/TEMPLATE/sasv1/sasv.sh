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

min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}

SECONDS=0

# General configuration
stage=1                 # Processes starts from the specified stage.
stop_stage=10000        # Processes is stopped at the specified stage.
skip_stages=            # Spicify the stage to be skipped
skip_data_prep=false    # Skip data preparation stages.
skip_spk_pretrain=false # Skip speaker model pretraining stage.
skip_train=false        # Skip sasv training stage.
skip_eval=false         # Skip decoding and evaluation stages.
skip_packing=true       # Skip the packing stage.
skip_upload_hf=true     # Skip uploading to huggingface stage.

eval_valid_set=false    # Run decoding for the validation set
ngpu=1                  # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1             # The number of nodes.
nj=8                    # The number of parallel jobs.
dumpdir=dump            # Directory to dump features.
expdir=exp              # Directory to save experiments.
python=python3          # Specify python to execute espnet commands.
fold_length=120000      # fold_length for speech data during enhancement training.

# Data preparation related
local_data_opts=        # The options given to local/data.sh

# Speed perturbation related
speed_perturb_factors="0.9 1.0 1.1" # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw                      # Feature type (raw, raw_copy, fbank_pitch, or extracted).
audio_format=wav                    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
multi_columns_input_wav_scp=false   # Enable multi columns mode for input wav.scp for format_wav_scp.py
multi_columns_output_wav_scp=false  # Enable multi columns mode for output wav.scp for format_wav_scp.py
fs=16k                              # Sampling rate.
min_wav_duration=1.0                # Minimum duration in second.
max_wav_duration=60.                # Maximum duration in second.

# Speaker model related
mode=sasv                       # Training mode: spk (pre-training) or sasv
spk_exp=                        # Specify the directory path for spk experiment.
sasv_exp=                       # Specify the directory path for sasv experiment.
spk_tag=                        # Suffix to the result dir for spk model training.
sasv_tag=                       # Suffix to the result dir for sasv model training.
spk_config=                     # Config for the spk model training.
sasv_config=                    # Config for the sasv model training.
spk_args=                       # Arguments for spk model training.
spf_args=                       # Arguments for sasv model training.
pretrained_model=               # Pretrained model to load for sasv training.
ignore_init_mismatch=false      # Ignore weights corresponding to mismatched keys in the pretrained model.

# Inference related
inference_config=conf/decode.yaml       # Inference configuration
inference_model=                        # Inference model weight file
score_norm=false                        # Apply score normalization in inference.
qmf_func=false                          # Apply quality measurement based calibration in inference.
use_pseudomos=false                     # Use pseudomos for post-scoring

# [Task dependent] Set the datadir name created by local/data.sh
spk_train_set=      # Name of speaker pretraining set.
sasv_train_set=     # Name of sasv training set.
spk_valid_set=      # Name of validation set used for monitoring/tuning network pretraining.
sasv_valid_set=     # Name of validation set used for monitoring/tuning network training.
cohort_set=         # Name of cohort set used for score normalization and qmf function.
test_sets=          # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
lang=multilingual   # The language type of corpus.


# Upload model related
hf_repo=

help_message=$(cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    stage=1               # Processes starts from the specified stage.
    stop_stage=10000      # Processes is stopped at the specified stage.
    skip_stages=          # Spicify the stage to be skipped
    skip_data_prep=false  # Skip data preparation stages.
    skip_train=false      # Skip training stages.
    skip_eval=false       # Skip decoding and evaluation stages.
    skip_packing=true     # Skip the packing stage.
    skip_upload_hf=true   # Skip uploading to huggingface stage.

    eval_valid_set=false  # Run decoding for the validation set
    ngpu=1                # The number of gpus ("0" uses cpu, otherwise use gpu).
    num_nodes=1           # The number of nodes.
    nj=32                 # The number of parallel jobs.
    gpu_inference=false   # Whether to perform gpu decoding.
    dumpdir=dump          # Directory to dump features.
    expdir=exp            # Directory to save experiments.
    python=python3        # Specify python to execute espnet commands.
    fold_length=80000     # fold_length for speech data during enhancement training

    # Speed perturbation related
    speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

    # Feature extraction related
    feats_type=raw       # Feature type (raw, raw_copy, fbank_pitch, or extracted).
    audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
    multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
    multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
    fs=16k               # Sampling rate.
    min_wav_duration=1.0  # Minimum duration in second.
    max_wav_duration=60.  # Maximum duration in second.

    # Speaker model related
    spk_exp=              # Specify the directory path for spk experiment.
    spk_tag=              # Suffix to the result dir for spk model training.
    spk_config=           # Config for the spk model training.
    spk_args=             # Arguments for spk model training.
    pretrained_model=     # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch= # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").

    # Inference related
    inference_config=     # Inference configuration file
    inference_model=      # Inference model weight file
    score_norm=false      # Apply score normalization in inference.
    qmf_func=false        # Apply quality measurement based calibration in inference.

    # [Task dependent] Set the datadir name created by local/data.sh
    train_set=        # Name of training set.
    valid_set=        # Name of validation set used for monitoring/tuning network training.
    cohort_set=       # Name of cohort set used for score normalization and qmf function.
    test_sets=        # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
    lang=multilingual # The language type of corpus.

    # Upload model related
    hf_repo=          # The huggingface repository directory

EOF
)

log "$0 $*"
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh



if [ $# -ne 0  ]; then
    log "${help_message}"
    echo "Positional arguments passed: $@"
    log "Error: No positional arguments are required."
        exit 2
fi

. ./path.sh
. ./cmd.sh

# Check feature type
if [ "${feats_type}" = raw  ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = raw_copy  ]; then
    # raw_copy is as same as raw except for skipping the format_wav stage
    data_feats=${dumpdir}/raw_copy
elif [ "${feats_type}" = fbank  ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" = extracted  ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Extra files for speaker recognition process
utt_extra_files="utt2category"

# Set tag for naming of spk model directory
if [ -z "${spk_tag}" ]; then
    if [ -n "${spk_config}" ]; then
        spk_tag="$(basename "${spk_config}" .yaml)_${feats_type}"
    else
        spk_tag="train_${feats_type}"
    fi
fi
# Set directory used for training commands
spk_stats_dir="${expdir}/spk_stats_${fs}"
if [ -z "${spk_exp}"  ]; then
    spk_exp="${expdir}/spk_${spk_tag}"
fi

# Set tag for naming of sasv model directory
if [ -z "${sasv_tag}" ]; then
    if [ -n "${sasv_config}" ]; then
        sasv_tag="$(basename "${sasv_config}" .yaml)_${feats_type}"
    else
        sasv_tag="train_${feats_type}"
    fi
fi
# Set directory used for training commands
sasv_stats_dir="${expdir}/sasv_stats_${fs}"
if [ -z "${sasv_exp}"  ]; then
    sasv_exp="${expdir}/sasv_${sasv_tag}"
fi

# Set dataset tuples
train_sets=("${spk_train_set}" "${sasv_train_set}")
valid_sets=("${spk_valid_set}" "${sasv_valid_set}")
stats_dirs=("${spk_stats_dir}" "${sasv_stats_dir}")
configs=("${spk_config}" "${sasv_config}")
exps=("${spk_exp}" "${sasv_exp}")


# Determine which stages to skip
if "${skip_data_prep}"; then
    skip_stages+="1 2 "
fi

if "${skip_spk_pretrain}"; then
    skip_stages+="5 "
fi

if "${skip_packing}"; then
    skip_stages+="10 "
fi
if "${skip_upload_hf}"; then
    skip_stages+="11 "
fi

skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"


if [ ${stage} -le 1  ] && [ ${stop_stage} -ge 1  ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]]  ]]; then
    log "Stage 1: Data preparation for train and evaluation."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
    log "Stage 1 FIN."
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then

    if [ -n "${speed_perturb_factors}" ]; then

        if [ "${mode}" = spk ]; then
            _train_set="${spk_train_set}"
        else
            _train_set="${sasv_train_set}"
        fi
        log "Stage 2: Speed perturbation: data/${_train_set} -> data/${_train_set}_sp"
        
        _scp_list="wav.scp "
        _dirs=""

        for factor in ${speed_perturb_factors}; do
            if ${python} -c "assert ${factor} != 1.0" 2>/dev/null; then
                scripts/utils/perturb_enh_data_dir_speed.sh --utt_extra_files "${utt_extra_files}" "${factor}" "data/${_train_set}" "data/${_train_set}_sp${factor}" "${_scp_list}"
                _dirs+="data/${_train_set}_sp${factor} "
            else
                # If speed factor is 1, same as the original
                _dirs+="data/${_train_set} "
            fi
        done
        utils/combine_data.sh --extra-files "${_scp_list}" "data/${_train_set}_sp" ${_dirs}
    else
        log "Skip stage 2: Speed perturbation"
    fi

    # Update the names if speed perturbation was applied
    if [ -n "${speed_perturb_factors}" ]; then
        if [ "${mode}" = spk ]; then
            spk_train_set="${spk_train_set}_sp" 
            spk_stats_dir="${spk_stats_dir}_sp"
            spk_exp="${spk_exp}_sp"
        else
            sasv_train_set="${sasv_train_set}_sp"
            sasv_stats_dir="${sasv_stats_dir}_sp" 
            sasv_exp="${sasv_exp}_sp"
        fi
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

    if [ "${mode}" = spk ]; then
        _train_set="${spk_train_set}"
        _valid_set="${spk_valid_set}"
    else
        _train_set="${sasv_train_set}"
        _valid_set="${sasv_valid_set}"
    fi

    if "${eval_valid_set}" = true; then
        _dsets="${_valid_set} ${test_sets}"
    else
        _dsets="${test_sets}"
    fi

    if [ "${feats_type}" = raw ]; then
        if [ "${skip_train}" = false ]; then
            log "Formatting training set: ${_train_set}"
            # Format Training Utterances
            utils/copy_data_dir.sh --validate_opts --non-print data/"${_train_set}" "${data_feats}/${_train_set}"

            # copy extra files that are not covered by copy_data_dir.sh
            # category2utt and spf2utt will be used by the data sampler
            cp data/"${_train_set}/spk2utt" "${data_feats}/${_train_set}/category2utt"
            # if mode is sasv, copy spf2utt for use by the sasv sampler
            if [ "${mode}" = sasv ]; then
                cp data/"${_train_set}/spf2utt" "${data_feats}/${_train_set}/spf2utt"
                cp data/"${_train_set}/utt2spf" "${data_feats}/${_train_set}/utt2spf"
            fi
            for x in music noise speech; do
                cp data/musan_${x}.scp ${data_feats}/musan_${x}.scp
            done
            cp data/rirs.scp ${data_feats}/rirs.scp

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${_train_set}/wav.scp" "${data_feats}/${_train_set}"

            echo "${feats_type}" > "${data_feats}/${_train_set}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_feats}/${_train_set}/audio_format"
            else
                echo "${audio_format}" > "${data_feats}/${_train_set}/audio_format"
            fi

        fi

        # Format Validation and Test Utterances
        for dset in ${_dsets}; do
            log "Formatting validation/eval set: ${dset}"
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"

            # copy extra files that are not covered by copy_data_dir.sh
            cp data/${dset}/trial_label "${data_feats}/${dset}"
            cp data/${dset}/spk2enroll "${data_feats}/${dset}"

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${dset}/wav.scp" "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
            echo "multi_${audio_format}" > "${data_feats}/${dset}/audio_format"
            for f in ${utt_extra_files}; do
                [ -f data/${dset}/${f} ] && cp data/${dset}/${f} ${data_feats}/${dset}/${f}
            done
        done

    elif [ "${feats_type}" = raw_copy ]; then
        if [ "${skip_train}" = false ]; then 
            log "Formatting training set: ${_train_set}"
            utils/copy_data_dir.sh --validate_opts --non-print data/"${_train_set}" "${data_feats}/${_train_set}"
            # category2utt will be used bydata sampler
            cp data/"${_train_set}/spk2utt" "${data_feats}/${_train_set}/category2utt"
            if [ "${mode}" = sasv ]; then
                cp data/"${_train_set}/spf2utt" "${data_feats}/${_train_set}/spf2utt"
            fi
            for x in music noise speech; do
                cp data/musan_${x}.scp ${data_feats}/musan_${x}.scp
            done
            cp data/rirs.scp ${data_feats}/rirs.scp

            echo "${feats_type}" > "${data_feats}/${_train_set}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_feats}/${_train_set}/audio_format"
            else
                echo "${audio_format}" > "${data_feats}/${_train_set}/audio_format"
            fi
        fi

        # Format Validation and Test Utterances
        for dset in ${_dsets}; do
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"
            cp data/${dset}/trial_label "${data_feats}/${dset}"
            cp data/${dset}/spk2enroll "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
            echo "multi_${audio_format}" > "${data_feats}/${dset}/audio_format"
            for f in ${utt_extra_files}; do
                [ -f data/${dset}/${f} ] && cp data/${dset}/${f} ${data_feats}/${dset}/${f}
            done
        done
    else
        log "${feats_type} is not supported yet."
        exit 1
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Collect stats"

    if [[ "${audio_format}" == *ark* ]]; then
        _type=kaldi_ark
    else
        # sound supports "wav", "flac", etc.
        _type=sound
    fi

    if [ "${mode}" = spk ]; then
        _train_set="${spk_train_set}"
        _valid_set="${spk_valid_set}"
        _stats_dir="${spk_stats_dir}"
        _exp="${spk_exp}"
        _config="${spk_config}"
    else
        _train_set="${sasv_train_set}"
        _valid_set="${sasv_valid_set}"
        _stats_dir="${sasv_stats_dir}"
        _exp="${sasv_exp}"
        _config="${sasv_config}"
    fi
    _test_sets="${test_sets}"

    _train_dir="${data_feats}/${_train_set}"
    _valid_dir="${data_feats}/${_valid_set}"
    _test_dir="${data_feats}/${_test_sets}"

    if [ -n "${_config}"  ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.spk_train --print_config --optim adam
        _opts+="--config ${_config} "
    fi

    # 1. Split key file
    _logdir="${_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    _nj=$(min "${nj}" "$(<${_train_dir}/wav.scp wc -l)" "$(<${_valid_dir}/wav.scp wc -l)")

    key_file="${_train_dir}/wav.scp"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_valid_dir}/wav.scp"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${_stats_dir}/run.sh'. You can resume the process from stage 3 using this script"
    mkdir -p "${_stats_dir}"; echo "${run_args} -- stage3 \"\$@\"; exit \$?" > "${_stats_dir}/run.sh"; chmod +x "${_stats_dir}/run.sh"

    # 3. Submit jobs
    log "Speaker collect-stats started... log: '${_logdir}/stats.*.log'"

    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.spk_train \
            --collect_stats true \
            --use_preprocessor false \
            --train_data_path_and_name_and_type ${_train_dir}/wav.scp,speech,${_type} \
            --valid_data_path_and_name_and_type ${_valid_dir}/wav.scp,speech,${_type} \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --spk2utt ${_train_dir}/spk2utt \
            --spk_num $(wc -l ${_train_dir}/spk2utt | cut -f1 -d" ") \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${spk_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1;  }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --skip_sum_stats --output_dir "${_stats_dir}"

    # Compute stats for the test set
    mkdir "${_stats_dir}/test"
    log "Computing stats for the test set"
    ${python} pyscripts/utils/spk_calculate_test_shape.py   --wav_scp "${_test_dir}/wav.scp" \
                                                            --output_dir "${_stats_dir}/test" \
                                                            --fs "${fs}" \
                                                            --nj "${nj}" \

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Speaker Pretraining."

    _spk_train_dir="${data_feats}/${spk_train_set}"
    _spk_valid_dir="${data_feats}/${spk_valid_set}"
    _opts=
    if [ -n "${spk_config}"  ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.spk_train --print_config --optim adam
        _opts+="--config ${spk_config} "
    fi

    log "Spk training started... log: '${spk_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${spk_exp})"
    else
        jobname="${spk_exp}/train.log"
    fi

    # copy spk_enroll and trial_label to spk_exp under subdir trial_info
    mkdir -p "${spk_exp}/trial_info"
    cp "${_spk_valid_dir}/spk2enroll" "${spk_exp}/trial_info"
    cp "${_spk_valid_dir}/trial_label" "${spk_exp}/trial_info"

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log ${spk_exp}/train.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${spk_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.spk_train \
            --use_preprocessor true \
            --resume true \
            ${pretrained_model:+--init_param $pretrained_model} \
            --ignore_init_mismatch ${ignore_init_mismatch} \
            --output_dir ${spk_exp} \
            --train_data_path_and_name_and_type ${_spk_train_dir}/wav.scp,speech,sound \
            --train_data_path_and_name_and_type ${_spk_train_dir}/utt2spk,spk_labels,text \
            --train_shape_file ${spk_stats_dir}/train/speech_shape \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/wav.scp,speech,sound \
            --spk2utt ${_spk_train_dir}/spk2utt \
            --spk_num $(wc -l ${_spk_train_dir}/spk2utt | cut -f1 -d" ") \
            --fold_length ${fold_length} \
            --valid_shape_file ${spk_stats_dir}/valid/speech_shape \
            --output_dir "${spk_exp}" \
            ${_opts} ${spk_args}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: SASV Training."

    _sasv_train_dir="${data_feats}/${sasv_train_set}"
    _sasv_valid_dir="${data_feats}/${sasv_valid_set}"
    _opts=
    if [ -n "${sasv_config}"  ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.spk_train --print_config --optim adam
        _opts+="--config ${sasv_config} "
    fi

    log "SASV training started... log: '${sasv_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${sasv_exp})"
    else
        jobname="${sasv_exp}/train.log"
    fi

    # copy spk_enroll and trial_label to spk_exp under subdir trial_info
    mkdir -p "${sasv_exp}/trial_info"
    cp "${_sasv_valid_dir}/spk2enroll" "${sasv_exp}/trial_info"
    cp "${_sasv_valid_dir}/trial_label" "${sasv_exp}/trial_info"

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log ${sasv_exp}/train.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${sasv_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.sasv_train \
            --use_preprocessor true \
            --resume true \
            ${pretrained_model:+--init_param $pretrained_model} \
            --ignore_init_mismatch ${ignore_init_mismatch} \
            --output_dir ${sasv_exp} \
            --train_data_path_and_name_and_type ${_sasv_train_dir}/wav.scp,speech,sound \
            --train_data_path_and_name_and_type ${_sasv_train_dir}/utt2spk,spk_labels,text \
            --train_data_path_and_name_and_type ${_sasv_train_dir}/utt2spf,spf_labels,text \
            --train_shape_file ${sasv_stats_dir}/train/speech_shape \
            --valid_data_path_and_name_and_type ${_sasv_valid_dir}/wav.scp,speech,sound \
            --spk2utt ${_sasv_train_dir}/spk2utt \
            --spf2utt ${_sasv_train_dir}/spf2utt \
            --spk_num $(wc -l ${_sasv_train_dir}/spk2utt | cut -f1 -d" ") \
            --spf_num $(wc -l ${_sasv_train_dir}/spf2utt | cut -f1 -d" ") \
            --fold_length ${fold_length} \
            --valid_shape_file ${sasv_stats_dir}/valid/speech_shape \
            --output_dir "${sasv_exp}" \
            ${_opts}
            # ${sasv_args}
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Speaker embedding extraction."

    infer_exp="${sasv_exp}/inference"
    _inference_dir=${data_feats}/${test_sets}
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${infer_exp})"
    else
        jobname="${infer_exp}/sasv_embed_extraction.log"
    fi

    # copy spk_enroll and trial_label to spk_exp under subdir trial_info
    mkdir -p "${infer_exp}/trial_info"
    cp "${_inference_dir}/spk2enroll" "${infer_exp}/trial_info"
    cp "${_inference_dir}/trial_label" "${infer_exp}/trial_info"

    log "Extracting speaker embeddings for inference... log: '${infer_exp}/sasv_embed_extraction_test.log'"
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log ${infer_exp}/sasv_embed_extraction_test.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${sasv_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.sasv_embed_extract \
            --use_preprocessor true \
            --output_dir ${infer_exp} \
            --data_path_and_name_and_type ${_inference_dir}/wav.scp,speech,sound \
            --shape_file ${sasv_stats_dir}/test/speech_shape \
            --fold_length ${fold_length} \
            --config ${inference_config} \
            --spk_train_config "${sasv_exp}/config.yaml" \
            --spk_model_file "${sasv_exp}"/${inference_model} \
            # ${sasv_args}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Score calculation and post-processing."

    infer_exp="${sasv_exp}/inference"
    _inference_dir=${data_feats}/${test_sets}
    cohort_dir="${data_feats}/${cohort_set}"

    if [ ${ngpu} -gt 0 ]; then
        _device="cuda"
    else
        _device="cpu"
    fi

    log "Stage 8-a: get scores for the test set."
    ${python} pyscripts/utils/sasv_calculate_scores_from_embeddings.py  --embeddings ${infer_exp}/${test_sets}_embeddings.npz \
                                                                        --trial_label ${_inference_dir}/trial_label \
                                                                        --output ${infer_exp}/${test_sets}_raw_trial_scores \
                                                                        --device ${_device}
    scorefile_cur=${infer_exp}/${test_sets}_raw_trial_scores

    if "${use_pseudomos}"; then
        log "Stage 8-b: apply MOS rule for rescoring accepts."
        # first, compute pseudomos scores for the test set
        if [ ! -d "${infer_exp}/pseudomos" ]; then
            mkdir -p ${infer_exp}/pseudomos
            ${python} pyscripts/utils/evaluate_pseudomos.py ${_inference_dir}/wav.scp \
                                                            --outdir ${infer_exp}/pseudomos \
                                                            --batch_size 4

        else
            log "pseudomos dir already exists. Skip."
        fi

        # second, apply MOS rule for rescoring accepts
        pmos_file=${infer_exp}/pseudomos/utt2pmos
        scorefile_proc=${infer_exp}/${test_sets}_processed_scores

        ${python} pyscripts/utils/sasv_pseudomos_rescore.py --input_scorefile ${scorefile_cur} \
                                                            --utt2pmos ${pmos_file} \
                                                            --output_scorefile ${scorefile_proc}
    fi

fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    log "Stage 9: Calculate metrics."
    infer_exp="${sasv_exp}/inference"
    _inference_dir=${data_feats}/${test_sets}

    if "${use_pseudomos}"; then
        score_dir=${infer_exp}/${test_sets}_processed_scores
    else
        score_dir=${infer_exp}/${test_sets}_raw_trial_scores
    fi

    log "calculate score with ${score_dir}"
    ${python} pyscripts/utils/calculate_adcf.py --scorefile ${score_dir} \
                                                --out_dir ${infer_exp}/${test_sets}_metrics

    # Show results in Markdown syntax
    ${python} pyscripts/utils/show_sasv_result.py "${infer_exp}/${test_sets}_metrics" "${sasv_exp}"/RESULTS.md $(echo ${sasv_config} | cut -d'.' -f1)
    cat "${sasv_exp}"/RESULTS.md
fi

packed_model="${sasv_exp}/${sasv_exp##*/}_${inference_model%.*}.zip"
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    log "Stage 10: Pack model: ${packed_model}"

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack spk \
        --train_config "${sasv_exp}"/config.yaml \
        --model_file "${sasv_exp}"/"${inference_model}" \
        --option "${sasv_exp}"/RESULTS.md \
        --option "${sasv_exp}"/images \
        --outpath "${packed_model}"
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ] && ! [[ " ${skip_stages} " =~ [[:space:]]11[[:space:]] ]]; then
    log "Stage 11: Upload model to HuggingFace: ${hf_repo}"
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1

    if [ ! -f "${packed_model}" ]; then
        log "ERROR: ${packed_model} does not exist. Please run stage 9 first."
        exit 1
    fi

    gitlfs=$(git lfs --version 2> /dev/null || true)
    [ -z "${gitlfs}" ] && \
        log "ERROR: You need to install git-lfs first" && \
        exit 1

    dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
    [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="git checkout $(git show -s --format=%H)"
    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/spk1/ -> foo/spk1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/asr1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=speaker-recognition
    # shellcheck disable=SC2034
    espnet_task=SPK
    # shellcheck disable=SC2034
    task_exp=${sasv_exp}
    eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

    this_folder=${PWD}
    cd ${dir_repo}
    if [ -n "$(git status --porcelain)" ]; then
        git add .
        git commit -m "Update model"
    fi
    git push
    cd ${this_folder}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
