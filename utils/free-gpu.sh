#!/usr/bin/env bash
# Author: Gaurav Kumar

if [ "$1" = "--help" ] ; then
    echo "Usage: $0 -n <Number of free GPUs on a machine>"
    echo "e.g.: $0 -n 2"
    echo "1, 2"
    exit 0
fi

# Allow requests for multiple GPUs
# (Optional) defaults to 1
req_gpus=1
while getopts ':n:' opt; do
  case ${opt} in
    n)
      req_gpus=${OPTARG}
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2
      exit 1
      ;;
    *)
      echo "Option -${OPTARG} is not supported" >&2
      exit 1
      ;;
  esac
done

# Number of free GPUs on a machine
n_gpus=$(lspci | grep -i "nvidia" | grep -c -v "Audio")

# Return -1 if there are no GPUs on the machine
# or if the requested number of GPUs exceed
# the number of GPUs installed.
if [ ${n_gpus} -eq 0 ] || [ ${req_gpus} -gt ${n_gpus} ]; then
  echo "-1"
  exit 1
fi

# shellcheck disable=SC2026
f_gpu=$(nvidia-smi | sed -e '1,/Processes/d' \
  | tail -n+3 | head -n-1 | awk '{print $2}' \
  | awk -v ng=${n_gpus} 'BEGIN{for (n=0;n<ng;++n){g[n] = 1}} {delete g[$1];} END{for (i in g) print i}' \
  | tail -n ${req_gpus})

# return -1 if not enough free GPUs were found
if [[ $(echo ${f_gpu} | grep -v '^$' | wc -w) -ne ${req_gpus} ]]; then
  echo "-1"
  exit 1
else
  echo ${f_gpu} | sed 's: :,:g'
fi
