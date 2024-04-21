# Team 33 IDLS24 project

## Baseline

Architecture: [SKA-TDNN](https://arxiv.org/abs/2204.01005)  

Hyperparams and training configs at [conf/train_ska_mel.yaml](conf/train_ska_mel.yaml)

Results on Vox1-O, after training on VoxCeleb1-dev

|Model| Params |EER (%) | minDCF| Link |
|---------|---------|---------|-------|-------|
| (Ours) Small 3Kernel SKA-TDNN | 19.94 M |2.601 | 0.184 |||
| (Ours) Small SKA-TDNN | 19.34 M |2.654 | 0.175 | [hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_ska_small) |
| SKA-TDNN | 36.51 M |2.665 | 0.191 | [hf](https://huggingface.co/alexgichamba/idls24_team33_baseline) |
| RawNet3 | 16.78 M |3.181 | 0.218 | [hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_rawnet3) |
| ECAPA-TDNN | 15.36 M |3.260 | 0.224 | [hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_ecapa) |
| Branch-ECAPA-TDNN | 34.25 M |s3.525 | 0.243 | [hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_branch_ecapa) |

## Hypothesis
### 1. Fully learnable front end for raw waveform spk embedding
Progress: FAILED <br>
unable to train, because of numerical underflow. Perhaps front end needs pretraining

### 2. Parallel (branchformer-style) branch for improved global feature modelling
Progress: poor results (see branch-ecapa-tdnn)

### 3. Combining Progressive Channel Fusion (as seen in PCF-ECAPA-TDNN) and Selective Kernel Attention (as seen in SKA-TDNN)
|Training Config|Assigned to|EER (%) | minDCF|
|---------|---------|-------|-------|
|[conf/train_swap_ska.yaml](conf/train_swap_ska.yaml)|Clifford|||
|[conf/train_quadms_ska.yaml](conf/train_quadms_ska.yaml)|Brian|||
|[conf/train_fw_cw_ska_tdnn.yaml](conf/train_fw_cw_ska_tdnn.yaml)|Alex|||
|[conf/train_three_kernel_ska.yaml](conf/train_three_kernel_ska.yaml)|Epa|2.601|0.184|
