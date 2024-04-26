# Team 33 IDLS24 project



Baseline Architecture: [SKA-TDNN](https://arxiv.org/abs/2204.01005)  

## Speaker Verification Results

Results on Vox1-O, after training on VoxCeleb1-dev

|Model| Params |EER (%) | minDCF| Link |
|---------|---------|---------|-------|-------|
| (Ours) 3 fcw SKA-TDNN | 21.92 M | 2.297 | 0.16635 |[hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_3fcwska)|
| (Ours) 4 Kernel SKA-TDNN | 25.25 M | 2.318 | 0.16167 |[hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_4k_ska)|
| (Ours) 4msSKA-TDNN | 21.14 M |2.340 | 0.15587 | [hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_4ms_ska) |
| (Ours) Small SKA-TDNN | 19.34 M |2.351 | 0.16071 | [hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_ska_small) |
| (Ours) cwfw SKA-TDNN | 19.34 M | 2.371 | 0.15496 ||
| (Ours) 3 Kernel SKA-TDNN | 19.94 M |2.383 | 0.16050 |[hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_3k_ska)|
| (Baseline) SKA-TDNN | 36.51 M |2.383 | 0.16746 | [hf](https://huggingface.co/alexgichamba/idls24_team33_baseline) |
| RawNet3 | 16.78 M |3.105 | 0.21155 | [hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_rawnet3) |
| ECAPA-TDNN | 15.36 M |3.106 | 0.21665 | [hf](https://huggingface.co/alexgichamba/idls24_team33_vox1_ecapa) |

## Hypothesis
### 1. Fully learnable front end for raw waveform spk embedding
Progress: FAILED <br>
unable to train, because of numerical underflow. Perhaps front end needs pretraining

### 2. Parallel (branchformer-style) branch for improved global feature modelling
Progress: poor results (see branch-ecapa-tdnn)

### 3. Combining Progressive Channel Fusion (as seen in PCF-ECAPA-TDNN) and Selective Kernel Attention (as seen in SKA-TDNN)
Progress: DONE!
