# ska_tdnn_utmos_projector.py

import torch

from espnet2.spk.projector.abs_projector import AbsProjector


class SkaTdnnUTMOSProjector(AbsProjector):
    def __init__(self, input_size, output_size, feat_dim):
        super().__init__()
        self._output_size = output_size

        self.bn = torch.nn.BatchNorm1d(input_size + feat_dim)
        self.fc = torch.nn.Linear(input_size + feat_dim, output_size)
        self.bn2 = torch.nn.BatchNorm1d(output_size)

    def output_size(self):
        return self._output_size

    def forward(self, x, feats):
        feats = feats.squeeze(1)
        x = torch.cat([x, feats], dim=-1)
        return self.bn2(self.fc(self.bn(x)))