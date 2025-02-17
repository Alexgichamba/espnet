#! /usr/bin/python
# -*- encoding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.sasv.loss.abs_loss import AbsLoss


class MSE(AbsLoss):
    """Mean Squared Error loss for MOS prediction.

    args:
        nout: dimensionality of input embeddings
    """

    def __init__(self, nout, **kwargs):
        super().__init__(nout)
        self.in_feats = nout
        self.weight = nn.Linear(nout, 1, bias=False)
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, x, label=None):
        """Calculate MSE loss between predicted and true MOS scores.

        args:
            x: input embeddings (batch_size, nout)
            label: MOS scores (batch_size,)
        """
        if len(label.size()) == 2:
            label = label.squeeze(1)  # Handle (batch_size, 1) labels
        

        assert x.size()[0] == label.size()[0], "Batch size mismatch"
        assert x.size()[1] == self.in_feats, "Feature dimension mismatch"

        # move weights to the same device as input
        self.weight = self.weight.to(x.device)

        predicted_mos = self.weight(x)  # (batch_size, nout) -> (batch_size, 1)
        predicted_mos = predicted_mos.squeeze(1)  # (batch_size, 1) -> (batch_size,)

        # calculate loss
        loss = self.mse(predicted_mos, label)
        return loss