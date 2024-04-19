#!/usr/bin/env python3
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, AnalyticFreeFB
from typeguard import typechecked
from espnet2.asr.frontend.abs_frontend import AbsFrontend

class AnalyticFreeFrontend(AbsFrontend):
    """Asteroid Filterbank Frontend.

    Provides a fully learnable analytic filterbank feature extractor.

    """

    @typechecked
    def __init__(
        self,
        n_filters: int = 256,
        kernel_size: int = 251,
        stride: int = 16,
        sample_rate: int = 16000,
        preemph_coef: float = 0.97,
        log_term: float = 1e-6,
    ):
        """Initialize.

        Args:
            n_filters: the filter numbers for analytic free fb.
            kernel_size: the kernel size for analytic free fb.
            stride: the stride size of the first conv layer where it decides
                the compression rate (Hz).
            preemph_coef: the coeifficient for preempahsis.
            log_term: the log term to prevent infinity.
        """
        super().__init__()
        # kernel for preemphasis
        # In PyTorch, the convolution operation uses cross-correlation,
        # so the filter is flipped
        self.register_buffer(
            "flipped_filter", torch.FloatTensor([-preemph_coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )
        self.norm = nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        self.n_filters = n_filters
        self.conv = Encoder(AnalyticFreeFB(n_filters, kernel_size, stride=stride))
        self.log_term = log_term
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_dim = n_filters

    def forward(
        self, input: torch.Tensor, input_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Asteroid filterbank frontend to the input.

        Args:
            input: Input (B, T).
            input_length: Input length (B,).

        Returns:
            Tensor: Frame-wise output (B, T', D).
        """
        # input check
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"
        with torch.cuda.amp.autocast(enabled=False):
            # reflect padding to match lengths of in/out
            x = input.unsqueeze(1)
            x = F.pad(x, (1, 0), "reflect")
            # apply preemphasis
            x = F.conv1d(x, self.flipped_filter)
            # apply norm
            x = self.norm(x)
            # apply frame feature extraction
            x = torch.log(torch.abs(self.conv(x)) + self.log_term)
            input_length = (input_length - self.kernel_size) // self.stride + 1
            x = x - torch.mean(x, dim=-1, keepdim=True)
        return x.permute(0, 2, 1), input_length

    def output_size(self) -> int:
        """Return output length of feature dimension D."""
        return self.n_filters