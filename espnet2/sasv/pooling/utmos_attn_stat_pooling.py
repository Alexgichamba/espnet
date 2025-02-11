import torch
import torch.nn as nn

from espnet2.spk.pooling.abs_pooling import AbsPooling


class UTMOSAttnStatPooling(AbsPooling):
    """Aggregates frame-level features to single utterance-level feature using attention weights for
    the weighted sum of the frame-level features.
    """

    def __init__(self, input_size: int = 1536, feat_dim: int = 2048, seq_len: int = 151 ):
        super().__init__()
        # Linear layer to project the utmos feat from feat_dim to seq_len (should be fixed)
        self.utmos_proj = nn.Linear(feat_dim, seq_len)
        # softmax layer to get the attention weights
        self.softmax = nn.Softmax(dim=-1)
        self._output_size = input_size

    def output_size(self):
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None, precomp_feats: torch.Tensor = None):
        """ Forward pass of the pooling layer.
        Args:
            x: frame-level embeddings of shape (B, D1, T)
            task_tokens: task tokens of shape (B, D2, T)
            precomp_feats: precomputed features of shape (B, 1, D2)
        Returns:
            utmos_attn_stat_pooled: utterance-level embeddings of shape (B, D1)
        """
        if task_tokens is not None:
            raise ValueError(
                "UTMOSAttnStatPooling is not adequate for task_tokens"
            )
        precomp_feats = precomp_feats.squeeze(1)
        # project the precomputed features to the seq_len
        attn = self.utmos_proj(precomp_feats)
        # apply softmax to get the attention weights
        attn = self.softmax(attn)

        # Expand dimensions so that attention weights can be broadcast with x.
        # attn: (B, T) -> (B, T, 1)
        attn = attn.unsqueeze(-1)

        # Transpose x to shape (B, T, D1)
        x = x.transpose(1, 2)
        # Compute the attention-weighted sum over the time dimension.
        # Resulting shape: (B, D1)
        out = torch.sum(attn * x, dim=1)
        return out
       
