import math

import torch
import torch.nn as nn

"""
Blocks for Branch-ECAPA-TDNN.
    paper:  Yao, J., Liang, C., Peng, Z., Zhang, B., Zhang, X.-L. (2023)
     Branch-ECAPA-TDNN: A Parallel Branch Architecture to Capture Local
      and Global Features for Speaker Verification. Proc. INTERSPEECH 2023,
        1943-1947, doi: 10.21437/Interspeech.2023-402
"""


class SEModule(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class EcapaBranch(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out

class AttentionBranch(nn.Module):
    """
    Parallel attention branch for Branch-ECAPA-TDNN.
    """
    def __init__(self, channels, num_heads, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels,
                                         num_heads=num_heads,
                                         dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = x.permute(1, 0, 2)
        x, _ = self.mha(query=x, key=x, value=x)
        x = x.permute(1, 0, 2)
        x = self.dropout(x)
        return x


class BranchEcapaBlock(nn.Module):
    """
    Branch-ECAPA-TDNN block.
    """
    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        merge_method: str = "concat",
        **kwargs,
    ):
        super().__init__()
        self.ecapa_branch = EcapaBranch(inplanes,
                                         planes,
                                           kernel_size,
                                             dilation,
                                               scale)
        self.attn_branch = AttentionBranch(planes, num_heads=4)
        self.merge_method = merge_method
        self.linear_cat = nn.Linear(planes * 2, planes)
    
    def forward(self, x):
        ecapa_out = self.ecapa_branch(x)
        attn_out = self.parallel_attn_branch(x)

        cat = torch.cat((ecapa_out, attn_out), dim=1)

        if self.merge_method == "concat":
            x = self.linear_cat(cat)
        else:
            raise NotImplementedError(f"merge method {self.merge_method} not implemented")
        
        return x

