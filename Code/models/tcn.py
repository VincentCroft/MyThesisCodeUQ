"""
Temporal Convolutional Network (TCN) Classifier
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CausalConv1d(nn.Module):
    """Causal dilated 1-D convolution with left-padding."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.drop(F.relu(self.bn(self.conv(x))))


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation, dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + res)


class TCNClassifier(nn.Module):
    """
    Temporal Convolutional Network for sequence classification.

    Input:  [B, T, F]   (batch, time, features)
    Output: [B, C]      (logits)
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2**i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.network(x)
        return self.classifier(x)
