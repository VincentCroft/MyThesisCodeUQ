"""
CNN-LSTM Hybrid Classifier
=========================
Architecture:
  1. Multi-layer 1D-CNN blocks  — extract local temporal / spectral features
  2. Bidirectional LSTM          — model long-range sequential dependencies
  3. Soft attention pooling      — focus on the most informative time steps
  4. Fully-connected head        — map to class logits

Input:  [B, T, F]   (batch, time-steps, features)
Output: [B, C]      (logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ──────────────────────────────────────────────────────────────
#  Helper: single CNN block (Conv → BN → ReLU → Dropout)
# ──────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    """1-D convolution block with BatchNorm, ReLU and Dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Residual projection when channel width changes
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + self.residual(x))


# ──────────────────────────────────────────────────────────────
#  Soft self-attention over time dimension
# ──────────────────────────────────────────────────────────────


class TemporalAttention(nn.Module):
    """Bahdanau-style additive attention pooling over T steps."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        weights = torch.softmax(self.score(x), dim=1)  # [B, T, 1]
        return (x * weights).sum(dim=1)  # [B, H]


# ──────────────────────────────────────────────────────────────
#  CNN-LSTM Classifier
# ──────────────────────────────────────────────────────────────


class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM hybrid classifier for PMU time-series fault detection.

    Parameters
    ----------
    input_size   : number of input features (F)
    num_classes  : number of output classes (C)
    cnn_channels : list of output channel sizes for each CNN block
    kernel_size  : kernel width used in every CNN block
    lstm_hidden  : hidden units per direction in the BiLSTM
    lstm_layers  : number of stacked LSTM layers
    dropout      : dropout rate applied throughout
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        cnn_channels: List[int] = [64, 128, 128],
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ── 1. Stacked CNN blocks ─────────────────────────────
        cnn_layers: List[nn.Module] = []
        in_ch = input_size
        for out_ch in cnn_channels:
            cnn_layers.append(ConvBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_ch = in_ch  # = cnn_channels[-1]

        # ── 2. Bidirectional LSTM ─────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── 3. Temporal attention pooling ────────────────────
        self.attention = TemporalAttention(lstm_hidden * 2)

        # ── 4. Classification head ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, F]

        # CNN expects [B, F, T]
        x = x.permute(0, 2, 1)  # [B, F, T]
        x = self.cnn(x)  # [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, C]

        # LSTM
        x, _ = self.lstm(x)  # [B, T, 2·H]

        # Attention-weighted pooling
        x = self.attention(x)  # [B, 2·H]

        return self.classifier(x)  # [B, num_classes]
