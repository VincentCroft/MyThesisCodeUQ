"""
Bidirectional LSTM Classifier
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier.

    Input:  [B, T, F]
    Output: [B, C]
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # [B, T, 2H]
        last = out[:, -1, :]  # [B, 2H]
        return self.classifier(last)
