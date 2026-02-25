"""
Model Factory
=============
build_model(cfg) -> nn.Module

Supported model types  (cfg['model']['type']):
  TCN         — Temporal Convolutional Network
  LSTM        — Bidirectional LSTM
  TRANSFORMER — Transformer Encoder
  CNN_LSTM    — CNN + BiLSTM with temporal attention (hybrid)
"""

from __future__ import annotations

import torch.nn as nn

from .tcn import TCNClassifier
from .lstm import LSTMClassifier
from .transformer import TransformerClassifier
from .cnn_lstm import CNNLSTMClassifier


def build_model(cfg: dict) -> nn.Module:
    """
    Construct a model from a YAML-derived config dict.

    Parameters
    ----------
    cfg : dict
        Must contain cfg['model']['type'] and matching sub-keys.

    Returns
    -------
    nn.Module  (un-trained, on CPU)
    """
    model_type = cfg["model"]["type"].upper()
    input_size = cfg["model"]["input_size"]
    num_classes = cfg["model"]["num_classes"]

    if model_type == "TCN":
        mc = cfg["model"]["tcn"]
        return TCNClassifier(
            input_size=input_size,
            num_classes=num_classes,
            num_channels=mc["num_channels"],
            kernel_size=mc["kernel_size"],
            dropout=mc["dropout"],
        )

    elif model_type == "LSTM":
        mc = cfg["model"]["lstm"]
        return LSTMClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"],
            dropout=mc["dropout"],
        )

    elif model_type == "TRANSFORMER":
        mc = cfg["model"]["transformer"]
        return TransformerClassifier(
            input_size=input_size,
            num_classes=num_classes,
            d_model=mc["d_model"],
            nhead=mc["nhead"],
            num_encoder_layers=mc["num_encoder_layers"],
            dim_feedforward=mc["dim_feedforward"],
            dropout=mc["dropout"],
        )

    elif model_type == "CNN_LSTM":
        mc = cfg["model"]["cnn_lstm"]
        return CNNLSTMClassifier(
            input_size=input_size,
            num_classes=num_classes,
            cnn_channels=mc["cnn_channels"],
            kernel_size=mc["kernel_size"],
            lstm_hidden=mc["lstm_hidden"],
            lstm_layers=mc["lstm_layers"],
            dropout=mc["dropout"],
        )

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            "Choose one of: TCN | LSTM | TRANSFORMER | CNN_LSTM"
        )
