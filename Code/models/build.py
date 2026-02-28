"""
Model Factory
=============
build_model(cfg) -> nn.Module

Supported model types  (cfg['model']['type']):
  TCN           — Temporal Convolutional Network
  LSTM          — Bidirectional LSTM
  TRANSFORMER   — Transformer Encoder
  CNN_LSTM      — CNN + BiLSTM with temporal attention (hybrid)
  GAN_CNN_LSTM  — GAN-CNN-LSTM Hybrid (GAN for data balance + CNN-LSTM classifier)
  SGAN_CNN      — Semi-supervised GAN + CNN Hybrid (feature matching, limited labels)
"""

from __future__ import annotations

import torch.nn as nn

from .tcn import TCNClassifier
from .lstm import LSTMClassifier
from .transformer import TransformerClassifier
from .cnn_lstm import CNNLSTMClassifier
from .gan_cnn_lstm import GANCNNLSTMTrainer
from .sgan_cnn import SGANCNNTrainer


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

    elif model_type == "GAN_CNN_LSTM":
        mc = cfg["model"]["gan_cnn_lstm"]
        return GANCNNLSTMTrainer(
            input_size=input_size,
            num_classes=num_classes,
            seq_len=cfg["data"]["window_size"],
            noise_dim=mc["noise_dim"],
            gan_conv_ch=mc["gan_conv_ch"],
            gan_kernel=mc["gan_kernel"],
            gan_lstm_h=mc["gan_lstm_h"],
            gan_lstm_l=mc["gan_lstm_l"],
            gan_dropout=mc["gan_dropout"],
            cls_channels=mc["cls_channels"],
            cls_kernel=mc["cls_kernel"],
            cls_lstm_h=mc["cls_lstm_h"],
            cls_lstm_l=mc["cls_lstm_l"],
            cls_dropout=mc["cls_dropout"],
        )

    elif model_type == "SGAN_CNN":
        mc = cfg["model"]["sgan_cnn"]
        return SGANCNNTrainer(
            input_size=input_size,
            num_classes=num_classes,
            seq_len=cfg["data"]["window_size"],
            noise_dim=mc["noise_dim"],
            gen_base_ch=mc["gen_base_ch"],
            gen_kernel=mc["gen_kernel"],
            gen_dropout=mc["gen_dropout"],
            disc_channels=mc["disc_channels"],
            disc_kernel=mc["disc_kernel"],
            disc_dropout=mc["disc_dropout"],
            cls_channels=mc["cls_channels"],
            cls_kernel=mc["cls_kernel"],
            cls_fc_hidden=mc["cls_fc_hidden"],
            cls_dropout=mc["cls_dropout"],
            fm_lambda=mc["fm_lambda"],
        )

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            "Choose one of: TCN | LSTM | TRANSFORMER | CNN_LSTM | GAN_CNN_LSTM | SGAN_CNN"
        )
