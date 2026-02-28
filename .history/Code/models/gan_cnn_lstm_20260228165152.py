"""
GAN-CNN-LSTM Hybrid Model
=========================
Based on:
  Liu et al., "A hybrid GAN-CNN-LSTM framework for power battery fault detection"
  (adapted for PMU time-series fault classification).

Overview
--------
Phase 1 — Data Balance (GAN):
  - Generator  : noise → Conv1D → BN+ReLU → LSTM → FC+Tanh → synthetic [B, T, F]
  - Discriminator: [B, T, F] → Conv1D → BN+ReLU → Dropout → LSTM → FC+Sigmoid → prob

  Minimax objective (Eq. 2.3):
      min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

Phase 2 — Classification (CNN-LSTM):
  - Uses the balanced dataset produced by Phase 1
  - CNN layers extract spatial/local features
  - LSTM layers capture long-term temporal dependencies
  - Final dense layer → class logits

Exposed classes
---------------
  GANGenerator          : time-series generator (noise → synthetic sample)
  GANDiscriminator      : real/fake discriminator
  GANCNNLSTMClassifier  : CNN-LSTM classifier (drop-in replacement for CNNLSTMClassifier)
  GANCNNLSTMTrainer     : high-level helper that ties together GAN + classifier training

Input / Output shapes
---------------------
  Generator    : z [B, noise_dim] → x_fake [B, T, F]
  Discriminator: x [B, T, F]     → prob   [B, 1]
  Classifier   : x [B, T, F]     → logits [B, C]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ══════════════════════════════════════════════════════════════
#  Shared building-block
# ══════════════════════════════════════════════════════════════


class _ConvBNReLU(nn.Module):
    """Conv1d → BatchNorm1d → ReLU block (used in both G and D)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════
#  Generator  (noise → synthetic time-series)
# ══════════════════════════════════════════════════════════════


class GANGenerator(nn.Module):
    """
    Time-series generator following the architecture in Figure 2.4 (left).

    Architecture
    ------------
    1. Expand   : Linear(noise_dim → proj_ch * seq_len) + reshape → [B, proj_ch, T]
    2. Conv1D   : capture spatial patterns
    3. BN+ReLU  : normalise + activate
    4. LSTM     : capture temporal dependencies
    5. FC+Tanh  : project to [B, T, out_features]

    Parameters
    ----------
    noise_dim   : dimension of the latent noise vector z
    seq_len     : number of time-steps T to generate
    out_features: number of output features F (must match real data)
    proj_ch     : intermediate channel width after the Expand step
    conv_ch     : Conv1D output channels
    kernel_size : Conv1D kernel width
    lstm_hidden : LSTM hidden units
    lstm_layers : number of stacked LSTM layers
    dropout     : dropout applied inside LSTM (if lstm_layers > 1)
    """

    def __init__(
        self,
        noise_dim: int = 100,
        seq_len: int = 64,
        out_features: int = 26,
        proj_ch: int = 64,
        conv_ch: int = 128,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.proj_ch = proj_ch

        # ── Step 1 : Expand noise → feature map ──────────────
        self.expand = nn.Sequential(
            nn.Linear(noise_dim, proj_ch * seq_len),
            nn.ReLU(inplace=True),
        )

        # ── Step 2-3 : Conv1D + BN + ReLU ────────────────────
        self.conv_block = _ConvBNReLU(proj_ch, conv_ch, kernel_size)

        # ── Step 4 : LSTM ─────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=conv_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Step 5 : FC + Tanh ────────────────────────────────
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden, out_features),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : [B, noise_dim]

        Returns
        -------
        x_fake : [B, T, F]
        """
        B = z.size(0)

        # Expand → [B, proj_ch * T] → [B, proj_ch, T]
        x = self.expand(z).view(B, self.proj_ch, self.seq_len)

        # Conv block: [B, proj_ch, T] → [B, conv_ch, T]
        x = self.conv_block(x)

        # LSTM expects [B, T, C]
        x = x.permute(0, 2, 1)             # [B, T, conv_ch]
        x, _ = self.lstm(x)                 # [B, T, lstm_hidden]

        # Project to output features + Tanh
        x = self.output_layer(x)            # [B, T, out_features]
        return x


# ══════════════════════════════════════════════════════════════
#  Discriminator  (time-series → real/fake probability)
# ══════════════════════════════════════════════════════════════


class GANDiscriminator(nn.Module):
    """
    Real/fake discriminator following the architecture in Figure 2.4 (right).

    Architecture
    ------------
    1. Conv1D   : capture spatial features
    2. BN+ReLU  : normalise + activate
    3. Dropout  : prevent overfitting
    4. LSTM     : capture temporal dependencies
    5. FC+Sigmoid: output real/fake probability

    Parameters
    ----------
    in_features : input feature dimension F
    conv_ch     : Conv1D output channels
    kernel_size : Conv1D kernel width
    lstm_hidden : LSTM hidden units
    lstm_layers : number of stacked LSTM layers
    dropout     : dropout probability (steps 3 and LSTM inter-layer)
    """

    def __init__(
        self,
        in_features: int = 26,
        conv_ch: int = 128,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Step 1-3 : Conv1D + BN + ReLU + Dropout ──────────
        self.conv_block = _ConvBNReLU(in_features, conv_ch, kernel_size, dropout=dropout)

        # ── Step 4 : LSTM ─────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=conv_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Step 5 : FC + Sigmoid ─────────────────────────────
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T, F]   (real or synthetic time-series)

        Returns
        -------
        prob : [B, 1]   (1 = synthetic, 0 = real, as per Figure 2.4)
        """
        # Conv1D expects [B, F, T]
        x = x.permute(0, 2, 1)             # [B, F, T]
        x = self.conv_block(x)              # [B, conv_ch, T]

        # LSTM expects [B, T, C]
        x = x.permute(0, 2, 1)             # [B, T, conv_ch]
        x, _ = self.lstm(x)                 # [B, T, lstm_hidden]

        # Use the last time-step hidden state for classification
        x = x[:, -1, :]                     # [B, lstm_hidden]
        return self.output_layer(x)         # [B, 1]


# ══════════════════════════════════════════════════════════════
#  CNN-LSTM Classifier  (phase 2 — multi-class fault detection)
# ══════════════════════════════════════════════════════════════


class _ConvResBlock(nn.Module):
    """Residual Conv1D block (Conv → BN → ReLU → Dropout + skip)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + self.residual(x))


class _TemporalAttention(nn.Module):
    """Bahdanau-style additive attention pooling over the time axis."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)   # [B, T, 1]
        return (x * weights).sum(dim=1)                  # [B, H]


class GANCNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM classifier — Phase 2 of the GAN-CNN-LSTM pipeline.

    This module is structurally equivalent to ``CNNLSTMClassifier``
    but is packaged together with the GAN components so the full
    pipeline lives in one file.  It can be used as a drop-in classifier
    after the GAN has balanced the training dataset.

    Parameters
    ----------
    input_size   : number of input features F
    num_classes  : output classes C
    cnn_channels : list of channel widths for each residual CNN block
    kernel_size  : kernel width for all CNN blocks
    lstm_hidden  : hidden units per direction in the BiLSTM
    lstm_layers  : number of stacked BiLSTM layers
    dropout      : dropout rate used throughout
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

        # ── CNN backbone ──────────────────────────────────────
        layers: List[nn.Module] = []
        in_ch = input_size
        for out_ch in cnn_channels:
            layers.append(_ConvResBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self._cnn_out_ch = in_ch

        # ── Bidirectional LSTM ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self._cnn_out_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Temporal attention pooling ────────────────────────
        self.attention = _TemporalAttention(lstm_hidden * 2)

        # ── Classification head ───────────────────────────────
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
        """
        Parameters
        ----------
        x : [B, T, F]

        Returns
        -------
        logits : [B, C]
        """
        x = x.permute(0, 2, 1)         # [B, F, T]
        x = self.cnn(x)                 # [B, C_cnn, T]
        x = x.permute(0, 2, 1)         # [B, T, C_cnn]
        x, _ = self.lstm(x)             # [B, T, 2·H]
        x = self.attention(x)           # [B, 2·H]
        return self.classifier(x)       # [B, num_classes]


# ══════════════════════════════════════════════════════════════
#  High-level trainer helper
# ══════════════════════════════════════════════════════════════


class GANCNNLSTMTrainer(nn.Module):
    """
    Full GAN-CNN-LSTM pipeline wrapper.

    Bundles the Generator, Discriminator, and CNN-LSTM Classifier
    into a single module that exposes the adversarial loss helpers
    and the standard ``forward`` pass for inference.

    Typical training workflow
    -------------------------
    1. Alternate GAN training steps (``train_gan_step``) to balance data.
    2. Feed both real + generated samples to ``train_classifier_step``.
    3. During inference, call ``forward`` (delegates to the classifier).

    Parameters
    ----------
    input_size   : feature dimension F of real data
    num_classes  : number of fault classes C
    seq_len      : time-series window length T
    noise_dim    : latent noise dimension for the generator
    gan_conv_ch  : Conv1D channels used in G and D
    gan_kernel   : Conv1D kernel size for G and D
    gan_lstm_h   : LSTM hidden units for G and D
    gan_lstm_l   : LSTM layers for G and D
    gan_dropout  : dropout used in G and D
    cls_channels : CNN block channel widths for the classifier
    cls_kernel   : CNN kernel size for the classifier
    cls_lstm_h   : BiLSTM hidden units for the classifier
    cls_lstm_l   : BiLSTM layers for the classifier
    cls_dropout  : dropout for the classifier
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        seq_len: int = 64,
        noise_dim: int = 100,
        # GAN hyper-params
        gan_conv_ch: int = 128,
        gan_kernel: int = 3,
        gan_lstm_h: int = 128,
        gan_lstm_l: int = 2,
        gan_dropout: float = 0.3,
        # Classifier hyper-params
        cls_channels: List[int] = [64, 128, 128],
        cls_kernel: int = 3,
        cls_lstm_h: int = 128,
        cls_lstm_l: int = 2,
        cls_dropout: float = 0.2,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.seq_len = seq_len

        self.generator = GANGenerator(
            noise_dim=noise_dim,
            seq_len=seq_len,
            out_features=input_size,
            conv_ch=gan_conv_ch,
            kernel_size=gan_kernel,
            lstm_hidden=gan_lstm_h,
            lstm_layers=gan_lstm_l,
            dropout=gan_dropout,
        )
        self.discriminator = GANDiscriminator(
            in_features=input_size,
            conv_ch=gan_conv_ch,
            kernel_size=gan_kernel,
            lstm_hidden=gan_lstm_h,
            lstm_layers=gan_lstm_l,
            dropout=gan_dropout,
        )
        self.classifier = GANCNNLSTMClassifier(
            input_size=input_size,
            num_classes=num_classes,
            cnn_channels=cls_channels,
            kernel_size=cls_kernel,
            lstm_hidden=cls_lstm_h,
            lstm_layers=cls_lstm_l,
            dropout=cls_dropout,
        )

    # ── GAN loss helpers ──────────────────────────────────────

    def discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss for the discriminator.

        Convention (same as Figure 2.4):
          label 1 → synthetic / fake
          label 0 → real

        Parameters
        ----------
        real : [B, T, F]  real samples
        fake : [B, T, F]  generated samples (detached)

        Returns
        -------
        loss : scalar tensor
        """
        bce = nn.BCELoss()
        B = real.size(0)
        device = real.device

        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake.detach())

        loss_real = bce(d_real, torch.zeros(B, 1, device=device))   # real → 0
        loss_fake = bce(d_fake, torch.ones(B, 1, device=device))    # fake → 1
        return (loss_real + loss_fake) * 0.5

    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """
        Generator objective: fool the discriminator (label fake as real).

        Parameters
        ----------
        fake : [B, T, F]  generated samples

        Returns
        -------
        loss : scalar tensor
        """
        bce = nn.BCELoss()
        B = fake.size(0)
        device = fake.device

        d_fake = self.discriminator(fake)
        # Generator wants discriminator to output 0 (real) for fake samples
        return bce(d_fake, torch.zeros(B, 1, device=device))

    # ── Convenience: sample noise ─────────────────────────────

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Draw z ~ N(0,1) with shape [batch_size, noise_dim]."""
        return torch.randn(batch_size, self.noise_dim, device=device)

    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate a batch of synthetic time-series samples."""
        z = self.sample_noise(batch_size, device)
        return self.generator(z)

    # ── Inference: delegate to classifier ────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard inference forward pass through the CNN-LSTM classifier.

        Parameters
        ----------
        x : [B, T, F]

        Returns
        -------
        logits : [B, C]
        """
        return self.classifier(x)
