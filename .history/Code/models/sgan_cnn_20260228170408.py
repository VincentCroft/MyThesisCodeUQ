"""
SGAN-CNN Hybrid Fault Diagnosis Model
======================================
Based on:
  [9]  Semi-supervised GAN + CNN for fault diagnosis under limited labels
       and class imbalance (Figure 2.6 – 2.8).

Overview
--------
The framework has two coupled branches:

  Branch 1 — SGAN (Semi-supervised GAN)
  ───────────────────────────────────────
  • Generator   : Noise → multi-path Deconv → Conv → synthetic [B, T, F]
                  (Figure 2.8: three parallel Deconv→Conv streams merged at output)
  • Discriminator: [B, T, F] →  Conv stack  →  two heads:
        ① Real/Fake head   (for unsupervised adversarial loss  L_unsupervised)
        ② Class logits head (for supervised cross-entropy loss L_supervised
                             on the small labeled subset)
    Intermediate activations are exposed for feature-matching loss.

  Branch 2 — CNN Classifier  (Figure 2.7 left)
  ─────────────────────────────────────────────
  • Conv1D → MaxPool → Conv1D → MaxPool → fully-connected → logits
  • Trained on the balanced (real + quality-filtered synthetic) dataset.

Combined objective (Eq. 2.4):
    L = L_supervised  +  L_unsupervised  +  L_G  +  λ · L_feature_matching

Feature matching (Figure 2.8):
    Forces the generator to match the mean/variance of intermediate
    discriminator features between real and synthetic batches, preventing
    mode collapse and ensuring synthetic samples pass a "closeness" check.

Exposed classes
---------------
  SGANGenerator           : noise (z) → synthetic time-series [B, T, F]
  SGANDiscriminator       : dual-head — real/fake prob + class logits
  SGANCNNClassifier       : CNN-only classifier for inference
  SGANCNNTrainer          : full pipeline wrapper with all loss helpers

Input / Output shapes
---------------------
  Generator    : z     [B, noise_dim]  → x_fake [B, T, F]
  Discriminator: x     [B, T, F]       → (prob [B,1],  logits [B, C])
  Classifier   : x     [B, T, F]       → logits [B, C]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════
#  Internal building blocks
# ══════════════════════════════════════════════════════════════


class _ConvBNReLU(nn.Module):
    """Conv1d → BN → ReLU (+ optional Dropout)."""

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


class _DeconvBNReLU(nn.Module):
    """ConvTranspose1d → BN → ReLU block (used in generator branches)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2 - (1 if stride == 2 else 0),
                output_padding=stride - 1,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════
#  SGAN Generator  (Figure 2.8 left — multi-path deconv)
# ══════════════════════════════════════════════════════════════


class SGANGenerator(nn.Module):
    """
    Multi-path generator following Figure 2.8.

    Architecture
    ───────────
    Noise z → Linear expand → reshape → 3 parallel (Deconv→Conv) branches
    → concatenate along channel → final Conv1d+Tanh → synthetic [B, T, F]

    The three parallel paths allow the generator to learn diverse temporal
    patterns at different scales, mirroring the three-branch layout in Fig 2.8.

    Parameters
    ----------
    noise_dim    : latent noise dimension
    seq_len      : output time-steps T
    out_features : output feature count F (matches real data)
    base_ch      : base channel width for each branch
    kernel_size  : Conv1D kernel width in each branch
    dropout      : dropout applied in the merging Conv
    """

    def __init__(
        self,
        noise_dim: int = 100,
        seq_len: int = 64,
        out_features: int = 26,
        base_ch: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.base_ch = base_ch
        self._init_len = max(seq_len // 4, 1)

        # ── Expand noise into initial feature map ─────────────
        self.expand = nn.Sequential(
            nn.Linear(noise_dim, base_ch * 3 * self._init_len),
            nn.ReLU(inplace=True),
        )

        # ── 3 parallel Deconv → Conv branches (Figure 2.8) ───
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    _DeconvBNReLU(base_ch, base_ch * 2, kernel_size=4, stride=2),
                    _DeconvBNReLU(base_ch * 2, base_ch, kernel_size=4, stride=2),
                    _ConvBNReLU(base_ch, base_ch, kernel_size),
                )
                for _ in range(3)
            ]
        )

        # ── Merge branches → output ───────────────────────────
        self.merge = nn.Sequential(
            nn.Conv1d(
                base_ch * 3,
                base_ch * 2,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(base_ch * 2, out_features, 1, bias=True),
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
        # Expand and reshape to 3 branches: [B, base_ch, init_len] each
        x = self.expand(z).view(B, 3, self.base_ch, self._init_len)

        branch_outs: List[torch.Tensor] = []
        for i, branch in enumerate(self.branches):
            bi = x[:, i]  # [B, base_ch, init_len]
            bi = branch(bi)  # [B, base_ch, T]
            branch_outs.append(bi)

        # Align lengths (small rounding differences from stride/padding)
        min_len = min(b.size(2) for b in branch_outs)
        branch_outs = [b[:, :, :min_len] for b in branch_outs]

        out = torch.cat(branch_outs, dim=1)  # [B, base_ch*3, T']
        out = self.merge(out)  # [B, F, T']

        # Crop or pad to exactly seq_len
        T_out = out.size(2)
        if T_out >= self.seq_len:
            out = out[:, :, : self.seq_len]
        else:
            pad = self.seq_len - T_out
            out = F.pad(out, (0, pad))

        return out.permute(0, 2, 1)  # [B, T, F]


# ══════════════════════════════════════════════════════════════
#  SGAN Discriminator  (dual-head: real/fake + class logits)
# ══════════════════════════════════════════════════════════════


class SGANDiscriminator(nn.Module):
    """
    Semi-supervised discriminator (Figure 2.7 right & 2.8 right).

    Dual output heads:
      ① ``prob``   [B, 1]  — real(0) / fake(1) sigmoid probability
                             → feeds L_unsupervised adversarial loss
      ② ``logits`` [B, C]  — class logits (softmax for labeled samples)
                             → feeds L_supervised cross-entropy loss

    Intermediate CNN features are accessible via ``extract_features()``
    for the feature-matching loss (Figure 2.8 closeness calculation).

    Parameters
    ----------
    in_features : input feature dimension F
    num_classes : number of fault classes C
    conv_channels: progressive Conv1D widths
    kernel_size : Conv kernel width
    dropout     : dropout in the shared trunk
    """

    def __init__(
        self,
        in_features: int = 26,
        num_classes: int = 4,
        conv_channels: List[int] = [64, 128, 128],
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Shared convolutional trunk ─────────────────────────
        trunk: List[nn.Module] = []
        in_ch = in_features
        for out_ch in conv_channels:
            trunk.append(_ConvBNReLU(in_ch, out_ch, kernel_size, dropout=dropout))
            trunk.append(nn.MaxPool1d(2, stride=2, padding=0))
            in_ch = out_ch
        self.trunk = nn.Sequential(*trunk)
        self._trunk_out_ch = in_ch

        # ── Global avg pool + flatten ──────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── Head ①: real / fake ────────────────────────────────
        self.real_fake_head = nn.Sequential(
            nn.Linear(self._trunk_out_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── Head ②: class logits (for labeled samples) ─────────
        self.class_head = nn.Sequential(
            nn.Linear(self._trunk_out_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return trunk feature vector for feature-matching loss.

        Parameters
        ----------
        x : [B, T, F]

        Returns
        -------
        feat : [B, trunk_out_ch]
        """
        h = self.trunk(x.permute(0, 2, 1))  # [B, C, T']
        h = self.gap(h).squeeze(-1)  # [B, C]
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, T, F]

        Returns
        -------
        prob   : [B, 1]   — real/fake probability (1 = fake)
        logits : [B, C]   — unnormalized class scores
        """
        feat = self.extract_features(x)  # [B, C]
        prob = self.real_fake_head(feat)  # [B, 1]
        logits = self.class_head(feat)  # [B, C]
        return prob, logits


# ══════════════════════════════════════════════════════════════
#  CNN Classifier  (Figure 2.7 left — inference branch)
# ══════════════════════════════════════════════════════════════


class SGANCNNClassifier(nn.Module):
    """
    Pure CNN diagnostic branch (Figure 2.7 left).

    Architecture
    ───────────
    Conv1D → MaxPool → Conv1D → MaxPool → Flatten → FC → ReLU → Dropout → FC → logits

    Trained on the balanced dataset (real + quality-filtered synthetic samples)
    produced by the SGAN adversarial phase.

    Parameters
    ----------
    input_size   : feature dimension F
    num_classes  : output classes C
    conv_channels: channel widths for the stacked Conv blocks
    kernel_size  : Conv1D kernel width
    fc_hidden    : hidden units in the FC layer
    dropout      : dropout rate
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        conv_channels: List[int] = [64, 128, 128],
        kernel_size: int = 3,
        fc_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ── Convolutional + Pooling stack ──────────────────────
        cnn: List[nn.Module] = []
        in_ch = input_size
        for out_ch in conv_channels:
            cnn.append(_ConvBNReLU(in_ch, out_ch, kernel_size, dropout=dropout))
            cnn.append(nn.MaxPool1d(2, stride=2, padding=0))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn)

        self.gap = nn.AdaptiveAvgPool1d(1)  # global avg pool → fixed size

        # ── Fully-connected head ───────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(in_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
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
        x = x.permute(0, 2, 1)  # [B, F, T]
        x = self.cnn(x)  # [B, C, T']
        x = self.gap(x).squeeze(-1)  # [B, C]
        return self.fc(x)  # [B, num_classes]


# ══════════════════════════════════════════════════════════════
#  Full SGAN-CNN Pipeline Trainer
# ══════════════════════════════════════════════════════════════


class SGANCNNTrainer(nn.Module):
    """
    Full SGAN-CNN pipeline (Figures 2.6 – 2.8).

    Bundles Generator, Discriminator and CNN Classifier.  Provides
    loss helpers that implement the combined objective (Eq. 2.4):

        L = L_supervised + L_unsupervised + L_G + λ · L_feature_matching

    Typical training loop
    ─────────────────────
    # Phase 1 — SGAN adversarial training (data balance)
    for batch in dataloader:
        # Step A: update discriminator
        d_loss = trainer.discriminator_loss(real, fake, labels, mask)
        d_loss.backward(); opt_D.step()

        # Step B: update generator
        z = trainer.sample_noise(B, device)
        fake = trainer.generator(z)
        g_loss = trainer.generator_loss(fake, real)
        g_loss.backward(); opt_G.step()

    # Phase 2 — CNN classifier on balanced dataset
    cls_loss = F.cross_entropy(trainer.classifier(x_balanced), y_balanced)
    cls_loss.backward(); opt_cls.step()

    # Inference
    logits = trainer(x)     # → delegates to CNN classifier

    Parameters
    ----------
    input_size       : feature dimension F of real data
    num_classes      : fault classes C
    seq_len          : window length T
    noise_dim        : generator latent dimension
    gen_base_ch      : base channels in generator branches
    gen_kernel       : Conv kernel size in generator
    gen_dropout      : generator merge-conv dropout
    disc_channels    : discriminator trunk Conv channel widths
    disc_kernel      : discriminator Conv kernel size
    disc_dropout     : discriminator dropout
    cls_channels     : CNN classifier Conv channel widths
    cls_kernel       : classifier Conv kernel size
    cls_fc_hidden    : classifier FC hidden units
    cls_dropout      : classifier dropout
    fm_lambda        : feature-matching loss weight λ
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        seq_len: int = 64,
        noise_dim: int = 100,
        # Generator
        gen_base_ch: int = 64,
        gen_kernel: int = 3,
        gen_dropout: float = 0.2,
        # Discriminator
        disc_channels: List[int] = [64, 128, 128],
        disc_kernel: int = 3,
        disc_dropout: float = 0.3,
        # CNN classifier
        cls_channels: List[int] = [64, 128, 128],
        cls_kernel: int = 3,
        cls_fc_hidden: int = 128,
        cls_dropout: float = 0.2,
        # Feature matching weight
        fm_lambda: float = 1.0,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.fm_lambda = fm_lambda

        self.generator = SGANGenerator(
            noise_dim=noise_dim,
            seq_len=seq_len,
            out_features=input_size,
            base_ch=gen_base_ch,
            kernel_size=gen_kernel,
            dropout=gen_dropout,
        )
        self.discriminator = SGANDiscriminator(
            in_features=input_size,
            num_classes=num_classes,
            conv_channels=disc_channels,
            kernel_size=disc_kernel,
            dropout=disc_dropout,
        )
        self.classifier = SGANCNNClassifier(
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=cls_channels,
            kernel_size=cls_kernel,
            fc_hidden=cls_fc_hidden,
            dropout=cls_dropout,
        )

    # ── Noise sampling ────────────────────────────────────────

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Draw z ~ N(0,1) with shape [batch_size, noise_dim]."""
        return torch.randn(batch_size, self.noise_dim, device=device)

    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate a batch of synthetic time-series samples."""
        z = self.sample_noise(batch_size, device)
        return self.generator(z)

    # ── Loss helpers (Eq. 2.4) ────────────────────────────────

    def discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combined discriminator loss:
            L_D = L_supervised  +  L_unsupervised

        L_unsupervised (adversarial, all samples):
            real → prob ≈ 0 (real),  fake → prob ≈ 1 (fake)

        L_supervised (labeled real samples only):
            cross-entropy on class logits for masked samples

        Parameters
        ----------
        real       : [B, T, F]   real samples
        fake       : [B, T, F]   generated samples (detached from G graph)
        labels     : [B]         integer class labels (needed for L_supervised)
        label_mask : [B] bool    True for labeled samples in the batch
                                 (implements 10% labeled / 90% unlabeled split)

        Returns
        -------
        total_loss : scalar
        """
        bce = nn.BCELoss()
        ce = nn.CrossEntropyLoss()
        B = real.size(0)
        device = real.device

        # ── L_unsupervised ────────────────────────────────────
        prob_real, _ = self.discriminator(real)
        prob_fake, _ = self.discriminator(fake.detach())

        loss_real = bce(prob_real, torch.zeros(B, 1, device=device))  # real → 0
        loss_fake = bce(prob_fake, torch.ones(B, 1, device=device))  # fake → 1
        l_unsup = (loss_real + loss_fake) * 0.5

        # ── L_supervised (labeled subset only) ───────────────
        l_sup = torch.tensor(0.0, device=device)
        if labels is not None:
            if label_mask is None:
                label_mask = torch.ones(B, dtype=torch.bool, device=device)
            if label_mask.any():
                _, logits_labeled = self.discriminator(real[label_mask])
                l_sup = ce(logits_labeled, labels[label_mask])

        return l_sup + l_unsup

    def generator_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generator objective:
            L_G = L_adversarial + λ · L_feature_matching

        L_adversarial  : fool discriminator → fake should be judged real (prob→0)
        L_feature_matching (Figure 2.8):
            || E[feat(real)] - E[feat(fake)] ||²   (mean feature alignment)
            force G to match mean + variance of intermediate D features

        Parameters
        ----------
        fake : [B, T, F]   generated samples  (connected to G)
        real : [B, T, F]   real samples        (no grad)

        Returns
        -------
        total_loss : scalar
        """
        bce = nn.BCELoss()
        B = fake.size(0)
        device = fake.device

        # Adversarial (fool D)
        prob_fake, _ = self.discriminator(fake)
        l_adv = bce(prob_fake, torch.zeros(B, 1, device=device))

        # Feature matching
        with torch.no_grad():
            feat_real = self.discriminator.extract_features(real)  # [B, C]
        feat_fake = self.discriminator.extract_features(fake)  # [B, C]

        # Match mean (closeness measure, Figure 2.8)
        l_fm = F.mse_loss(feat_fake.mean(0), feat_real.mean(0)) + F.mse_loss(
            feat_fake.std(0), feat_real.std(0)
        )

        return l_adv + self.fm_lambda * l_fm

    # ── Inference ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference: delegate to CNN classifier.

        Parameters
        ----------
        x : [B, T, F]

        Returns
        -------
        logits : [B, C]
        """
        return self.classifier(x)
