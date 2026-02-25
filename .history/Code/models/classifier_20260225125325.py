"""
PMU Fault Classifier — Backward-Compatibility Shim
===================================================
This file is kept for backward compatibility only.
All model classes now live in their own modules:
  models/tcn.py          — TCNClassifier
  models/lstm.py         — LSTMClassifier
  models/transformer.py  — TransformerClassifier
  models/cnn_lstm.py     — CNNLSTMClassifier  (new)
  models/build.py        — build_model factory
"""

# Re-export everything so any existing import still works
from .tcn import CausalConv1d, TCNBlock, TCNClassifier          # noqa: F401
from .lstm import LSTMClassifier                                 # noqa: F401
from .transformer import PositionalEncoding, TransformerClassifier  # noqa: F401
from .cnn_lstm import ConvBlock, TemporalAttention, CNNLSTMClassifier  # noqa: F401
from .build import build_model                                   # noqa: F401

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


# ══════════════════════════════════════════════════════════════
#  2.  Bi-LSTM Classifier
# ══════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════
#  3.  Transformer Encoder Classifier
# ══════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1), :])


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder classifier.
    Input:  [B, T, F]
    Output: [B, C]
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.input_proj(x))  # [B, T, d_model]
        x = self.encoder(x)  # [B, T, d_model]
        x = x.mean(dim=1)  # [B, d_model]  global avg pool
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════


def build_model(cfg: dict) -> nn.Module:
    """
    Build model from config dict.
    cfg['model']['type'] must be one of: TCN | LSTM | Transformer
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
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose TCN | LSTM | Transformer"
        )
