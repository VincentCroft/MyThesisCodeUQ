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
from .tcn import CausalConv1d, TCNBlock, TCNClassifier  # noqa: F401
from .lstm import LSTMClassifier  # noqa: F401
from .transformer import PositionalEncoding, TransformerClassifier  # noqa: F401
from .cnn_lstm import ConvBlock, TemporalAttention, CNNLSTMClassifier  # noqa: F401
from .build import build_model  # noqa: F401
