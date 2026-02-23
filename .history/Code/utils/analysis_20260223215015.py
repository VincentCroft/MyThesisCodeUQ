"""
PMU Fault Classifier — Analysis Utilities
File: Code/utils/analysis.py

Provides:
  - collect_predictions()      : gather model predictions & true labels from a DataLoader
  - save_confusion_matrix()    : compute & save confusion matrix as JSON
  - save_tsne_embeddings()     : extract penultimate-layer features, run t-SNE, save as JSON
  - load_confusion_matrix()    : load saved confusion matrix JSON
  - load_tsne_embeddings()     : load saved t-SNE JSON
  - compute_classification_report() : per-class precision / recall / F1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Optional heavy imports (graceful fallback) ─────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        precision_recall_fscore_support,
    )
    _SK_OK = True
except ImportError:
    _SK_OK = False


# ════════════════════════════════════════════════════════════
#  1.  Collect predictions from a DataLoader
# ════════════════════════════════════════════════════════════

def collect_predictions(
    model: "nn.Module",
    loader: "DataLoader",
    device: "torch.device",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference over all batches in `loader`.

    Returns
    -------
    all_preds : np.ndarray[int]  shape [N]
    all_true  : np.ndarray[int]  shape [N]
    """
    assert _TORCH_OK, "PyTorch not installed."
    model.eval()
    preds_list, true_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            preds   = logits.argmax(dim=1).cpu().numpy()
            preds_list.append(preds)
            true_list.append(y_batch.numpy())

    return (
        np.concatenate(preds_list, axis=0),
        np.concatenate(true_list,  axis=0),
    )


# ════════════════════════════════════════════════════════════
#  2.  Confusion Matrix
# ════════════════════════════════════════════════════════════

def save_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    class_names: List[str],
    save_path:  Path,
) -> dict:
    """
    Compute confusion matrix and per-class metrics, then save to JSON.

    Saved JSON structure
    --------------------
    {
      "class_names": [...],
      "matrix": [[...], ...],          # raw counts [N_cls × N_cls]
      "matrix_norm": [[...], ...],     # row-normalised (recall per cell)
      "per_class": {
          "NORMAL": {"precision": 0.97, "recall": 0.98, "f1": 0.975, "support": 1234},
          ...
      },
      "overall_accuracy": 0.965
    }
    """
    assert _SK_OK, "scikit-learn not installed."
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    # Row-normalise (each row sums to 1 = recall per true class)
    row_sums  = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm   = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    # Per-class P / R / F1
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1":        round(float(f1[i]),   4),
            "support":   int(sup[i]),
        }

    overall_acc = float((y_true == y_pred).mean())

    result = {
        "class_names":       class_names,
        "matrix":            cm.tolist(),
        "matrix_norm":       [[round(v, 4) for v in row] for row in cm_norm.tolist()],
        "per_class":         per_class,
        "overall_accuracy":  round(overall_acc, 6),
    }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def load_confusion_matrix(path: Path) -> dict:
    """Load previously saved confusion matrix JSON."""
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════
#  3.  t-SNE Embeddings
# ════════════════════════════════════════════════════════════

class _FeatureHook:
    """Forward hook to capture penultimate-layer activations."""
    def __init__(self):
        self.features: list[np.ndarray] = []

    def __call__(self, module, input, output):
        self.features.append(output.detach().cpu().numpy())

    def reset(self):
        self.features = []

    def get(self) -> np.ndarray:
        return np.concatenate(self.features, axis=0)


def _get_penultimate_module(model: "nn.Module") -> "nn.Module":
    """
    Heuristically find the last Linear layer before the final classifier head.
    Works for TCN, LSTM, Transformer as defined in classifier.py.
    """
    # All three architectures end with model.classifier = nn.Sequential(...)
    clf = getattr(model, "classifier", None)
    if clf is not None and isinstance(clf, nn.Sequential):
        # Walk backwards to find the second-to-last Linear
        linears = [(i, m) for i, m in enumerate(clf) if isinstance(m, nn.Linear)]
        if len(linears) >= 2:
            return linears[-2][1]   # penultimate Linear
        if len(linears) == 1:
            return linears[-1][1]
    # Fallback: return whole model
    return model


def save_tsne_embeddings(
    model:       "nn.Module",
    loader:      "DataLoader",
    device:      "torch.device",
    class_names: List[str],
    save_path:   Path,
    max_samples: int = 3000,
    perplexity:  int = 40,
    random_state: int = 42,
) -> dict:
    """
    Extract penultimate-layer features for up to `max_samples` validation
    samples, run t-SNE (2-D), and save to JSON.

    Saved JSON structure
    --------------------
    {
      "class_names": [...],
      "x": [float, ...],         # t-SNE dim-1
      "y": [float, ...],         # t-SNE dim-2
      "labels": [int, ...],      # true class indices
      "label_names": [str, ...]  # true class names
    }
    """
    assert _TORCH_OK and _SK_OK, "PyTorch and scikit-learn must be installed."
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    hook_fn = _FeatureHook()
    target_module = _get_penultimate_module(model)
    handle = target_module.register_forward_hook(hook_fn)

    model.eval()
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            model(X_batch)
            all_true.append(y_batch.numpy())
            if sum(len(b) for b in hook_fn.features) >= max_samples:
                break

    handle.remove()

    feats      = hook_fn.get()[:max_samples]
    all_true   = np.concatenate(all_true, axis=0)[:max_samples]

    # t-SNE reduction
    tsne       = TSNE(n_components=2, perplexity=min(perplexity, len(feats) - 1),
                      random_state=random_state, n_iter=1000, init="pca")
    coords     = tsne.fit_transform(feats)   # [N, 2]

    label_names = [class_names[i] for i in all_true]

    result = {
        "class_names": class_names,
        "x":           [round(float(v), 4) for v in coords[:, 0]],
        "y":           [round(float(v), 4) for v in coords[:, 1]],
        "labels":      all_true.tolist(),
        "label_names": label_names,
    }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def load_tsne_embeddings(path: Path) -> dict:
    """Load previously saved t-SNE JSON."""
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════
#  4.  Classification Report (dict form, GUI-friendly)
# ════════════════════════════════════════════════════════════

def compute_classification_report(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: List[str],
) -> dict:
    """
    Returns a dict with per-class and overall metrics.
    Compatible with pandas DataFrame for tabular display.
    """
    assert _SK_OK, "scikit-learn not installed."
    n = len(class_names)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0
    )
    rows = []
    for i, cls in enumerate(class_names):
        rows.append({
            "Class":     cls,
            "Precision": round(float(prec[i]), 4),
            "Recall":    round(float(rec[i]),  4),
            "F1-Score":  round(float(f1[i]),   4),
            "Support":   int(sup[i]),
        })
    # Macro avg
    rows.append({
        "Class":     "macro avg",
        "Precision": round(float(prec.mean()), 4),
        "Recall":    round(float(rec.mean()),  4),
        "F1-Score":  round(float(f1.mean()),   4),
        "Support":   int(sup.sum()),
    })
    # Overall accuracy
    rows.append({
        "Class":     "accuracy",
        "Precision": "-",
        "Recall":    "-",
        "F1-Score":  round(float((y_true == y_pred).mean()), 4),
        "Support":   int(len(y_true)),
    })
    return rows
