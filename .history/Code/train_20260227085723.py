"""
PMU Fault Classifier — train.py
Usage:
    python train.py [--model TCN|LSTM|Transformer|CNN_LSTM] [--run-name my_experiment]

Each training run is saved to:
    Code/logs/checkpoints/<run_name>/
If --run-name is omitted the current timestamp is used (e.g. 2024-05-21-23-59).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")

# ── Resolve project root & register module paths ────────────
CODE_DIR = Path(__file__).parent.resolve()
THESIS_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))  # 让 Python 能找到 models/ Utils/ configs/ 等子包

# ── Imports (require torch) ─────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError:
    print("❌  PyTorch not found. Run:  pip install torch")
    sys.exit(1)

from models.classifier import build_model
from models.feature_engineering import (
    CLASS_NAMES,
    FeatureNormalizer,
    load_class_data,
    sliding_windows,
)
from Utils.analysis import (
    collect_predictions,
    save_confusion_matrix,
    save_tsne_embeddings,
)
from configs.paths import (
    PROCESSED_DATA_ROOT,
    CKPT_BASE_DIR,
    TRAIN_CONFIG_YAML,
    run_paths,
)


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def export_onnx(
    model: torch.nn.Module,
    window_size: int,
    input_size: int,
    out_path: "Path",
    device: torch.device,
) -> bool:
    """
    Export model to ONNX format.

    The dummy input shape is [1, T, F]  (batch=1, time=window_size, features=input_size).
    Dynamic axes are set for batch and time so the exported model accepts any
    sequence length at inference time — useful for streaming on Raspberry Pi / Jetson.

    Returns True on success, False if torch.onnx is unavailable or export fails.
    """
    try:
        import torch.onnx  # always present in PyTorch >= 1.x

        model.eval()
        dummy = torch.zeros(1, window_size, input_size, device=device)

        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=17,  # widely supported on edge runtimes
            do_constant_folding=True,  # fold constant ops for smaller/faster graph
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"},
            },
        )
        return True
    except Exception as exc:
        print(f"  ⚠  ONNX export failed: {exc}")
        return False


# ════════════════════════════════════════════════════════════
#  Data loading
# ════════════════════════════════════════════════════════════


def build_dataset(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load all classes, apply sliding window, return (X, y)."""
    data_root = PROCESSED_DATA_ROOT
    window = cfg["data"]["window_size"]
    step = cfg["data"]["step_size"]

    all_X, all_y = [], []

    for class_name in cfg["data"]["fault_classes"]:
        result = load_class_data(data_root, class_name, cfg)
        if result is None:
            print(f"  ⚠  No data for class: {class_name}")
            continue
        feat, labels = result
        X, y = sliding_windows(feat, labels, window, step)
        all_X.append(X)
        all_y.append(y)
        print(f"  ✔  {class_name:25s}  raw={len(feat):7d}  windows={len(X):6d}")

    if not all_X:
        raise RuntimeError(
            "No training data found! Check processed_data_root in config."
        )

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# ════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def per_class_accuracy(model, loader, device, num_classes: int):
    model.eval()
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(1).cpu().numpy()
            labels = y_batch.numpy()
            for c in range(num_classes):
                mask = labels == c
                class_correct[c] += (preds[mask] == c).sum()
                class_total[c] += mask.sum()
    return {
        CLASS_NAMES[c]: (
            float(class_correct[c] / class_total[c]) if class_total[c] > 0 else 0.0
        )
        for c in range(num_classes)
    }


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="PMU Fault Classifier — Training")
    parser.add_argument(
        "--model",
        default=None,
        choices=["TCN", "LSTM", "Transformer", "CNN_LSTM"],
        help="Override model type",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        dest="run_name",
        help=(
            "Name for this training run (saved to logs/checkpoints/<run_name>). "
            "Defaults to current timestamp: YYYY-MM-DD-HH-MM"
        ),
    )
    args = parser.parse_args()

    # ── Resolve run name & output paths ──
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d-%H-%M")
    paths = run_paths(run_name)
    RUN_CKPT_DIR = paths["ckpt_dir"]
    NORMALIZER_NPZ = paths["normalizer_npz"]
    BEST_MODEL_PT = paths["best_model_pt"]
    TRAINING_HISTORY_JSON = paths["training_history_json"]
    CLASS_ACC_JSON = paths["class_acc_json"]
    MODEL_META_JSON = paths["model_meta_json"]
    CONFUSION_MATRIX_JSON = paths["confusion_matrix_json"]
    TSNE_EMBEDDINGS_JSON = paths["tsne_embeddings_json"]
    BEST_MODEL_ONNX = paths["best_model_onnx"]

    config_path = TRAIN_CONFIG_YAML
    cfg = load_config(config_path)

    if args.model:
        cfg["model"]["type"] = args.model

    device = get_device()
    print(f"\n{'='*60}")
    print(f"  PMU Fault Classifier — Training")
    print(f"  Model  : {cfg['model']['type']}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── 1. Load & window data ──
    print("📂  Loading processed data ...")
    X_all, y_all = build_dataset(cfg)
    print(f"\n  Total windows : {len(X_all):,}")
    print(f"  Window shape  : {X_all.shape[1:]}")
    print(f"  Class balance : {np.bincount(y_all).tolist()}\n")

    # ── 2. Normalize ──
    N, T, F = X_all.shape
    normalizer = FeatureNormalizer()
    X_flat = X_all.reshape(-1, F)
    X_flat = normalizer.fit_transform(X_flat)
    X_all = X_flat.reshape(N, T, F)

    cfg["model"]["input_size"] = F
    print(f"  Input features: {F}")

    # Save normalizer
    RUN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    normalizer.save(NORMALIZER_NPZ)
    print(f"  Saved normalizer → {NORMALIZER_NPZ}\n")

    # ── 3. Train/val split ──
    rng = np.random.default_rng(cfg["data"]["random_seed"])
    indices = rng.permutation(len(X_all))
    val_size = int(len(X_all) * cfg["data"]["val_split"])
    val_idx = indices[:val_size]
    trn_idx = indices[val_size:]

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)

    trn_ds = TensorDataset(X_tensor[trn_idx], y_tensor[trn_idx])
    val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

    bs = cfg["training"]["batch_size"]
    trn_loader = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    print(f"  Train samples : {len(trn_ds):,}  |  Val samples : {len(val_ds):,}\n")

    # ── 4. Build model ──
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params  : {n_params:,}\n")

    # Class weights (handle imbalance)
    counts = np.bincount(y_all[trn_idx], minlength=cfg["model"]["num_classes"]).astype(
        float
    )
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    sched_type = cfg["training"]["scheduler"]
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["training"]["step_lr_step"],
            gamma=cfg["training"]["step_lr_gamma"],
        )
    else:
        scheduler = None

    # ── 5. Train ──
    best_val_acc = 0.0
    patience_ctr = 0
    patience_limit = cfg["training"]["early_stopping_patience"]
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Stop-flag: GUI writes this file to request a clean stop at epoch boundary
    STOP_FLAG = RUN_CKPT_DIR / "STOP"
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()

    print(
        f"{'Epoch':>6}  {'Trn Loss':>9}  {'Trn Acc':>8}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>10}"
    )
    print("-" * 60)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        trn_loss, trn_acc = train_one_epoch(
            model, trn_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(trn_loss)
        history["train_acc"].append(trn_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            # Save best checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "cfg": cfg,
            }
            torch.save(ckpt, BEST_MODEL_PT)
            marker = "  ★ best"
        else:
            patience_ctr += 1

        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  {trn_loss:>9.4f}  {trn_acc:>7.2%}  "
            f"{val_loss:>9.4f}  {val_acc:>7.2%}  {cur_lr:>10.2e}"
            f"  {elapsed:.1f}s{marker}"
        )

        if patience_ctr >= patience_limit:
            print(f"\n  ⏹  Early stopping at epoch {epoch}.")
            break

        # Clean stop requested by GUI — finish this epoch then exit
        if STOP_FLAG.exists():
            STOP_FLAG.unlink()
            print(f"\n  ⏹  Stop requested — finishing after epoch {epoch}.")
            break

    # ── 6. Final evaluation ──
    print(f"\n{'='*60}")
    print(f"  Best val accuracy : {best_val_acc:.4%}")

    # Reload best weights
    ckpt = torch.load(BEST_MODEL_PT, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    class_acc = per_class_accuracy(
        model, val_loader, device, cfg["model"]["num_classes"]
    )
    print("\n  Per-class accuracy (validation):")
    for cls, acc in class_acc.items():
        print(f"    {cls:30s}: {acc:.4%}")

    # Save training history & results
    save_json(history, TRAINING_HISTORY_JSON)
    save_json(class_acc, CLASS_ACC_JSON)
    save_json(
        {
            "input_size": F,
            "num_classes": cfg["model"]["num_classes"],
            "model_type": cfg["model"]["type"],
            "window_size": cfg["data"]["window_size"],
        },
        MODEL_META_JSON,
    )

    # ── 7. Save confusion matrix & t-SNE embeddings for analysis tab ──
    print("\n  📊  Computing confusion matrix & embeddings ...")

    all_preds, all_true = collect_predictions(model, val_loader, device)
    save_confusion_matrix(all_true, all_preds, CLASS_NAMES, CONFUSION_MATRIX_JSON)
    save_tsne_embeddings(model, val_loader, device, CLASS_NAMES, TSNE_EMBEDDINGS_JSON)
    print(f"  Saved confusion matrix & embeddings → {RUN_CKPT_DIR}")
    # ── 8. Export ONNX ────────────────────────────────────────
    print("\n  📦  Exporting ONNX model ...")
    onnx_ok = export_onnx(
        model,
        window_size=cfg["data"]["window_size"],
        input_size=F,
        out_path=BEST_MODEL_ONNX,
        device=device,
    )
    if onnx_ok:
        print(f"  Saved ONNX model    \u2192 {BEST_MODEL_ONNX}")
        # Update model_meta with ONNX info
        save_json(
            {
                "input_size": F,
                "num_classes": cfg["model"]["num_classes"],
                "model_type": cfg["model"]["type"],
                "window_size": cfg["data"]["window_size"],
                "onnx_opset": 17,
                "onnx_file": "best_model.onnx",
            },
            MODEL_META_JSON,
        )
    print(f"\n  Checkpoints saved to: {RUN_CKPT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
