"""
PMU Fault Classifier — Centralised Path Configuration
File: Code/configs/paths.py

All file paths are defined here.  Edit this file to relocate the project.
"""

from pathlib import Path

# ── Project roots (auto-derived — no manual edits needed) ───
# Code/configs/paths.py → Code/ → Thesis/
CODE_DIR: Path = Path(__file__).resolve().parent.parent  # .../Thesis/Code
THESIS_ROOT: Path = CODE_DIR.parent  # .../Thesis

# ── Data directories ────────────────────────────────────────
RAW_DATA_ROOT: Path = THESIS_ROOT / "DataSet"
PROCESSED_DATA_ROOT: Path = THESIS_ROOT / "ProcessedData"

# ── Checkpoint base directory ────────────────────────────────
# Each training run is stored in its own sub-folder:
#   CKPT_BASE_DIR / <run_name> /
#     best_model.pt
#     normalizer.npz
#     training_history.json
#     class_accuracy.json
#     model_meta.json
#     confusion_matrix.json
#     tsne_embeddings.json
CKPT_BASE_DIR: Path = CODE_DIR / "logs" / "checkpoints"

# ── Config file ──────────────────────────────────────────────
TRAIN_CONFIG_YAML: Path = CODE_DIR / "configs" / "train_config.yaml"

# ── Sub-module directories ───────────────────────────────────
MODELS_DIR: Path = CODE_DIR / "models"
UTILS_DIR: Path = CODE_DIR / "Utils"
GUI_DIR: Path = CODE_DIR / "GUI"


# ── Helper: resolve paths for a specific run ────────────────
def run_paths(run_name: str) -> dict:
    """Return a dict of all output paths for the given run_name."""
    d = CKPT_BASE_DIR / run_name
    return {
        "ckpt_dir": d,
        "best_model_pt": d / "best_model.pt",
        "best_model_onnx": d / "best_model.onnx",
        "normalizer_npz": d / "normalizer.npz",
        "training_history_json": d / "training_history.json",
        "class_acc_json": d / "class_accuracy.json",
        "model_meta_json": d / "model_meta.json",
        "confusion_matrix_json": d / "confusion_matrix.json",
        "tsne_embeddings_json": d / "tsne_embeddings.json",
    }


# ── Legacy aliases (kept for backward-compatibility) ────────
# These point to a fixed "default" folder so old code still works.
_DEFAULT = CKPT_BASE_DIR / "default"
CKPT_DIR = _DEFAULT
TRAINING_HISTORY_JSON = _DEFAULT / "training_history.json"
CLASS_ACC_JSON = _DEFAULT / "class_accuracy.json"
MODEL_META_JSON = _DEFAULT / "model_meta.json"
NORMALIZER_NPZ = _DEFAULT / "normalizer.npz"
BEST_MODEL_PT = _DEFAULT / "best_model.pt"
CONFUSION_MATRIX_JSON = _DEFAULT / "confusion_matrix.json"
TSNE_EMBEDDINGS_JSON = _DEFAULT / "tsne_embeddings.json"
