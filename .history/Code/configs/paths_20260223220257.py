"""
PMU Fault Classifier — 路径统一配置
File: Code/configs/paths.py

所有需要调整的路径都在这里集中定义。
修改此文件即可更新整个项目的目录结构。
"""

from pathlib import Path

# ── 项目根目录（自动推算，无需手动修改）────────────────────
# Code/configs/paths.py → Code/ → Thesis/
CODE_DIR   : Path = Path(__file__).resolve().parent.parent   # .../Thesis/Code
THESIS_ROOT: Path = CODE_DIR.parent                          # .../Thesis

# ── 数据目录 ────────────────────────────────────────────────
RAW_DATA_ROOT       : Path = THESIS_ROOT / "DataSet"         # 原始采集数据
PROCESSED_DATA_ROOT : Path = THESIS_ROOT / "ProcessedData"   # 预处理后数据

# ── 模型 / 日志 / 检查点 ─────────────────────────────────────
LOGS_DIR  : Path = CODE_DIR / "logs"
CKPT_DIR  : Path = CODE_DIR / "logs" / "checkpoints"

# ── 分析结果文件 ─────────────────────────────────────────────
TRAINING_HISTORY_JSON : Path = CKPT_DIR / "training_history.json"
CLASS_ACC_JSON        : Path = CKPT_DIR / "class_accuracy.json"
MODEL_META_JSON       : Path = CKPT_DIR / "model_meta.json"
NORMALIZER_NPZ        : Path = CKPT_DIR / "normalizer.npz"
BEST_MODEL_PT         : Path = CKPT_DIR / "best_model.pt"
CONFUSION_MATRIX_JSON : Path = CKPT_DIR / "confusion_matrix.json"
TSNE_EMBEDDINGS_JSON  : Path = CKPT_DIR / "tsne_embeddings.json"

# ── 配置文件 ─────────────────────────────────────────────────
TRAIN_CONFIG_YAML : Path = CODE_DIR / "configs" / "train_config.yaml"

# ── 子模块目录 ───────────────────────────────────────────────
MODELS_DIR : Path = CODE_DIR / "models"
UTILS_DIR  : Path = CODE_DIR / "Utils"
GUI_DIR    : Path = CODE_DIR / "GUI"
