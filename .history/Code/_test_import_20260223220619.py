import sys, os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

print("CODE_DIR:", CODE_DIR)
print("Utils/ exists:", os.path.isdir(os.path.join(CODE_DIR, "Utils")))

# 测试1: Utils.analysis 导入
try:
    from Utils.analysis import save_confusion_matrix, collect_predictions, save_tsne_embeddings
    print("✅ Utils.analysis 导入成功!")
except Exception as e:
    print("❌ Utils.analysis 导入失败:", e)

# 测试2: configs.paths 导入
try:
    from configs.paths import (
        CODE_DIR as P_CODE,
        THESIS_ROOT,
        PROCESSED_DATA_ROOT,
        CKPT_DIR,
        BEST_MODEL_PT,
        CONFUSION_MATRIX_JSON,
        TSNE_EMBEDDINGS_JSON,
        TRAIN_CONFIG_YAML,
    )
    print("✅ configs.paths 导入成功!")
    print(f"   PROCESSED_DATA_ROOT = {PROCESSED_DATA_ROOT}")
    print(f"   PROCESSED_DATA_ROOT exists = {PROCESSED_DATA_ROOT.exists()}")
    print(f"   CKPT_DIR = {CKPT_DIR}")
    print(f"   TRAIN_CONFIG_YAML = {TRAIN_CONFIG_YAML}")
    print(f"   TRAIN_CONFIG_YAML exists = {TRAIN_CONFIG_YAML.exists()}")
except Exception as e:
    print("❌ configs.paths 导入失败:", e)
    import traceback; traceback.print_exc()
