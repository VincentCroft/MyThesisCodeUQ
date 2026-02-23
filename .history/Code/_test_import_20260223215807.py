import sys, os
# Make sure Code/ directory is on path (same logic as train.py)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

print("CODE_DIR:", CODE_DIR)
print("sys.path[0]:", sys.path[0])
print("utils/ exists:", os.path.isdir(os.path.join(CODE_DIR, "utils")))
print("utils/__init__.py exists:", os.path.isfile(os.path.join(CODE_DIR, "utils", "__init__.py")))

try:
    from utils.analysis import save_confusion_matrix, collect_predictions, save_tsne_embeddings
    print("✅ utils.analysis imported successfully!")
except Exception as e:
    print("❌ Import failed:", e)
    import traceback; traceback.print_exc()
