"""
PMU Fault Classifier — Streamlit GUI
File: Code/GUI/dashboard.py

Tabs:
  1. 🏠 Home        — Project overview
  2. 🚀 Train       — Launch / monitor training
  3. 🔍 Inference   — Upload CSV → fault prediction
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ── path setup ─────────────────────────────────────────────
GUI_DIR   = Path(__file__).parent.resolve()
CODE_DIR  = GUI_DIR.parent
THESIS    = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

# ── page config (must be FIRST streamlit call) ──────────────
st.set_page_config(
    page_title="PMU Fault Classifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS styling ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── general ── */
html, body, [class*="css"] { font-family: "Inter", sans-serif; }

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.78rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important; font-size: 1.6rem;
}

/* ── section headers ── */
.section-header {
    font-size: 1.1rem; font-weight: 600;
    color: #38bdf8; margin-bottom: 0.3rem;
    border-left: 4px solid #38bdf8; padding-left: 10px;
}

/* ── fault badge ── */
.fault-badge {
    display:inline-block; padding:6px 16px;
    border-radius:9999px; font-weight:700;
    font-size:1rem; margin-bottom:8px;
}
.badge-NORMAL         { background:#064e3b; color:#34d399; }
.badge-SLG_FAULT      { background:#7c2d12; color:#fb923c; }
.badge-LL_FAULT       { background:#713f12; color:#fbbf24; }
.badge-THREE_PHASE_FAULT { background:#4c1d95; color:#c4b5fd; }
.badge-UNKNOWN        { background:#1e293b; color:#94a3b8; }

/* ── probability bar ── */
.prob-bar-wrap { background:#1e293b; border-radius:8px; height:14px;
                 overflow:hidden; margin:4px 0; }
.prob-bar-fill { height:100%; border-radius:8px; transition:width .4s ease; }

/* ── log box ── */
.log-box {
    background:#0f172a; border:1px solid #334155;
    border-radius:8px; padding:12px;
    font-family:monospace; font-size:0.78rem;
    color:#94a3b8; max-height:380px; overflow-y:auto;
    white-space:pre-wrap;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════
CLASS_NAMES  = ["NORMAL", "SLG_FAULT", "LL_FAULT", "THREE_PHASE_FAULT"]
CLASS_COLORS = {
    "NORMAL":            "#34d399",
    "SLG_FAULT":         "#fb923c",
    "LL_FAULT":          "#fbbf24",
    "THREE_PHASE_FAULT": "#c4b5fd",
    "UNKNOWN":           "#94a3b8",
}
CLASS_DESC = {
    "NORMAL":            "正常运行 — Normal operation",
    "SLG_FAULT":         "单相接地故障 — Single line-to-ground (ERROR 201)",
    "LL_FAULT":          "线间故障 — Line-to-line fault (ERROR 202)",
    "THREE_PHASE_FAULT": "三相短路 — Three-phase short circuit (ERROR 204)",
}
CONFIG_PATH   = CODE_DIR / "configs" / "train_config.yaml"
CKPT_DIR      = CODE_DIR / "logs" / "checkpoints"
HISTORY_PATH  = CKPT_DIR / "training_history.json"
BEST_MODEL    = CKPT_DIR / "best_model.pt"
NORMALIZER    = CKPT_DIR / "normalizer.npz"
META_PATH     = CKPT_DIR / "model_meta.json"
PYTHON_EXE    = sys.executable


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_cfg(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def model_exists() -> bool:
    return BEST_MODEL.exists() and NORMALIZER.exists()


def load_history() -> Optional[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return None


def badge_html(cls: str) -> str:
    safe = cls.replace(" ", "_")
    return f'<span class="fault-badge badge-{safe}">{cls}</span>'


def prob_bar_html(prob: float, color: str) -> str:
    pct = f"{prob*100:.1f}%"
    return f"""
    <div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
      <div class="prob-bar-wrap" style="flex:1;">
        <div class="prob-bar-fill" style="width:{pct};background:{color};"></div>
      </div>
      <span style="color:{color};font-weight:600;min-width:52px;text-align:right;">{pct}</span>
    </div>"""


# ═══════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ PMU Fault AI")
    st.markdown("---")
    page = st.radio(
        "导航 / Navigation",
        ["🏠  Home", "🚀  Train", "🔍  Inference"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**模型状态 / Model**")
    if model_exists():
        st.success("✅  已训练  Model ready")
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
            st.caption(f"Type: **{meta.get('model_type','?')}**  |  Classes: **{meta.get('num_classes','?')}**")
    else:
        st.warning("⚠️  未训练  Not trained")
    st.markdown("---")
    st.caption("Protocol v1.2 · PyTorch · Streamlit")


# ═══════════════════════════════════════════════════════════
#  Page: HOME
# ═══════════════════════════════════════════════════════════

if page == "🏠  Home":
    st.title("⚡ PMU 电力故障智能分类系统")
    st.markdown("#### PMU Edge AI Fault Classifier  —  Protocol v1.2")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("支持故障类型", "4 类")
    with col2:
        data_root = THESIS / "ProcessedData"
        n_files   = sum(1 for _ in data_root.rglob("*.csv")) if data_root.exists() else 0
        st.metric("训练数据文件", f"{n_files} 个")
    with col3:
        st.metric("特征维度", "14 维")
    with col4:
        hist = load_history()
        if hist:
            best = max(hist["val_acc"])
            st.metric("最佳验证准确率", f"{best:.2%}")
        else:
            st.metric("最佳验证准确率", "—")

    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-header">🗂  支持故障类型</div>', unsafe_allow_html=True)
        st.markdown("")
        for cls, desc in CLASS_DESC.items():
            color = CLASS_COLORS[cls]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                f'<div style="width:14px;height:14px;border-radius:50%;background:{color};"></div>'
                f'<span style="color:#e2e8f0;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown('<div class="section-header">📐  输入特征说明</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame({
            "特征":  ["DFDT", "FREQ", "IA/IB/IC (Re,Im)", "VA/VB/VC (Re,Im)"],
            "维度":  [1, 1, 6, 6],
            "描述":  [
                "频率变化率 Hz/s",
                "系统频率 Hz",
                "三相电流幅值角→实虚部",
                "三相电压幅值角→实虚部",
            ],
        })
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊  训练历史</div>', unsafe_allow_html=True)
    hist = load_history()
    if hist:
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            epochs = list(range(1, len(hist["val_acc"]) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=hist["train_acc"],
                                     name="Train Acc", line=dict(color="#38bdf8", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=hist["val_acc"],
                                     name="Val Acc", line=dict(color="#34d399", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=hist["train_loss"],
                                     name="Train Loss", line=dict(color="#fb923c", width=2, dash="dot"),
                                     yaxis="y2"))
            fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"],
                                     name="Val Loss", line=dict(color="#fbbf24", width=2, dash="dot"),
                                     yaxis="y2"))
            fig.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title="Accuracy", tickformat=".0%",
                           gridcolor="#1e293b", color="#94a3b8"),
                yaxis2=dict(title="Loss", overlaying="y", side="right",
                            gridcolor="#1e293b", color="#94a3b8"),
                xaxis=dict(title="Epoch", gridcolor="#1e293b", color="#94a3b8"),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart({"Val Acc": hist["val_acc"], "Train Acc": hist["train_acc"]})
    else:
        st.info("尚未训练，请前往 🚀 Train 页面开始训练。")


# ═══════════════════════════════════════════════════════════
#  Page: TRAIN
# ═══════════════════════════════════════════════════════════

elif page == "🚀  Train":
    st.title("🚀 模型训练")
    st.markdown("Configure and launch model training. / 配置并启动模型训练。")
    st.markdown("---")

    cfg = load_cfg()

    # ── Config editor ──
    with st.expander("⚙️  训练配置 / Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            model_type = st.selectbox("模型类型 Model Type",
                                       ["TCN", "LSTM", "Transformer"],
                                       index=["TCN", "LSTM", "Transformer"].index(
                                           cfg["model"]["type"]))
            epochs = st.number_input("训练轮数 Epochs",
                                      min_value=5, max_value=500,
                                      value=cfg["training"]["epochs"])
            batch_size = st.number_input("批次大小 Batch Size",
                                          min_value=16, max_value=1024,
                                          value=cfg["training"]["batch_size"])
        with c2:
            lr = st.number_input("学习率 Learning Rate",
                                  min_value=1e-5, max_value=1e-1,
                                  value=float(cfg["training"]["learning_rate"]),
                                  format="%.5f")
            scheduler = st.selectbox("学习率调度 Scheduler",
                                      ["cosine", "step", "none"],
                                      index=["cosine", "step", "none"].index(
                                          cfg["training"]["scheduler"]))
            patience = st.number_input("早停耐心 Early Stopping",
                                        min_value=3, max_value=50,
                                        value=cfg["training"]["early_stopping_patience"])
        with c3:
            window_size = st.number_input("窗口大小 Window Size",
                                           min_value=16, max_value=512,
                                           value=cfg["data"]["window_size"])
            step_size = st.number_input("滑动步长 Step Size",
                                         min_value=8, max_value=256,
                                         value=cfg["data"]["step_size"])
            val_split = st.slider("验证集比例 Val Split",
                                   min_value=0.1, max_value=0.4,
                                   value=float(cfg["data"]["val_split"]),
                                   step=0.05)

        if st.button("💾  保存配置 Save Config", use_container_width=True):
            cfg["model"]["type"]                          = model_type
            cfg["training"]["epochs"]                     = int(epochs)
            cfg["training"]["batch_size"]                 = int(batch_size)
            cfg["training"]["learning_rate"]              = float(lr)
            cfg["training"]["scheduler"]                  = scheduler
            cfg["training"]["early_stopping_patience"]    = int(patience)
            cfg["data"]["window_size"]                    = int(window_size)
            cfg["data"]["step_size"]                      = int(step_size)
            cfg["data"]["val_split"]                      = float(val_split)
            save_cfg(cfg)
            st.success("✅  配置已保存！")

    st.markdown("---")

    # ── Launch training ──
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        start_btn = st.button("▶️  开始训练 Start Training",
                               use_container_width=True, type="primary")
    with col_info:
        if model_exists():
            st.info("ℹ️  已存在训练模型，继续训练将覆盖最优模型。")

    log_placeholder    = st.empty()
    metric_placeholder = st.empty()

    if start_btn:
        train_script = CODE_DIR / "train.py"
        cmd = [PYTHON_EXE, str(train_script),
               "--config", str(CONFIG_PATH),
               "--model", cfg["model"]["type"]]

        log_lines: list[str] = []

        def stream_process():
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(CODE_DIR)
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
            proc.wait()
            log_lines.append(f"\n  Process exited with code {proc.returncode}")

        thread = threading.Thread(target=stream_process, daemon=True)
        thread.start()

        progress_bar = st.progress(0, text="Training in progress…")
        start_t      = time.time()
        epoch_seen   = 0
        max_epoch    = int(epochs)

        while thread.is_alive() or epoch_seen < max_epoch:
            time.sleep(0.5)
            # Update log
            text = "\n".join(log_lines[-60:])
            log_placeholder.markdown(
                f'<div class="log-box">{text}</div>', unsafe_allow_html=True
            )
            # Parse epoch progress
            for line in log_lines:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        ep = int(parts[0])
                        if ep > epoch_seen:
                            epoch_seen = ep
                            progress_bar.progress(
                                min(epoch_seen / max_epoch, 1.0),
                                text=f"Epoch {epoch_seen}/{max_epoch}"
                            )
                except Exception:
                    pass
            if not thread.is_alive():
                break

        thread.join()
        progress_bar.progress(1.0, text="✅  Training complete!")
        text = "\n".join(log_lines)
        log_placeholder.markdown(
            f'<div class="log-box">{text}</div>', unsafe_allow_html=True
        )

        # Show metrics
        hist = load_history()
        if hist:
            best_val = max(hist["val_acc"])
            m1, m2, m3 = metric_placeholder.columns(3)
            m1.metric("最佳验证准确率", f"{best_val:.2%}")
            m2.metric("训练轮数", f"{len(hist['val_acc'])}")
            m3.metric("最终验证Loss", f"{hist['val_loss'][-1]:.4f}")
        st.balloons()


# ═══════════════════════════════════════════════════════════
#  Page: INFERENCE
# ═══════════════════════════════════════════════════════════

elif page == "🔍  Inference":
    st.title("🔍 故障推理 / Fault Inference")
    st.markdown("上传符合协议格式的 CSV 文件，系统将分析可能的故障类型及概率。")
    st.markdown("Upload a protocol-format CSV to get fault classification results.")
    st.markdown("---")

    if not model_exists():
        st.error("❌  未找到已训练模型！请先在 🚀 Train 页面完成训练。")
        st.stop()

    # ── Lazy load inference engine ──
    @st.cache_resource
    def load_inference_engine():
        import torch
        from models.feature_engineering import FeatureNormalizer, load_inference_csv
        from models.classifier import build_model

        meta = json.loads(META_PATH.read_text())
        ckpt = torch.load(BEST_MODEL, map_location="cpu")

        cfg = ckpt["cfg"]
        cfg["model"]["input_size"] = meta["input_size"]

        model = build_model(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        normalizer = FeatureNormalizer.load(NORMALIZER)
        return model, normalizer, cfg

    try:
        model, normalizer, inf_cfg = load_inference_engine()
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        st.stop()

    # ── File uploader ──
    uploaded = st.file_uploader(
        "📂  选择 CSV 文件 / Choose CSV file",
        type=["csv"],
        help="文件需包含协议规定的 16 列（含 TIMESTAMP 和 ERROR_CODE）",
    )

    if uploaded is not None:
        import torch
        import torch.nn.functional as F_nn
        from models.feature_engineering import (
            FeatureNormalizer, CLASS_NAMES, CLASS_CODES,
            build_feature_matrix, impute_nan, sliding_windows,
        )

        # ── 1. Read & preview ──
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        try:
            df_raw = pd.read_csv(tmp_path, keep_default_na=True)
        except Exception as e:
            st.error(f"读取 CSV 失败: {e}")
            st.stop()

        st.markdown('<div class="section-header">📋  数据预览 Data Preview</div>',
                    unsafe_allow_html=True)
        st.dataframe(df_raw.head(10), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总行数 Rows", f"{len(df_raw):,}")
        c2.metric("列数 Columns", len(df_raw.columns))
        nan_pct = df_raw.iloc[:, :14].isna().mean().mean()
        c3.metric("NaN 比例", f"{nan_pct:.1%}")

        if "ERROR_CODE" in df_raw.columns:
            usable = (df_raw["ERROR_CODE"] < 300).sum()
            c4.metric("有效行数 Usable", f"{usable:,}")

        st.markdown("---")

        # ── 2. Feature extraction & inference ──
        with st.spinner("🔄  正在推理 Inferring..."):
            df_infer = df_raw.copy()
            if "ERROR_CODE" in df_infer.columns:
                df_infer = df_infer[df_infer["ERROR_CODE"] < 300].reset_index(drop=True)

            if len(df_infer) == 0:
                st.error("❌  过滤 UNUSABLE 行后没有可用数据。")
                st.stop()

            feat       = build_feature_matrix(df_infer, inf_cfg)
            feat       = impute_nan(feat)
            feat_norm  = normalizer.transform(feat)

            window     = inf_cfg["data"]["window_size"]
            step       = inf_cfg["data"]["step_size"]
            dummy_lbls = np.zeros(len(feat_norm), dtype=np.int64)

            X_wins, _  = sliding_windows(feat_norm, dummy_lbls, window, step)

            if len(X_wins) == 0:
                # Not enough rows for a window — use all rows as single window (padded)
                pad_len = window - len(feat_norm)
                feat_pad = np.pad(feat_norm, ((pad_len, 0), (0, 0)), mode="edge")
                X_wins = feat_pad[np.newaxis, :, :]

            X_tensor = torch.tensor(X_wins, dtype=torch.float32)

            all_probs = []
            bs = 256
            with torch.no_grad():
                for i in range(0, len(X_tensor), bs):
                    logits = model(X_tensor[i:i+bs])
                    probs  = F_nn.softmax(logits, dim=-1)
                    all_probs.append(probs.numpy())

            all_probs    = np.concatenate(all_probs, axis=0)   # [N_win, 4]
            mean_probs   = all_probs.mean(axis=0)               # [4]
            pred_label   = int(np.argmax(mean_probs))
            pred_class   = CLASS_NAMES[pred_label]
            pred_conf    = float(mean_probs[pred_label])

        # ── 3. Results ──
        st.markdown('<div class="section-header">🎯  推理结果 Inference Results</div>',
                    unsafe_allow_html=True)
        st.markdown("")

        res_col, detail_col = st.columns([1, 1])

        with res_col:
            st.markdown(f"**主要预测 Primary Prediction**")
            st.markdown(badge_html(pred_class), unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:#94a3b8;margin:4px 0;">'
                f'{CLASS_DESC.get(pred_class, "")}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color:#38bdf8;font-size:2rem;font-weight:700;margin:0;">'
                f'{pred_conf:.1%}</p>',
                unsafe_allow_html=True,
            )
            st.caption(f"分析窗口数 Windows analyzed: {len(X_wins)}")

        with detail_col:
            st.markdown("**各类别概率 Class Probabilities**")
            for i, cls in enumerate(CLASS_NAMES):
                color = CLASS_COLORS[cls]
                prob  = float(mean_probs[i])
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
                    f'<span style="color:{color};font-weight:600;min-width:160px;">{cls}</span>'
                    f'{prob_bar_html(prob, color)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── 4. Per-window distribution (Plotly) ──
        st.markdown('<div class="section-header">📈  逐窗口预测分布 Per-Window Distribution</div>',
                    unsafe_allow_html=True)

        try:
            import plotly.graph_objects as go

            fig2 = go.Figure()
            win_idx = list(range(len(all_probs)))
            for i, cls in enumerate(CLASS_NAMES):
                fig2.add_trace(go.Scatter(
                    x=win_idx, y=all_probs[:, i],
                    mode="lines", name=cls,
                    line=dict(color=list(CLASS_COLORS.values())[i], width=2),
                    fill="tozeroy", opacity=0.3,
                ))
            fig2.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                xaxis=dict(title="Window Index", gridcolor="#1e293b"),
                yaxis=dict(title="Probability", tickformat=".0%",
                           gridcolor="#1e293b", range=[0, 1]),
                legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
                margin=dict(l=10, r=10, t=20, b=10),
                height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            prob_df = pd.DataFrame(all_probs, columns=CLASS_NAMES)
            st.line_chart(prob_df)

        # ── 5. Summary table ──
        st.markdown("---")
        st.markdown('<div class="section-header">📊  概率汇总 Probability Summary</div>',
                    unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "故障类型 Fault Class": CLASS_NAMES,
            "平均概率 Mean Prob":   [f"{p:.2%}" for p in mean_probs],
            "最大概率 Max Prob":    [f"{p:.2%}" for p in all_probs.max(axis=0)],
            "最小概率 Min Prob":    [f"{p:.2%}" for p in all_probs.min(axis=0)],
            "描述 Description":     [CLASS_DESC[c] for c in CLASS_NAMES],
        }).sort_values("平均概率 Mean Prob", ascending=False)

        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # ── 6. Download results ──
        result_dict = {
            "file": uploaded.name,
            "prediction": pred_class,
            "confidence": f"{pred_conf:.4f}",
            "windows_analyzed": int(len(X_wins)),
            "class_probabilities": {
                cls: f"{float(mean_probs[i]):.6f}"
                for i, cls in enumerate(CLASS_NAMES)
            },
        }
        st.download_button(
            label="⬇️  下载推理报告 Download Report (JSON)",
            data=json.dumps(result_dict, indent=2, ensure_ascii=False),
            file_name=f"pmu_fault_report_{uploaded.name}.json",
            mime="application/json",
        )
