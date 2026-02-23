"""
PMU Fault Classifier — Streamlit GUI
File: Code/GUI/dashboard.py

Pages:
  1. Home       — Project overview & training history
  2. Train      — Configure and launch training
  3. Analysis   — Fit curves, confusion matrix, report, t-SNE
  4. Inference  — Upload CSV and get fault prediction
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

# ── Path setup ──────────────────────────────────────────────
GUI_DIR = Path(__file__).parent.resolve()
CODE_DIR = GUI_DIR.parent
THESIS = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

# ── Centralised path config (edit configs/paths.py to relocate) ──
from configs.paths import (
    PROCESSED_DATA_ROOT,
    CKPT_BASE_DIR,
    TRAIN_CONFIG_YAML as CONFIG_PATH,
    run_paths,
)

# ── Page config (must be the FIRST Streamlit call) ──────────
st.set_page_config(
    page_title="PMU Fault Classifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: "Inter", sans-serif; }

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label {
    color: #94a3b8 !important; font-size: 0.78rem;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important; font-size: 1.6rem;
}

.section-header {
    font-size: 1.1rem; font-weight: 600;
    color: #38bdf8; margin-bottom: 0.3rem;
    border-left: 4px solid #38bdf8; padding-left: 10px;
}

.fault-badge {
    display: inline-block; padding: 6px 16px;
    border-radius: 9999px; font-weight: 700;
    font-size: 1rem; margin-bottom: 8px;
}
.badge-NORMAL            { background: #064e3b; color: #34d399; }
.badge-SLG_FAULT         { background: #7c2d12; color: #fb923c; }
.badge-LL_FAULT          { background: #713f12; color: #fbbf24; }
.badge-THREE_PHASE_FAULT { background: #4c1d95; color: #c4b5fd; }
.badge-UNKNOWN           { background: #1e293b; color: #94a3b8; }

.prob-bar-wrap { background: #1e293b; border-radius: 8px; height: 14px;
                 overflow: hidden; margin: 4px 0; }
.prob-bar-fill { height: 100%; border-radius: 8px; transition: width .4s ease; }

.log-box {
    background: #0f172a; border: 1px solid #334155;
    border-radius: 8px; padding: 12px;
    font-family: monospace; font-size: 0.78rem;
    color: #94a3b8; max-height: 380px; overflow-y: auto;
    white-space: pre-wrap;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════
CLASS_NAMES = ["NORMAL", "SLG_FAULT", "LL_FAULT", "THREE_PHASE_FAULT"]
CLASS_COLORS = {
    "NORMAL": "#34d399",
    "SLG_FAULT": "#fb923c",
    "LL_FAULT": "#fbbf24",
    "THREE_PHASE_FAULT": "#c4b5fd",
    "UNKNOWN": "#94a3b8",
}
CLASS_DESC = {
    "NORMAL": "Normal operation  (ERROR 000)",
    "SLG_FAULT": "Single line-to-ground fault  (ERROR 201)",
    "LL_FAULT": "Line-to-line fault  (ERROR 202)",
    "THREE_PHASE_FAULT": "Three-phase short-circuit  (ERROR 204)",
}
PYTHON_EXE = sys.executable


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════


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
    pct = f"{prob * 100:.1f}%"
    return (
        f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;">'
        f'  <div class="prob-bar-wrap" style="flex:1;">'
        f'    <div class="prob-bar-fill" style="width:{pct};background:{color};"></div>'
        f"  </div>"
        f'  <span style="color:{color};font-weight:600;min-width:52px;text-align:right;">{pct}</span>'
        f"</div>"
    )


# ════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ PMU Fault AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Home", "🚀  Train", "📈  Analysis", "🔍  Inference"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model Status**")
    if model_exists():
        st.success("✅  Model ready")
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
            st.caption(
                f"Type: **{meta.get('model_type', '?')}**  |  "
                f"Classes: **{meta.get('num_classes', '?')}**"
            )
    else:
        st.warning("⚠️  Not trained yet")
    st.markdown("---")
    st.caption("Protocol v1.2  ·  PyTorch  ·  Streamlit")


# ════════════════════════════════════════════════════════════
#  Page: HOME
# ════════════════════════════════════════════════════════════

if page == "🏠  Home":
    st.title("⚡ PMU Fault Classifier")
    st.markdown("#### PMU Edge AI Fault Classifier  —  Protocol v1.2")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fault Classes", "4")
    with col2:
        n_files = (
            sum(1 for _ in PROCESSED_DATA_ROOT.rglob("*.csv"))
            if PROCESSED_DATA_ROOT.exists()
            else 0
        )
        st.metric("Training Files", f"{n_files}")
    with col3:
        st.metric("Feature Dimensions", "14")
    with col4:
        hist = load_history()
        if hist:
            best = max(hist["val_acc"])
            st.metric("Best Val Accuracy", f"{best:.2%}")
        else:
            st.metric("Best Val Accuracy", "—")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            '<div class="section-header">Fault Classes</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        for cls, desc in CLASS_DESC.items():
            color = CLASS_COLORS[cls]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                f'<div style="width:14px;height:14px;border-radius:50%;background:{color};"></div>'
                f'<span style="color:#e2e8f0;">{desc}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown(
            '<div class="section-header">Input Features</div>',
            unsafe_allow_html=True,
        )
        feat_df = pd.DataFrame(
            {
                "Feature": ["DFDT", "FREQ", "IA/IB/IC (Re, Im)", "VA/VB/VC (Re, Im)"],
                "Dim": [1, 1, 6, 6],
                "Description": [
                    "Rate of change of frequency  (Hz/s)",
                    "System frequency  (Hz)",
                    "3-phase current phasor → rectangular",
                    "3-phase voltage phasor → rectangular",
                ],
            }
        )
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="section-header">Training History</div>',
        unsafe_allow_html=True,
    )
    hist = load_history()
    if hist:
        try:
            import plotly.graph_objects as go

            epochs = list(range(1, len(hist["val_acc"]) + 1))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["train_acc"],
                    name="Train Acc",
                    line=dict(color="#38bdf8", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["val_acc"],
                    name="Val Acc",
                    line=dict(color="#34d399", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["train_loss"],
                    name="Train Loss",
                    line=dict(color="#fb923c", width=2, dash="dot"),
                    yaxis="y2",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["val_loss"],
                    name="Val Loss",
                    line=dict(color="#fbbf24", width=2, dash="dot"),
                    yaxis="y2",
                )
            )
            fig.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(
                    title="Accuracy",
                    tickformat=".0%",
                    gridcolor="#1e293b",
                    color="#94a3b8",
                ),
                yaxis2=dict(
                    title="Loss",
                    overlaying="y",
                    side="right",
                    gridcolor="#1e293b",
                    color="#94a3b8",
                ),
                xaxis=dict(title="Epoch", gridcolor="#1e293b", color="#94a3b8"),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart({"Val Acc": hist["val_acc"], "Train Acc": hist["train_acc"]})
    else:
        st.info("No training history found. Go to the Train page to start training.")


# ════════════════════════════════════════════════════════════
#  Page: TRAIN
# ════════════════════════════════════════════════════════════

elif page == "🚀  Train":
    st.title("🚀 Model Training")
    st.markdown("Configure hyperparameters and launch training.")
    st.markdown("---")

    cfg = load_cfg()

    with st.expander("⚙️  Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            model_type = st.selectbox(
                "Model Type",
                ["TCN", "LSTM", "Transformer"],
                index=["TCN", "LSTM", "Transformer"].index(cfg["model"]["type"]),
            )
            epochs = st.number_input(
                "Epochs",
                min_value=5,
                max_value=500,
                value=cfg["training"]["epochs"],
            )
            batch_size = st.number_input(
                "Batch Size",
                min_value=16,
                max_value=1024,
                value=cfg["training"]["batch_size"],
            )

        with c2:
            lr = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-1,
                value=float(cfg["training"]["learning_rate"]),
                format="%.5f",
            )
            scheduler = st.selectbox(
                "LR Scheduler",
                ["cosine", "step", "none"],
                index=["cosine", "step", "none"].index(cfg["training"]["scheduler"]),
            )
            patience = st.number_input(
                "Early Stopping Patience",
                min_value=3,
                max_value=50,
                value=cfg["training"]["early_stopping_patience"],
            )

        with c3:
            window_size = st.number_input(
                "Window Size",
                min_value=16,
                max_value=512,
                value=cfg["data"]["window_size"],
            )
            step_size = st.number_input(
                "Step Size",
                min_value=8,
                max_value=256,
                value=cfg["data"]["step_size"],
            )
            val_split = st.slider(
                "Validation Split",
                min_value=0.1,
                max_value=0.4,
                value=float(cfg["data"]["val_split"]),
                step=0.05,
            )

        if st.button("💾  Save Config", use_container_width=True):
            cfg["model"]["type"] = model_type
            cfg["training"]["epochs"] = int(epochs)
            cfg["training"]["batch_size"] = int(batch_size)
            cfg["training"]["learning_rate"] = float(lr)
            cfg["training"]["scheduler"] = scheduler
            cfg["training"]["early_stopping_patience"] = int(patience)
            cfg["data"]["window_size"] = int(window_size)
            cfg["data"]["step_size"] = int(step_size)
            cfg["data"]["val_split"] = float(val_split)
            save_cfg(cfg)
            st.success("✅  Configuration saved.")

    st.markdown("---")

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        start_btn = st.button(
            "▶️  Start Training", use_container_width=True, type="primary"
        )
    with col_info:
        if model_exists():
            st.info("ℹ️  A trained model already exists. Training will overwrite it.")

    log_placeholder = st.empty()
    metric_placeholder = st.empty()

    if start_btn:
        train_script = CODE_DIR / "train.py"
        cmd = [
            PYTHON_EXE,
            str(train_script),
            "--config",
            str(CONFIG_PATH),
            "--model",
            cfg["model"]["type"],
        ]

        log_lines: list[str] = []

        def stream_process():
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(CODE_DIR),
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
            proc.wait()
            log_lines.append(f"\n  Process exited with code {proc.returncode}")

        thread = threading.Thread(target=stream_process, daemon=True)
        thread.start()

        progress_bar = st.progress(0, text="Training in progress…")
        epoch_seen = 0
        max_epoch = int(epochs)

        while thread.is_alive() or epoch_seen < max_epoch:
            time.sleep(0.5)
            text = "\n".join(log_lines[-60:])
            log_placeholder.markdown(
                f'<div class="log-box">{text}</div>', unsafe_allow_html=True
            )
            for line in log_lines:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        ep = int(parts[0])
                        if ep > epoch_seen:
                            epoch_seen = ep
                            progress_bar.progress(
                                min(epoch_seen / max_epoch, 1.0),
                                text=f"Epoch {epoch_seen} / {max_epoch}",
                            )
                except Exception:
                    pass
            if not thread.is_alive():
                break

        thread.join()
        progress_bar.progress(1.0, text="✅  Training complete!")
        log_placeholder.markdown(
            f'<div class="log-box">{"".join(log_lines)}</div>',
            unsafe_allow_html=True,
        )

        hist = load_history()
        if hist:
            best_val = max(hist["val_acc"])
            m1, m2, m3 = metric_placeholder.columns(3)
            m1.metric("Best Val Accuracy", f"{best_val:.2%}")
            m2.metric("Total Epochs", f"{len(hist['val_acc'])}")
            m3.metric("Final Val Loss", f"{hist['val_loss'][-1]:.4f}")
        st.balloons()


# ════════════════════════════════════════════════════════════
#  Page: ANALYSIS
# ════════════════════════════════════════════════════════════

elif page == "📈  Analysis":
    st.title("📈 Model Analysis")
    st.markdown(
        "Inspect fit curves, confusion matrix, classification report and "
        "t-SNE feature clusters after training."
    )
    st.markdown("---")

    if not model_exists():
        st.error(
            "No trained model found. Please complete training on the Train page first."
        )
        st.stop()

    try:
        import plotly.graph_objects as go
        import plotly.figure_factory as ff
        import plotly.express as px

        _PLOTLY = True
    except ImportError:
        _PLOTLY = False
        st.warning("plotly is not installed — charts will be shown in simplified form.")

    tab_curve, tab_cm, tab_report, tab_tsne = st.tabs(
        [
            "📉  Loss & Accuracy",
            "🔲  Confusion Matrix",
            "📋  Classification Report",
            "🔵  Feature Cluster (t-SNE)",
        ]
    )

    # ── Tab 1: Loss & Accuracy curves ──────────────────────
    with tab_curve:
        st.markdown(
            '<div class="section-header">Training / Validation Fit Curves</div>',
            unsafe_allow_html=True,
        )
        hist = load_history()
        if not hist:
            st.info("No training history available. Train the model first.")
        else:
            epochs = list(range(1, len(hist["val_acc"]) + 1))
            best_ep = int(np.argmax(hist["val_acc"])) + 1
            best_acc = max(hist["val_acc"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best Val Accuracy", f"{best_acc:.2%}")
            m2.metric("Best Epoch", best_ep)
            m3.metric("Final Train Loss", f"{hist['train_loss'][-1]:.4f}")
            m4.metric("Final Val Loss", f"{hist['val_loss'][-1]:.4f}")
            st.markdown("")

            if _PLOTLY:
                fig_acc = go.Figure()
                fig_acc.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=[v * 100 for v in hist["train_acc"]],
                        name="Train Acc (%)",
                        mode="lines+markers",
                        line=dict(color="#38bdf8", width=2),
                        marker=dict(size=4),
                    )
                )
                fig_acc.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=[v * 100 for v in hist["val_acc"]],
                        name="Val Acc (%)",
                        mode="lines+markers",
                        line=dict(color="#34d399", width=2),
                        marker=dict(size=4),
                    )
                )
                fig_acc.add_vline(
                    x=best_ep,
                    line_dash="dot",
                    line_color="#fbbf24",
                    annotation_text=f"Best epoch {best_ep}",
                    annotation_font_color="#fbbf24",
                )
                fig_acc.update_layout(
                    title="Accuracy Curve",
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(title="Epoch", gridcolor="#1e293b"),
                    yaxis=dict(title="Accuracy (%)", gridcolor="#1e293b"),
                    legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=300,
                )

                fig_loss = go.Figure()
                fig_loss.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=hist["train_loss"],
                        name="Train Loss",
                        mode="lines+markers",
                        line=dict(color="#fb923c", width=2),
                        marker=dict(size=4),
                    )
                )
                fig_loss.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=hist["val_loss"],
                        name="Val Loss",
                        mode="lines+markers",
                        line=dict(color="#f472b6", width=2, dash="dot"),
                        marker=dict(size=4),
                    )
                )
                fig_loss.add_vline(x=best_ep, line_dash="dot", line_color="#fbbf24")
                fig_loss.update_layout(
                    title="Loss Curve",
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(title="Epoch", gridcolor="#1e293b"),
                    yaxis=dict(title="Loss", gridcolor="#1e293b"),
                    legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=300,
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(fig_acc, use_container_width=True)
                with col_b:
                    st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.line_chart(
                    {"Train Acc": hist["train_acc"], "Val Acc": hist["val_acc"]}
                )
                st.line_chart(
                    {"Train Loss": hist["train_loss"], "Val Loss": hist["val_loss"]}
                )

            st.markdown("---")
            st.markdown(
                '<div class="section-header">Overfitting Analysis</div>',
                unsafe_allow_html=True,
            )
            gap = [
                round((ta - va) * 100, 2)
                for ta, va in zip(hist["train_acc"], hist["val_acc"])
            ]
            if _PLOTLY:
                fig_gap = go.Figure()
                fig_gap.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=gap,
                        name="Train − Val Gap (%)",
                        fill="tozeroy",
                        line=dict(color="#f472b6", width=2),
                    )
                )
                fig_gap.add_hline(
                    y=5,
                    line_dash="dot",
                    line_color="#fbbf24",
                    annotation_text="5 % threshold",
                )
                fig_gap.add_hline(
                    y=10,
                    line_dash="dot",
                    line_color="#ef4444",
                    annotation_text="10 % threshold",
                )
                fig_gap.update_layout(
                    title="Train − Val Accuracy Gap (overfit indicator)",
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(title="Epoch", gridcolor="#1e293b"),
                    yaxis=dict(title="Gap (%)", gridcolor="#1e293b"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=250,
                )
                st.plotly_chart(fig_gap, use_container_width=True)

            final_gap = gap[-1]
            if final_gap < 3:
                st.success(f"✅  Final Train−Val Gap = {final_gap:.2f} % — good fit.")
            elif final_gap < 8:
                st.warning(
                    f"⚠️  Final Train−Val Gap = {final_gap:.2f} % — "
                    "mild overfitting. Consider increasing dropout."
                )
            else:
                st.error(
                    f"❌  Final Train−Val Gap = {final_gap:.2f} % — "
                    "significant overfitting. Increase regularisation or reduce model complexity."
                )

    # ── Tab 2: Confusion Matrix ─────────────────────────────
    with tab_cm:
        st.markdown(
            '<div class="section-header">Confusion Matrix</div>',
            unsafe_allow_html=True,
        )
        if not CM_PATH.exists():
            st.info(
                "Confusion matrix data not yet generated. "
                "Re-run training — it is saved automatically at the end."
            )
        else:
            cm_data = json.loads(CM_PATH.read_text())
            cls_names = cm_data["class_names"]
            matrix = np.array(cm_data["matrix"])
            matrix_norm = np.array(cm_data["matrix_norm"])
            oa = cm_data["overall_accuracy"]

            st.metric("Overall Accuracy", f"{oa:.2%}")
            st.markdown("")

            view_mode = st.radio(
                "Display mode",
                ["Raw Counts", "Normalised (Recall)"],
                horizontal=True,
            )
            use_norm = view_mode == "Normalised (Recall)"
            disp_matrix = matrix_norm if use_norm else matrix
            fmt_vals = [
                [f"{v:.2f}" if use_norm else str(int(v)) for v in row]
                for row in disp_matrix
            ]

            if _PLOTLY:
                colorscale = [
                    [0.0, "#0f172a"],
                    [0.3, "#1e3a5f"],
                    [0.6, "#1d4ed8"],
                    [1.0, "#38bdf8"],
                ]
                fig_cm = ff.create_annotated_heatmap(
                    z=disp_matrix.tolist(),
                    x=cls_names,
                    y=cls_names,
                    annotation_text=fmt_vals,
                    colorscale=colorscale,
                    showscale=True,
                )
                fig_cm.update_layout(
                    title="Confusion Matrix"
                    + (" (Normalised)" if use_norm else " (Raw Counts)"),
                    paper_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(title="Predicted", side="bottom"),
                    yaxis=dict(title="True", autorange="reversed"),
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=420,
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                df_cm = pd.DataFrame(disp_matrix, index=cls_names, columns=cls_names)
                st.dataframe(df_cm, use_container_width=True)

            st.markdown("---")
            st.markdown(
                '<div class="section-header">Per-Class Metrics</div>',
                unsafe_allow_html=True,
            )
            rows = []
            for cls, m in cm_data["per_class"].items():
                rows.append(
                    {
                        "Class": cls,
                        "Precision": f"{m['precision']:.4f}",
                        "Recall": f"{m['recall']:.4f}",
                        "F1-Score": f"{m['f1']:.4f}",
                        "Support": m["support"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Tab 3: Classification Report ───────────────────────
    with tab_report:
        st.markdown(
            '<div class="section-header">Classification Report</div>',
            unsafe_allow_html=True,
        )
        if not CM_PATH.exists():
            st.info(
                "Classification report data not yet generated. Complete training first."
            )
        else:
            cm_data = json.loads(CM_PATH.read_text())
            cls_names = cm_data["class_names"]
            oa = cm_data["overall_accuracy"]

            if _PLOTLY:
                per = cm_data["per_class"]
                metrics_radar = ["Precision", "Recall", "F1-Score"]
                radar_fig = go.Figure()
                for cls in cls_names:
                    vals = [
                        per[cls]["precision"],
                        per[cls]["recall"],
                        per[cls]["f1"],
                        per[cls]["precision"],
                    ]
                    radar_fig.add_trace(
                        go.Scatterpolar(
                            r=vals,
                            theta=metrics_radar + [metrics_radar[0]],
                            fill="toself",
                            name=cls,
                            line=dict(color=CLASS_COLORS.get(cls, "#94a3b8"), width=2),
                            opacity=0.7,
                        )
                    )
                radar_fig.update_layout(
                    polar=dict(
                        bgcolor="#1e293b",
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            gridcolor="#334155",
                            color="#94a3b8",
                        ),
                        angularaxis=dict(gridcolor="#334155", color="#e2e8f0"),
                    ),
                    paper_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
                    title="Per-Class Precision / Recall / F1 Radar",
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=380,
                )
                st.plotly_chart(radar_fig, use_container_width=True)

                f1_vals = [per[c]["f1"] for c in cls_names]
                colors = [CLASS_COLORS.get(c, "#94a3b8") for c in cls_names]
                fig_f1 = go.Figure()
                fig_f1.add_trace(
                    go.Bar(
                        x=cls_names,
                        y=f1_vals,
                        marker_color=colors,
                        text=[f"{v:.3f}" for v in f1_vals],
                        textposition="outside",
                    )
                )
                fig_f1.update_layout(
                    title="F1-Score per Class",
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(range=[0, 1.1], gridcolor="#1e293b"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=280,
                )
                st.plotly_chart(fig_f1, use_container_width=True)

            rows = []
            for cls, m in cm_data["per_class"].items():
                rows.append(
                    {
                        "Class": cls,
                        "Precision": m["precision"],
                        "Recall": m["recall"],
                        "F1-Score": m["f1"],
                        "Support": m["support"],
                    }
                )
            rows.append(
                {
                    "Class": "macro avg",
                    "Precision": round(
                        np.mean(
                            [cm_data["per_class"][c]["precision"] for c in cls_names]
                        ),
                        4,
                    ),
                    "Recall": round(
                        np.mean([cm_data["per_class"][c]["recall"] for c in cls_names]),
                        4,
                    ),
                    "F1-Score": round(
                        np.mean([cm_data["per_class"][c]["f1"] for c in cls_names]), 4
                    ),
                    "Support": sum(
                        cm_data["per_class"][c]["support"] for c in cls_names
                    ),
                }
            )
            rows.append(
                {
                    "Class": "overall accuracy",
                    "Precision": "—",
                    "Recall": "—",
                    "F1-Score": oa,
                    "Support": sum(
                        cm_data["per_class"][c]["support"] for c in cls_names
                    ),
                }
            )
            st.markdown("---")
            st.markdown(
                '<div class="section-header">Metrics Summary</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Tab 4: t-SNE Feature Cluster ───────────────────────
    with tab_tsne:
        st.markdown(
            '<div class="section-header">Feature Cluster (t-SNE)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "t-SNE projects the penultimate-layer embeddings onto 2D — "
            "well-separated clusters indicate good class discriminability."
        )
        if not TSNE_PATH.exists():
            st.info(
                "t-SNE embedding data not yet generated. "
                "Re-run training — it is saved automatically at the end."
            )
        else:
            tsne_data = json.loads(TSNE_PATH.read_text())
            cls_names = tsne_data["class_names"]
            df_tsne = pd.DataFrame(
                {
                    "x": tsne_data["x"],
                    "y": tsne_data["y"],
                    "label_name": tsne_data["label_names"],
                    "label": tsne_data["labels"],
                }
            )

            st.metric("Visualised Samples", f"{len(df_tsne):,}")

            if _PLOTLY:
                fig_tsne = go.Figure()
                for cls in cls_names:
                    sub = df_tsne[df_tsne["label_name"] == cls]
                    fig_tsne.add_trace(
                        go.Scatter(
                            x=sub["x"],
                            y=sub["y"],
                            mode="markers",
                            name=cls,
                            marker=dict(
                                color=CLASS_COLORS.get(cls, "#94a3b8"),
                                size=5,
                                opacity=0.75,
                                line=dict(width=0),
                            ),
                        )
                    )
                fig_tsne.update_layout(
                    title="t-SNE Feature Cluster (validation set)",
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(
                        title="t-SNE Dim 1",
                        gridcolor="#1e293b",
                        zeroline=False,
                        showticklabels=False,
                    ),
                    yaxis=dict(
                        title="t-SNE Dim 2",
                        gridcolor="#1e293b",
                        zeroline=False,
                        showticklabels=False,
                    ),
                    legend=dict(
                        bgcolor="#1e293b", bordercolor="#334155", itemsizing="constant"
                    ),
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=520,
                )
                st.plotly_chart(fig_tsne, use_container_width=True)

                st.markdown("---")
                st.markdown(
                    '<div class="section-header">Class Distribution in Embedded Space</div>',
                    unsafe_allow_html=True,
                )
                dist = df_tsne["label_name"].value_counts().reset_index()
                dist.columns = ["Class", "Count"]
                fig_dist = go.Figure(
                    go.Bar(
                        x=dist["Class"],
                        y=dist["Count"],
                        marker_color=[
                            CLASS_COLORS.get(c, "#94a3b8") for c in dist["Class"]
                        ],
                        text=dist["Count"],
                        textposition="outside",
                    )
                )
                fig_dist.update_layout(
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b"),
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=240,
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.scatter_chart(
                    df_tsne[["x", "y", "label_name"]], x="x", y="y", color="label_name"
                )


# ════════════════════════════════════════════════════════════
#  Page: INFERENCE
# ════════════════════════════════════════════════════════════

elif page == "🔍  Inference":
    st.title("🔍 Fault Inference")
    st.markdown(
        "Upload a Protocol v1.2 CSV file to classify the fault type and "
        "view per-window probability distributions."
    )
    st.markdown("---")

    if not model_exists():
        st.error(
            "No trained model found. "
            "Please complete training on the Train page first."
        )
        st.stop()

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
        st.error(f"Failed to load model: {e}")
        st.stop()

    uploaded = st.file_uploader(
        "📂  Choose a CSV file",
        type=["csv"],
        help="File must contain all 16 protocol columns including TIMESTAMP and ERROR_CODE.",
    )

    if uploaded is not None:
        import torch
        import torch.nn.functional as F_nn
        from models.feature_engineering import (
            FeatureNormalizer,
            CLASS_NAMES,
            CLASS_CODES,
            build_feature_matrix,
            impute_nan,
            sliding_windows,
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        try:
            df_raw = pd.read_csv(tmp_path, keep_default_na=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        st.markdown(
            '<div class="section-header">Data Preview</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(df_raw.head(10), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df_raw):,}")
        c2.metric("Columns", len(df_raw.columns))
        nan_pct = df_raw.iloc[:, :14].isna().mean().mean()
        c3.metric("NaN Rate", f"{nan_pct:.1%}")
        if "ERROR_CODE" in df_raw.columns:
            usable = (df_raw["ERROR_CODE"] < 300).sum()
            c4.metric("Usable Rows", f"{usable:,}")

        st.markdown("---")

        with st.spinner("Running inference…"):
            df_infer = df_raw.copy()
            if "ERROR_CODE" in df_infer.columns:
                df_infer = df_infer[df_infer["ERROR_CODE"] < 300].reset_index(drop=True)

            if len(df_infer) == 0:
                st.error("No usable rows remain after filtering UNUSABLE entries.")
                st.stop()

            feat = build_feature_matrix(df_infer, inf_cfg)
            feat = impute_nan(feat)
            feat_norm = normalizer.transform(feat)

            window = inf_cfg["data"]["window_size"]
            step = inf_cfg["data"]["step_size"]
            dummy_lbls = np.zeros(len(feat_norm), dtype=np.int64)
            X_wins, _ = sliding_windows(feat_norm, dummy_lbls, window, step)

            if len(X_wins) == 0:
                pad_len = window - len(feat_norm)
                feat_pad = np.pad(feat_norm, ((pad_len, 0), (0, 0)), mode="edge")
                X_wins = feat_pad[np.newaxis, :, :]

            X_tensor = torch.tensor(X_wins, dtype=torch.float32)
            all_probs = []
            with torch.no_grad():
                for i in range(0, len(X_tensor), 256):
                    logits = model(X_tensor[i : i + 256])
                    all_probs.append(F_nn.softmax(logits, dim=-1).numpy())

            all_probs = np.concatenate(all_probs, axis=0)
            mean_probs = all_probs.mean(axis=0)
            pred_label = int(np.argmax(mean_probs))
            pred_class = CLASS_NAMES[pred_label]
            pred_conf = float(mean_probs[pred_label])

        st.markdown(
            '<div class="section-header">Inference Results</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        res_col, detail_col = st.columns(2)

        with res_col:
            st.markdown("**Primary Prediction**")
            st.markdown(badge_html(pred_class), unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:#94a3b8;margin:4px 0;">'
                f'{CLASS_DESC.get(pred_class, "")}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color:#38bdf8;font-size:2rem;font-weight:700;margin:0;">'
                f"{pred_conf:.1%}</p>",
                unsafe_allow_html=True,
            )
            st.caption(f"Windows analysed: {len(X_wins)}")

        with detail_col:
            st.markdown("**Class Probabilities**")
            for i, cls in enumerate(CLASS_NAMES):
                color = CLASS_COLORS[cls]
                prob = float(mean_probs[i])
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
                    f'<span style="color:{color};font-weight:600;min-width:160px;">{cls}</span>'
                    f"{prob_bar_html(prob, color)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown(
            '<div class="section-header">Per-Window Probability Distribution</div>',
            unsafe_allow_html=True,
        )
        try:
            import plotly.graph_objects as go

            fig2 = go.Figure()
            win_idx = list(range(len(all_probs)))
            for i, cls in enumerate(CLASS_NAMES):
                fig2.add_trace(
                    go.Scatter(
                        x=win_idx,
                        y=all_probs[:, i],
                        mode="lines",
                        name=cls,
                        line=dict(color=list(CLASS_COLORS.values())[i], width=2),
                        fill="tozeroy",
                        opacity=0.3,
                    )
                )
            fig2.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                xaxis=dict(title="Window Index", gridcolor="#1e293b"),
                yaxis=dict(
                    title="Probability",
                    tickformat=".0%",
                    gridcolor="#1e293b",
                    range=[0, 1],
                ),
                legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
                margin=dict(l=10, r=10, t=20, b=10),
                height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.line_chart(pd.DataFrame(all_probs, columns=CLASS_NAMES))

        st.markdown("---")
        st.markdown(
            '<div class="section-header">Probability Summary</div>',
            unsafe_allow_html=True,
        )
        summary_df = pd.DataFrame(
            {
                "Fault Class": CLASS_NAMES,
                "Mean Prob": [f"{p:.2%}" for p in mean_probs],
                "Max Prob": [f"{p:.2%}" for p in all_probs.max(axis=0)],
                "Min Prob": [f"{p:.2%}" for p in all_probs.min(axis=0)],
                "Description": [CLASS_DESC[c] for c in CLASS_NAMES],
            }
        ).sort_values("Mean Prob", ascending=False)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        result_dict = {
            "file": uploaded.name,
            "prediction": pred_class,
            "confidence": f"{pred_conf:.4f}",
            "windows_analyzed": int(len(X_wins)),
            "class_probabilities": {
                cls: f"{float(mean_probs[i]):.6f}" for i, cls in enumerate(CLASS_NAMES)
            },
        }
        st.download_button(
            label="⬇️  Download Report (JSON)",
            data=json.dumps(result_dict, indent=2, ensure_ascii=False),
            file_name=f"pmu_fault_report_{uploaded.name}.json",
            mime="application/json",
        )
