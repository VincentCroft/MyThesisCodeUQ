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
import streamlit.components.v1 as _components
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
#  Plotly iframe renderer  (eliminates Streamlit re-render
#  flicker and enables proper legend toggle behaviour)
# ════════════════════════════════════════════════════════════
_PLOTLY_RENDER_COUNTER: list[int] = [0]  # mutable counter for unique div ids


def render_plotly(fig, height: int = 400, key: str = "") -> None:
    """Render a Plotly figure inside a stable <iframe> via components.html().

    Benefits vs st.plotly_chart():
    • The iframe is an independent document — Streamlit's Virtual-DOM
      diffing never touches its interior, so legend clicks / trace
      isolation never cause a page re-render / flicker.
    • displaylogo=false  removes the "Produced with Plotly" button.
    • A custom full-screen button is injected into the modebar.
    • Multi-select legend behaviour is kept (click = toggle one trace,
      double-click = isolate / restore all).
    """
    import plotly.io as _pio
    import json as _json
    import copy as _copy

    _PLOTLY_RENDER_COUNTER[0] += 1
    div_id = f"plotly_div_{_PLOTLY_RENDER_COUNTER[0]}_{key}"

    # ── Post-process figure layout ───────────────────────────────────────
    fig2 = _copy.deepcopy(fig)
    lay  = fig2.layout

    # 1. Axis: automargin + standoff so tick labels never overlap axis title
    _STANDOFF = 16
    for _ak in ("xaxis", "yaxis", "xaxis2", "yaxis2",
                "xaxis3", "yaxis3", "xaxis4", "yaxis4"):
        _ax = getattr(lay, _ak, None)
        if _ax is None:
            continue
        _ax.automargin = True
        try:
            if _ax.title:
                _ax.title.standoff = _STANDOFF
        except Exception:
            pass

    # 2. Legend → horizontal, anchored below the plot area
    #    This moves it completely outside the chart so it cannot overlap
    #    data and cannot be clipped by the iframe edge.
    #    Exception: radar / polar charts have no cartesian axes — keep
    #    their legend on the right but still outside the plot.
    _has_polar = bool(getattr(lay, "polar", None))
    if _has_polar:
        # For radar: put legend to the right with enough margin
        lay.legend = dict(
            orientation="v",
            x=1.02, xanchor="left",
            y=1.0,  yanchor="top",
            bgcolor="#1e293b",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=12),
        )
        # extra right margin so legend text is not clipped
        _mr = 160
    else:
        lay.legend = dict(
            orientation="h",
            x=0.5,    xanchor="center",
            y=-0.22,  yanchor="top",
            bgcolor="rgba(30,41,59,0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=12),
            traceorder="normal",
        )
        _mr = 20

    # 3. Margins: generous left/bottom so axis titles are always visible;
    #    right sized for legend placement; extra bottom for h-legend.
    _legend_b_extra = 0 if _has_polar else 70   # room for horizontal legend
    lay.margin = dict(
        l=75, r=_mr,
        t=getattr(getattr(lay, "margin", None), "t", None) or 50,
        b=60 + _legend_b_extra,
        pad=4,
    )

    fig_json = _pio.to_json(fig2, validate=False)

    # iframe must be tall enough to show chart + bottom legend
    iframe_h = height + _legend_b_extra + 30   # +30 for modebar

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ background:#0f172a; overflow:hidden; width:100%; height:100%; }}
  #chart-wrap {{ width:100%; height:{iframe_h}px; }}
</style>
</head>
<body>
<div id="chart-wrap">
  <div id="{div_id}" style="width:100%;height:100%;"></div>
</div>
<script>
(function(){{
  var fig    = {fig_json};
  var divId  = '{div_id}';
  var normalH = {iframe_h};
  var normalW = null;

  function containerW() {{
    return document.getElementById('chart-wrap').clientWidth || window.innerWidth;
  }}

  fig.layout        = fig.layout || {{}};
  fig.layout.height = normalH;
  fig.layout.autosize = false;

  function relayoutFS(entering) {{
    if (entering) {{
      Plotly.relayout(divId, {{ width: screen.width, height: screen.height }});
    }} else {{
      Plotly.relayout(divId, {{ width: normalW, height: normalH }});
    }}
  }}

  var config = {{
    displaylogo: false,
    responsive: false,
    modeBarButtonsToRemove: [],
    modeBarButtonsToAdd: [{{
      name: 'Full screen',
      icon: {{
        width: 500, height: 500,
        path: 'M 0 0 L 180 0 L 180 60 L 60 60 L 60 180 L 0 180 Z '
            + 'M 320 0 L 500 0 L 500 180 L 440 180 L 440 60 L 320 60 Z '
            + 'M 0 320 L 60 320 L 60 440 L 180 440 L 180 500 L 0 500 Z '
            + 'M 440 320 L 500 320 L 500 500 L 320 500 L 320 440 L 440 440 Z',
        ascent: 500, descent: 0,
      }},
      click: function() {{
        var isFS = !!(document.fullscreenElement
                   || document.webkitFullscreenElement
                   || document.mozFullScreenElement);
        var doc  = document.documentElement;
        if (!isFS) {{
          relayoutFS(true);
          var req = doc.requestFullscreen || doc.webkitRequestFullscreen
                 || doc.mozRequestFullScreen || doc.msRequestFullscreen;
          if (req) req.call(doc).catch(function(){{}});
        }} else {{
          relayoutFS(false);
          var exit = document.exitFullscreen || document.webkitExitFullscreen
                  || document.mozCancelFullScreen || document.msExitFullscreen;
          if (exit) exit.call(document);
        }}
      }}
    }}],
  }};

  fig.layout.width = containerW();
  Plotly.newPlot(divId, fig.data, fig.layout, config).then(function(){{
    normalW = containerW();
  }});

  if (window.ResizeObserver) {{
    new ResizeObserver(function() {{
      var inFS = !!(document.fullscreenElement
                || document.webkitFullscreenElement
                || document.mozFullScreenElement);
      if (!inFS) {{
        var w = containerW();
        normalW = w;
        Plotly.relayout(divId, {{ width: w, height: normalH }});
      }}
    }}).observe(document.getElementById('chart-wrap'));
  }}

  function onFSChange() {{
    var inFS = !!(document.fullscreenElement
              || document.webkitFullscreenElement
              || document.mozFullScreenElement);
    if (!inFS) relayoutFS(false);
  }}
  document.addEventListener('fullscreenchange',       onFSChange);
  document.addEventListener('webkitfullscreenchange', onFSChange);
  document.addEventListener('mozfullscreenchange',    onFSChange);
  document.addEventListener('MSFullscreenChange',     onFSChange);
}})();
</script>
</body>
</html>
"""
    _components.html(html, height=iframe_h + 4, scrolling=False)


# ════════════════════════════════════════════════════════════
#  Run management helpers
# ════════════════════════════════════════════════════════════


def list_runs() -> list[str]:
    """Return all run names that have a valid best_model.pt, newest first."""
    if not CKPT_BASE_DIR.exists():
        return []
    runs = []
    for d in sorted(CKPT_BASE_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "best_model.pt").exists():
            runs.append(d.name)
    return runs


def get_run_paths(run_name: str) -> dict:
    return run_paths(run_name)


# ════════════════════════════════════════════════════════════
#  Legacy helpers (operate on a given paths-dict)
# ════════════════════════════════════════════════════════════


def load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_cfg(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def model_ready(p: dict) -> bool:
    return p["best_model_pt"].exists() and p["normalizer_npz"].exists()


def load_history(p: dict) -> Optional[dict]:
    path = p["training_history_json"]
    if path.exists():
        with open(path) as f:
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

    # ── Active run selector ──────────────────────────────
    st.markdown("**Active Run**")
    all_runs = list_runs()
    if all_runs:
        selected_run = st.selectbox(
            "Select checkpoint run",
            all_runs,
            index=0,
            label_visibility="collapsed",
            help="Choose which trained model to use for Analysis and Inference.",
        )
    else:
        selected_run = None
        st.caption("No trained runs yet.")

    # Store in session state so pages can access it
    st.session_state["selected_run"] = selected_run

    st.markdown("---")
    st.markdown("**Model Status**")
    if selected_run:
        rp = get_run_paths(selected_run)
        if model_ready(rp):
            st.success(f"✅  Run: `{selected_run}`")
            meta_path = rp["model_meta_json"]
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                st.caption(
                    f"Type: **{meta.get('model_type', '?')}**  |  "
                    f"Classes: **{meta.get('num_classes', '?')}**"
                )
        else:
            st.warning("⚠️  Run folder incomplete")
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
        runs = list_runs()
        if runs:
            rp = get_run_paths(runs[0])
            hist = load_history(rp)
            if hist:
                best = max(hist["val_acc"])
                st.metric("Best Val Accuracy", f"{best:.2%}")
            else:
                st.metric("Best Val Accuracy", "—")
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
    _home_runs = list_runs()
    hist = load_history(get_run_paths(_home_runs[0])) if _home_runs else None
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
            render_plotly(fig, height=320, key="home_history")
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

    # ── Track whether config has unsaved changes ─────────
    # Snapshot the values read from disk so we can compare
    # against whatever the user has typed in the widgets.
    _disk_snapshot = {
        "model_type": cfg["model"]["type"],
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "lr": float(cfg["training"]["learning_rate"]),
        "scheduler": cfg["training"]["scheduler"],
        "patience": cfg["training"]["early_stopping_patience"],
        "window_size": cfg["data"]["window_size"],
        "step_size": cfg["data"]["step_size"],
        "val_split": float(cfg["data"]["val_split"]),
    }
    if "config_saved" not in st.session_state:
        st.session_state["config_saved"] = True  # starts clean

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

        # Detect unsaved changes
        _current = {
            "model_type": model_type,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "scheduler": scheduler,
            "patience": int(patience),
            "window_size": int(window_size),
            "step_size": int(step_size),
            "val_split": float(val_split),
        }
        _has_unsaved = (
            _current["model_type"] != _disk_snapshot["model_type"]
            or _current["epochs"] != _disk_snapshot["epochs"]
            or _current["batch_size"] != _disk_snapshot["batch_size"]
            or abs(_current["lr"] - _disk_snapshot["lr"]) > 1e-9
            or _current["scheduler"] != _disk_snapshot["scheduler"]
            or _current["patience"] != _disk_snapshot["patience"]
            or _current["window_size"] != _disk_snapshot["window_size"]
            or _current["step_size"] != _disk_snapshot["step_size"]
            or abs(_current["val_split"] - _disk_snapshot["val_split"]) > 1e-6
        )

        if _has_unsaved:
            st.session_state["config_saved"] = False
            st.warning("⚠️  Unsaved changes — click **Save Config** before training.")
        else:
            st.session_state["config_saved"] = True

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
            st.session_state["config_saved"] = True
            st.success("✅  Configuration saved.")
            st.rerun()

    st.markdown("---")

    # ── Run Name (requires explicit confirmation) ─────────
    from datetime import datetime as _dt

    if "confirmed_run_name" not in st.session_state:
        st.session_state["confirmed_run_name"] = _dt.now().strftime("%Y-%m-%d-%H-%M")

    with st.container():
        st.markdown("**Run Name**")
        rn_col, btn_col = st.columns([3, 1])
        with rn_col:
            run_name_input = st.text_input(
                "run_name_field",
                value=st.session_state["confirmed_run_name"],
                placeholder=_dt.now().strftime("%Y-%m-%d-%H-%M"),
                label_visibility="collapsed",
                help=(
                    "Checkpoints will be saved to  logs/checkpoints/<run_name>/. "
                    "Edit and click Confirm to lock in the name."
                ),
            )
        with btn_col:
            if st.button("✔  Confirm Name", use_container_width=True):
                name_candidate = run_name_input.strip() or _dt.now().strftime(
                    "%Y-%m-%d-%H-%M"
                )
                st.session_state["confirmed_run_name"] = name_candidate
                st.success(f"Run name set to **{name_candidate}**")

    run_name_final = st.session_state["confirmed_run_name"]
    existing_runs = list_runs()

    if run_name_final in existing_runs:
        st.warning(
            f"⚠️  Run **{run_name_final}** already exists — it will be overwritten."
        )
    else:
        st.caption(f"Will save to  `logs/checkpoints/{run_name_final}/`")

    st.markdown("---")

    # ── Start / Stop controls ─────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        start_btn = st.button(
            "▶️  Start Training",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.get("training_running", False),
        )
    with ctrl_col2:
        stop_btn = st.button(
            "⏹  Stop Training",
            use_container_width=True,
            disabled=not st.session_state.get("training_running", False),
        )
    with ctrl_col3:
        if st.session_state.get("training_running", False):
            st.info("🔄  Training in progress…")

    log_placeholder = st.empty()
    progress_placeholder = st.empty()
    metric_placeholder = st.empty()

    # ── Guard: unsaved config ────────────────────────────
    if start_btn and not st.session_state.get("config_saved", True):
        st.error(
            "❌  You have unsaved configuration changes.  "
            "Please click **Save Config** before starting training."
        )
        start_btn = False  # cancel the start

    # ── Launch training ───────────────────────────────────
    if start_btn:
        st.session_state["training_running"] = True
        st.session_state["stop_requested"] = False
        train_script = CODE_DIR / "train.py"
        cmd = [
            PYTHON_EXE,
            str(train_script),
            "--model",
            cfg["model"]["type"],
            "--run-name",
            run_name_final,
        ]

        log_lines: list[str] = []
        proc_holder: list = [None]  # mutable container so thread can write proc ref

        def stream_process():
            import subprocess as _sp

            proc = _sp.Popen(
                cmd,
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(CODE_DIR),
            )
            proc_holder[0] = proc
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                if st.session_state.get("stop_requested", False):
                    proc.terminate()
                    log_lines.append("\n  ⏹  Training terminated by user.")
                    break
            proc.wait()
            log_lines.append(f"\n  Process exited with code {proc.returncode}")
            st.session_state["training_running"] = False

        thread = threading.Thread(target=stream_process, daemon=True)
        thread.start()

        progress_bar = progress_placeholder.progress(0, text="Training in progress…")
        epoch_seen = 0
        max_epoch = int(epochs)

        while thread.is_alive():
            time.sleep(0.5)

            # Check stop button via rerun signal
            if stop_btn:
                st.session_state["stop_requested"] = True

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

        thread.join()
        progress_bar.progress(1.0, text="✅  Done!")
        log_placeholder.markdown(
            f'<div class="log-box">{"<br>".join(log_lines)}</div>',
            unsafe_allow_html=True,
        )
        st.session_state["training_running"] = False

        _finished_rp = get_run_paths(run_name_final)
        hist_fin = load_history(_finished_rp)
        if hist_fin:
            best_val = max(hist_fin["val_acc"])
            m1, m2, m3 = metric_placeholder.columns(3)
            m1.metric("Best Val Accuracy", f"{best_val:.2%}")
            m2.metric("Total Epochs", f"{len(hist_fin['val_acc'])}")
            m3.metric("Final Val Loss", f"{hist_fin['val_loss'][-1]:.4f}")
            st.balloons()

        st.info(
            f"✅  Run **{run_name_final}** saved. Select it from the sidebar to analyse."
        )

    # ── Stop button outside the start block ──────────────
    if stop_btn and st.session_state.get("training_running", False):
        st.session_state["stop_requested"] = True
        st.warning(
            "⏹  Stop signal sent — training will terminate after the current epoch."
        )


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

    _sel_run = st.session_state.get("selected_run")
    if not _sel_run:
        st.error(
            "No trained run found. Please complete training on the Train page first."
        )
        st.stop()

    _rp = get_run_paths(_sel_run)
    if not model_ready(_rp):
        st.error(f"Run **{_sel_run}** is missing model files. Please re-run training.")
        st.stop()

    st.info(f"📂  Analysing run: **{_sel_run}**  (change in the sidebar)")
    CM_PATH = _rp["confusion_matrix_json"]
    TSNE_PATH = _rp["tsne_embeddings_json"]

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
        hist = load_history(_rp)
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
                    render_plotly(fig_acc, height=300, key="curve_acc")
                with col_b:
                    render_plotly(fig_loss, height=300, key="curve_loss")
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
                render_plotly(fig_gap, height=250, key="curve_gap")

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
                render_plotly(fig_cm, height=420, key="cm_heatmap")
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
                render_plotly(radar_fig, height=380, key="report_radar")

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
                render_plotly(fig_f1, height=280, key="report_f1bar")

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
                render_plotly(fig_tsne, height=520, key="tsne_scatter")

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
                render_plotly(fig_dist, height=240, key="tsne_dist_bar")
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
        "Upload one or more Protocol v1.2 CSV files to classify fault types and "
        "view per-window probability distributions."
    )
    st.markdown("---")

    _sel_run = st.session_state.get("selected_run")
    if not _sel_run:
        st.error(
            "No trained run found. Please complete training on the Train page first."
        )
        st.stop()

    _rp = get_run_paths(_sel_run)
    if not model_ready(_rp):
        st.error(f"Run **{_sel_run}** is missing model files. Please re-run training.")
        st.stop()

    st.info(f"📂  Using run: **{_sel_run}**  (change in the sidebar)")

    @st.cache_resource(show_spinner="Loading model…")
    def load_inference_engine(run_name: str):
        import torch
        from models.feature_engineering import FeatureNormalizer
        from models.classifier import build_model

        rp = get_run_paths(run_name)
        meta = json.loads(rp["model_meta_json"].read_text())
        ckpt = torch.load(rp["best_model_pt"], map_location="cpu")
        cfg_ck = ckpt["cfg"]
        cfg_ck["model"]["input_size"] = meta["input_size"]

        mdl = build_model(cfg_ck)
        mdl.load_state_dict(ckpt["model_state"])
        mdl.eval()

        normalizer = FeatureNormalizer.load(rp["normalizer_npz"])
        return mdl, normalizer, cfg_ck

    try:
        model, normalizer, inf_cfg = load_inference_engine(_sel_run)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # ── File uploader (multiple files allowed) ────────────
    uploaded_files = st.file_uploader(
        "📂  Choose CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Select one or more Protocol v1.2 CSV files.  Each will be analysed independently.",
    )

    if uploaded_files:
        import torch
        import torch.nn.functional as F_nn
        from models.feature_engineering import (
            CLASS_NAMES,
            build_feature_matrix,
            impute_nan,
            sliding_windows,
        )

        # ── Helper: run inference on one uploaded file ────
        def _infer_one(uploaded_f):
            """Return (pred_class, pred_conf, mean_probs, all_probs, df_raw, X_wins) or raise."""
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(uploaded_f.read())
                tmp_path = Path(tmp.name)
            df_raw = pd.read_csv(tmp_path, keep_default_na=True)
            df_infer = df_raw.copy()
            if "ERROR_CODE" in df_infer.columns:
                df_infer = df_infer[df_infer["ERROR_CODE"] < 300].reset_index(drop=True)
            if len(df_infer) == 0:
                raise ValueError(
                    "No usable rows remain after filtering UNUSABLE entries."
                )

            feat = build_feature_matrix(df_infer, inf_cfg)
            feat = impute_nan(feat)
            feat_norm = normalizer.transform(feat)

            window = inf_cfg["data"]["window_size"]
            step = inf_cfg["data"]["step_size"]
            dummy = np.zeros(len(feat_norm), dtype=np.int64)
            X_wins, _ = sliding_windows(feat_norm, dummy, window, step)
            if len(X_wins) == 0:
                pad_len = window - len(feat_norm)
                feat_pad = np.pad(feat_norm, ((pad_len, 0), (0, 0)), mode="edge")
                X_wins = feat_pad[np.newaxis, :, :]

            X_tensor = torch.tensor(X_wins, dtype=torch.float32)
            all_probs_list = []
            with torch.no_grad():
                for i in range(0, len(X_tensor), 256):
                    logits = model(X_tensor[i : i + 256])
                    all_probs_list.append(F_nn.softmax(logits, dim=-1).numpy())

            all_probs_arr = np.concatenate(all_probs_list, axis=0)
            mean_probs = all_probs_arr.mean(axis=0)
            pred_label = int(np.argmax(mean_probs))
            pred_class = CLASS_NAMES[pred_label]
            pred_conf = float(mean_probs[pred_label])
            return pred_class, pred_conf, mean_probs, all_probs_arr, df_raw, X_wins

        # ── Run all files (with one spinner) ─────────────
        results_cache: list[dict] = []
        errors_cache: list[str] = []
        with st.spinner(f"Running inference on {len(uploaded_files)} file(s)…"):
            for uf in uploaded_files:
                try:
                    pred_class, pred_conf, mean_probs, all_probs, df_raw, X_wins = (
                        _infer_one(uf)
                    )
                    results_cache.append(
                        {
                            "name": uf.name,
                            "pred_class": pred_class,
                            "pred_conf": pred_conf,
                            "mean_probs": mean_probs,
                            "all_probs": all_probs,
                            "df_raw": df_raw,
                            "X_wins": X_wins,
                        }
                    )
                except Exception as exc:
                    errors_cache.append(f"**{uf.name}**: {exc}")

        # Show errors
        for err in errors_cache:
            st.error(err)

        if not results_cache:
            st.stop()

        # ── Batch summary table (multi-file only) ─────────
        if len(results_cache) > 1:
            st.markdown(
                '<div class="section-header">Batch Summary</div>',
                unsafe_allow_html=True,
            )
            summary_rows = []
            for r in results_cache:
                summary_rows.append(
                    {
                        "File": r["name"],
                        "Prediction": r["pred_class"],
                        "Confidence": f"{r['pred_conf']:.1%}",
                        **{
                            cls: f"{float(r['mean_probs'][i]):.2%}"
                            for i, cls in enumerate(CLASS_NAMES)
                        },
                    }
                )
            st.dataframe(
                pd.DataFrame(summary_rows), hide_index=True, use_container_width=True
            )
            st.markdown("---")

        # ── Per-file detailed results ─────────────────────
        all_reports: list[dict] = []
        for r in results_cache:
            fname = r["name"]
            pred_class = r["pred_class"]
            pred_conf = r["pred_conf"]
            mean_probs = r["mean_probs"]
            all_probs = r["all_probs"]
            df_raw = r["df_raw"]
            X_wins = r["X_wins"]

            # Expand by default if only one file, else collapsed
            expand_by_default = len(results_cache) == 1
            with st.expander(
                f"📄  {fname}  —  {pred_class}  ({pred_conf:.1%})",
                expanded=expand_by_default,
            ):

                st.markdown(
                    '<div class="section-header">Data Preview</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(df_raw.head(10), use_container_width=True)

                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Rows", f"{len(df_raw):,}")
                dc2.metric("Columns", len(df_raw.columns))
                nan_pct = df_raw.iloc[:, :14].isna().mean().mean()
                dc3.metric("NaN Rate", f"{nan_pct:.1%}")
                if "ERROR_CODE" in df_raw.columns:
                    usable_cnt = (df_raw["ERROR_CODE"] < 300).sum()
                    dc4.metric("Usable Rows", f"{usable_cnt:,}")

                st.markdown("---")
                st.markdown(
                    '<div class="section-header">Inference Results</div>',
                    unsafe_allow_html=True,
                )

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
                    import plotly.graph_objects as go  # noqa: F811

                    fig2 = go.Figure()
                    win_idx = list(range(len(all_probs)))
                    for i, cls in enumerate(CLASS_NAMES):
                        fig2.add_trace(
                            go.Scatter(
                                x=win_idx,
                                y=all_probs[:, i],
                                mode="lines",
                                name=cls,
                                line=dict(
                                    color=list(CLASS_COLORS.values())[i], width=2
                                ),
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
                    render_plotly(fig2, height=280, key=f"infer_prob_{fname}")
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

                report_dict = {
                    "file": fname,
                    "prediction": pred_class,
                    "confidence": f"{pred_conf:.4f}",
                    "windows_analyzed": int(len(X_wins)),
                    "class_probabilities": {
                        cls: f"{float(mean_probs[i]):.6f}"
                        for i, cls in enumerate(CLASS_NAMES)
                    },
                }
                all_reports.append(report_dict)
                st.download_button(
                    label="⬇️  Download Report (JSON)",
                    data=json.dumps(report_dict, indent=2, ensure_ascii=False),
                    file_name=f"pmu_report_{fname}.json",
                    mime="application/json",
                    key=f"dl_{fname}",
                )

        # ── Batch download (multi-file only) ─────────────
        if len(all_reports) > 1:
            st.markdown("---")
            st.download_button(
                label=f"⬇️  Download All {len(all_reports)} Reports (JSON)",
                data=json.dumps(all_reports, indent=2, ensure_ascii=False),
                file_name="pmu_batch_report.json",
                mime="application/json",
                type="primary",
            )
