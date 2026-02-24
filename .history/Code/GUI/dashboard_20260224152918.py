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


# ── Theme Management ────────────────────────────────────────
def toggle_theme():
    config_path = THESIS / ".streamlit" / "config.toml"
    config_path.parent.mkdir(exist_ok=True)

    current_theme = "dark"
    if config_path.exists():
        content = config_path.read_text()
        if 'base="light"' in content.replace(" ", ""):
            current_theme = "light"

    new_theme = "light" if current_theme == "dark" else "dark"
    config_path.write_text(f'[theme]\nbase="{new_theme}"\n')


# Determine current theme
_config_path = THESIS / ".streamlit" / "config.toml"
_current_theme = "dark"
if _config_path.exists():
    _content = _config_path.read_text()
    if 'base="light"' in _content.replace(" ", ""):
        _current_theme = "light"

if _current_theme == "dark":
    theme_vars = """
    :root {
        --bg-color: #0f172a;
        --text-color: #f8fafc;
        --text-muted: #94a3b8;
        --border-color: rgba(255, 255, 255, 0.1);
        --accent-color: #38bdf8;
        --metric-bg: rgba(30, 41, 59, 0.7);
        --shadow-color: rgba(0, 0, 0, 0.4);
        --badge-normal-bg: rgba(6, 78, 59, 0.8); --badge-normal-text: #34d399;
        --badge-slg-bg: rgba(124, 45, 18, 0.8); --badge-slg-text: #fb923c;
        --badge-ll-bg: rgba(113, 63, 18, 0.8); --badge-ll-text: #fbbf24;
        --badge-3p-bg: rgba(76, 29, 149, 0.8); --badge-3p-text: #c4b5fd;
        --badge-unk-bg: rgba(30, 41, 59, 0.8); --badge-unk-text: #94a3b8;
        --bar-bg: rgba(255, 255, 255, 0.05);
        --log-bg: rgba(15, 23, 42, 0.8);
        --sidebar-bg: rgba(15, 23, 42, 0.6);
        --expander-bg: rgba(30, 41, 59, 0.6);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-hover: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.1);
    }
    """
else:
    theme_vars = """
    :root {
        --bg-color: #f8fafc;
        --text-color: #0f172a;
        --text-muted: #475569;
        --border-color: rgba(0, 0, 0, 0.1);
        --accent-color: #0284c7;
        --metric-bg: rgba(255, 255, 255, 0.7);
        --shadow-color: rgba(0, 0, 0, 0.05);
        --badge-normal-bg: rgba(209, 250, 229, 0.8); --badge-normal-text: #065f46;
        --badge-slg-bg: rgba(255, 237, 213, 0.8); --badge-slg-text: #9a3412;
        --badge-ll-bg: rgba(254, 243, 199, 0.8); --badge-ll-text: #92400e;
        --badge-3p-bg: rgba(237, 233, 254, 0.8); --badge-3p-text: #5b21b6;
        --badge-unk-bg: rgba(241, 245, 249, 0.8); --badge-unk-text: #475569;
        --bar-bg: rgba(0, 0, 0, 0.05);
        --log-bg: rgba(248, 250, 252, 0.8);
        --sidebar-bg: rgba(248, 250, 252, 0.6);
        --expander-bg: rgba(255, 255, 255, 0.6);
        --glass-bg: rgba(255, 255, 255, 0.6);
        --glass-hover: rgba(255, 255, 255, 0.9);
        --glass-border: rgba(0, 0, 0, 0.08);
    }
    """

# ── CSS ─────────────────────────────────────────────────────
css_path = GUI_DIR / "style.css"
custom_css = css_path.read_text() if css_path.exists() else ""

# Inject CSS and a small JS script to clear Streamlit's localStorage theme preference
# This ensures that the app always respects the config.toml theme instead of the user's manual override.
js_clear_theme = """
<script>
    if (window.localStorage.getItem('stActiveTheme')) {
        window.localStorage.removeItem('stActiveTheme');
    }
</script>
"""
st.markdown(f"<style>{theme_vars}\n{custom_css}</style>\n{js_clear_theme}", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════
CLASS_NAMES = ["NORMAL", "SLG_FAULT", "LL_FAULT", "THREE_PHASE_FAULT"]
CLASS_COLORS = {
    "NORMAL": "#34d399" if _current_theme == "dark" else "#059669",
    "SLG_FAULT": "#fb923c" if _current_theme == "dark" else "#ea580c",
    "LL_FAULT": "#fbbf24" if _current_theme == "dark" else "#d97706",
    "THREE_PHASE_FAULT": "#c4b5fd" if _current_theme == "dark" else "#7c3aed",
    "UNKNOWN": "#94a3b8" if _current_theme == "dark" else "#64748b",
}
CLASS_DESC = {
    "NORMAL": "Normal operation  (ERROR 000)",
    "SLG_FAULT": "Single line-to-ground fault  (ERROR 201)",
    "LL_FAULT": "Line-to-line fault  (ERROR 202)",
    "THREE_PHASE_FAULT": "Three-phase short-circuit  (ERROR 204)",
}
PYTHON_EXE = sys.executable

# Theme-dependent Plotly colors
IFRAME_BG = "#0f172a" if _current_theme == "dark" else "#f8fafc"
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.1)" if _current_theme == "dark" else "rgba(0,0,0,0.1)"
TEXT_COLOR = "#94a3b8" if _current_theme == "dark" else "#475569"
LEGEND_BG = "rgba(30, 41, 59, 0.8)" if _current_theme == "dark" else "rgba(255, 255, 255, 0.8)"
LEGEND_BORDER = "rgba(255,255,255,0.1)" if _current_theme == "dark" else "rgba(0,0,0,0.1)"
ACCENT_BLUE = "#38bdf8" if _current_theme == "dark" else "#0284c7"
ACCENT_GREEN = "#34d399" if _current_theme == "dark" else "#059669"
ACCENT_ORANGE = "#fb923c" if _current_theme == "dark" else "#ea580c"
ACCENT_YELLOW = "#fbbf24" if _current_theme == "dark" else "#d97706"
ACCENT_PINK = "#f472b6" if _current_theme == "dark" else "#db2777"
ACCENT_RED = "#ef4444" if _current_theme == "dark" else "#dc2626"

LOG_BG = "rgba(15, 23, 42, 0.8)" if _current_theme == "dark" else "rgba(248, 250, 252, 0.8)"
LOG_BORDER = "rgba(255, 255, 255, 0.1)" if _current_theme == "dark" else "rgba(0, 0, 0, 0.1)"
LOG_TEXT = "#94a3b8" if _current_theme == "dark" else "#475569"

# ════════════════════════════════════════════════════════════
#  Plotly iframe renderer  (eliminates Streamlit re-render
#  flicker and enables proper legend toggle behaviour)
# ════════════════════════════════════════════════════════════
_PLOTLY_RENDER_COUNTER: list[int] = [0]  # mutable counter for unique div ids


def _render_log(placeholder, lines: list[str], box_h: int = 380) -> None:
    """Render log lines in a hidden-scrollbar box that auto-scrolls to bottom."""
    import html as _html_mod

    content = _html_mod.escape("\n".join(lines[-200:]))
    html_str = f"""\
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;}}
#lb{{
  background:{LOG_BG};border:1px solid {LOG_BORDER};border-radius:12px;
  padding:16px;font-family:monospace;font-size:0.85rem;color:{LOG_TEXT};
  height:{box_h}px;overflow-y:scroll;white-space:pre-wrap;
  scrollbar-width:none;-ms-overflow-style:none;
  box-shadow: inset 0 4px 12px rgba(0,0,0,0.2);
}}
#lb::-webkit-scrollbar{{display:none;}}
</style></head><body>
<div id="lb">{content}</div>
<script>var e=document.getElementById('lb');e.scrollTop=e.scrollHeight;</script>
</body></html>"""
    with placeholder:
        _components.html(html_str, height=box_h + 4, scrolling=False)


def render_plotly(fig, height: int = 400, key: str = "") -> None:
    """Render a Plotly figure inside a stable <iframe> via components.html().

    Benefits vs st.plotly_chart():
    • The iframe is an independent document — Streamlit's Virtual-DOM
      diffing never touches its interior, so legend clicks / trace
      isolation never cause a page re-render / flicker.
    • displaylogo=false  removes the "Produced with Plotly" button.
    • A custom full-screen button is injected into the modebar.
    """
    import plotly.io as _pio
    import copy as _copy

    _PLOTLY_RENDER_COUNTER[0] += 1
    div_id = f"plotly_div_{_PLOTLY_RENDER_COUNTER[0]}_{key}"

    # ── Post-process figure layout ───────────────────────────────────────
    fig2 = _copy.deepcopy(fig)
    lay = fig2.layout

    # 1. All cartesian axes: automargin + standoff so titles don't overlap
    for _ak in (
        "xaxis",
        "yaxis",
        "xaxis2",
        "yaxis2",
        "xaxis3",
        "yaxis3",
        "xaxis4",
        "yaxis4",
    ):
        _ax = getattr(lay, _ak, None)
        if _ax is None:
            continue
        _ax.automargin = True
        try:
            if _ax.title:
                _ax.title.standoff = 16
        except Exception:
            pass

    # 2. Margins — preserve each chart's own t and b so horizontal
    #    bottom legends are never clipped.
    _prev_t = None
    _prev_b = None
    try:
        _prev_t = lay.margin.t
    except Exception:
        pass
    try:
        _prev_b = lay.margin.b
    except Exception:
        pass
    lay.margin = dict(
        l=75,
        t=(_prev_t or 50),
        b=(_prev_b if _prev_b and _prev_b > 60 else 60),
        pad=4,
    )
    # Explicitly remove 'r' if it exists so Plotly auto-sizes it
    lay.margin.r = None

    # 3. Let JS measure the real container width and pass it explicitly.
    #    autosize=True ensures Plotly calculates margin.r to fit the legend.
    lay.autosize = True
    lay.width = None  # placeholder; JS will fill this before newPlot
    lay.height = height

    fig_json = _pio.to_json(fig2, validate=False)

    # iframe height = chart height + modebar (30) + small safety pad (6)
    iframe_h = height + 36

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{
    background: {IFRAME_BG};
    width: 100%; height: 100%;
    overflow: hidden;
  }}
  #pw {{
    position: absolute;
    inset: 0;
  }}
</style>
</head>
<body>
<div id="pw">
  <div id="{div_id}" style="width:100%;height:100%;"></div>
</div>
<script>
(function(){{
  var fig   = {fig_json};
  var divId = '{div_id}';
  var normH = {height};
  var pw    = document.getElementById('pw');

  /* Measure container width BEFORE newPlot so margin.r is respected */
  function containerW() {{ return pw.clientWidth || window.innerWidth; }}

  /* Initial plot with explicit width — autosize:true guarantees margin.r is calculated */
  fig.layout.width  = containerW();
  fig.layout.autosize = true;

  var cfg = {{
    displaylogo:  false,
    responsive:   false,   /* we manage resize manually via ResizeObserver */
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
          /* Enter fullscreen first; relayout AFTER the transition */
          var req = doc.requestFullscreen || doc.webkitRequestFullscreen
                 || doc.mozRequestFullScreen || doc.msRequestFullscreen;
          if (req) req.call(doc).catch(function(){{}});
        }} else {{
          var exit = document.exitFullscreen || document.webkitExitFullscreen
                  || document.mozCancelFullScreen || document.msExitFullscreen;
          if (exit) exit.call(document).catch(function(){{}});
        }}
      }}
    }}],
  }};

  Plotly.newPlot(divId, fig.data, fig.layout, cfg);

  /* Fullscreen change — relayout AFTER browser has resized the viewport */
  function onFSChange() {{
    var inFS = !!(document.fullscreenElement
              || document.webkitFullscreenElement
              || document.mozFullScreenElement);
    if (inFS) {{
      Plotly.relayout(divId, {{ width: screen.width, height: screen.height }});
    }} else {{
      Plotly.relayout(divId, {{ width: containerW(), height: normH }});
    }}
  }}
  document.addEventListener('fullscreenchange',       onFSChange);
  document.addEventListener('webkitfullscreenchange', onFSChange);
  document.addEventListener('mozfullscreenchange',    onFSChange);
  document.addEventListener('MSFullscreenChange',     onFSChange);

  /* Track Streamlit column width changes */
  if (window.ResizeObserver) {{
    var ro = new ResizeObserver(function() {{
      var inFS = !!(document.fullscreenElement || document.webkitFullscreenElement);
      if (!inFS) Plotly.relayout(divId, {{ width: containerW() }});
    }});
    ro.observe(pw);
  }}
}})();
</script>
</body>
</html>
"""
    _components.html(html, height=iframe_h, scrolling=False)


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

    # Theme toggle button
    theme_icon = "🌙" if _current_theme == "dark" else "☀️"
    theme_label = "Dark Mode" if _current_theme == "dark" else "Light Mode"
    if st.button(
        f"{theme_icon}  {theme_label}", use_container_width=True, on_click=toggle_theme
    ):
        pass  # The on_click callback handles the theme switch and Streamlit auto-reruns

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
                f'<span style="color:{TEXT_COLOR};">{desc}</span>'
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
                    line=dict(color=ACCENT_BLUE, width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["val_acc"],
                    name="Val Acc",
                    line=dict(color=ACCENT_GREEN, width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["train_loss"],
                    name="Train Loss",
                    line=dict(color=ACCENT_ORANGE, width=2, dash="dot"),
                    yaxis="y2",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["val_loss"],
                    name="Val Loss",
                    line=dict(color=ACCENT_YELLOW, width=2, dash="dot"),
                    yaxis="y2",
                )
            )
            fig.update_layout(
                paper_bgcolor=PAPER_BG,
                plot_bgcolor=PLOT_BG,
                font=dict(color=TEXT_COLOR),
                legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BORDER, x=1.1),
                yaxis=dict(
                    title="Accuracy",
                    tickformat=".0%",
                    gridcolor=GRID_COLOR,
                    color=TEXT_COLOR,
                ),
                yaxis2=dict(
                    title="Loss",
                    overlaying="y",
                    side="right",
                    gridcolor=GRID_COLOR,
                    color=TEXT_COLOR,
                ),
                xaxis=dict(title="Epoch", gridcolor=GRID_COLOR, color=TEXT_COLOR),
                height=320,
            )
            render_plotly(fig, 320, "home_history")
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

        # Flag file path — same logic as train.py's STOP_FLAG
        _run_paths = get_run_paths(run_name_final)
        stop_flag_path = _run_paths["ckpt_dir"] / "STOP"
        # Clean up any stale flag from a previous run
        if stop_flag_path.exists():
            stop_flag_path.unlink()

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
            if stop_btn or st.session_state.get("stop_requested", False):
                if not stop_flag_path.exists():
                    stop_flag_path.parent.mkdir(parents=True, exist_ok=True)
                    stop_flag_path.touch()
                    log_lines.append(
                        "\n  ⏹  Stop signal sent — finishing current epoch then saving…"
                    )
                    st.session_state["stop_requested"] = True

            _render_log(log_placeholder, log_lines)
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
        # Clean up stop flag if still present
        if stop_flag_path.exists():
            stop_flag_path.unlink()
        progress_bar.progress(1.0, text="✅  Done!")
        _render_log(log_placeholder, log_lines)
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
        _run_paths_stop = get_run_paths(run_name_final)
        _sfp = _run_paths_stop["ckpt_dir"] / "STOP"
        _sfp.parent.mkdir(parents=True, exist_ok=True)
        _sfp.touch()
        st.warning(
            "⏹  Stop signal sent — training will finish the current epoch then save."
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
                        line=dict(color=ACCENT_BLUE, width=2),
                        marker=dict(size=4),
                    )
                )
                fig_acc.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=[v * 100 for v in hist["val_acc"]],
                        name="Val Acc (%)",
                        mode="lines+markers",
                        line=dict(color=ACCENT_GREEN, width=2),
                        marker=dict(size=4),
                    )
                )
                fig_acc.add_vline(
                    x=best_ep,
                    line_dash="dot",
                    line_color=ACCENT_YELLOW,
                    annotation_text=f"Best epoch {best_ep}",
                    annotation_font_color=ACCENT_YELLOW,
                )
                fig_acc.update_layout(
                    title="Accuracy Curve",
                    paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG,
                    font=dict(color=TEXT_COLOR),
                    xaxis=dict(title="Epoch", gridcolor=GRID_COLOR),
                    yaxis=dict(title="Accuracy (%)", gridcolor=GRID_COLOR),
                    legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BORDER),
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
                        line=dict(color=ACCENT_ORANGE, width=2),
                        marker=dict(size=4),
                    )
                )
                fig_loss.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=hist["val_loss"],
                        name="Val Loss",
                        mode="lines+markers",
                        line=dict(color=ACCENT_PINK, width=2, dash="dot"),
                        marker=dict(size=4),
                    )
                )
                fig_loss.add_vline(x=best_ep, line_dash="dot", line_color=ACCENT_YELLOW)
                fig_loss.update_layout(
                    title="Loss Curve",
                    paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG,
                    font=dict(color=TEXT_COLOR),
                    xaxis=dict(title="Epoch", gridcolor=GRID_COLOR),
                    yaxis=dict(title="Loss", gridcolor=GRID_COLOR),
                    legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BORDER),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=300,
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    render_plotly(fig_acc, 300, "train_acc")
                with col_b:
                    render_plotly(fig_loss, 300, "train_loss")
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
                        line=dict(color=ACCENT_PINK, width=2),
                    )
                )
                fig_gap.add_hline(
                    y=5,
                    line_dash="dot",
                    line_color=ACCENT_YELLOW,
                    annotation_text="5 % threshold",
                )
                fig_gap.add_hline(
                    y=10,
                    line_dash="dot",
                    line_color=ACCENT_RED,
                    annotation_text="10 % threshold",
                )
                fig_gap.update_layout(
                    title="Train − Val Accuracy Gap (overfit indicator)",
                    paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG,
                    font=dict(color=TEXT_COLOR),
                    xaxis=dict(title="Epoch", gridcolor=GRID_COLOR),
                    yaxis=dict(title="Gap (%)", gridcolor=GRID_COLOR),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=250,
                )
                render_plotly(fig_gap, 250, "fig_gap")

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
                if _current_theme == "dark":
                    colorscale = [
                        [0.0, "#0f172a"],
                        [0.3, "#1e3a5f"],
                        [0.6, "#1d4ed8"],
                        [1.0, "#38bdf8"],
                    ]
                else:
                    colorscale = [
                        [0.0, "#f8fafc"],
                        [0.3, "#bae6fd"],
                        [0.6, "#38bdf8"],
                        [1.0, "#0284c7"],
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
                    paper_bgcolor=PAPER_BG,
                    font=dict(color=TEXT_COLOR),
                    xaxis=dict(title="Predicted", side="bottom"),
                    yaxis=dict(title="True", autorange="reversed"),
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=420,
                )
                render_plotly(fig_cm, 420, "fig_cm")
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
                            line=dict(color=CLASS_COLORS.get(cls, TEXT_COLOR), width=2),
                            opacity=0.7,
                        )
                    )
                radar_fig.update_layout(
                    polar=dict(
                        bgcolor=LEGEND_BG,
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            gridcolor=LEGEND_BORDER,
                            color=TEXT_COLOR,
                        ),
                        angularaxis=dict(gridcolor=LEGEND_BORDER, color=TEXT_COLOR),
                    ),
                    paper_bgcolor=PAPER_BG,
                    font=dict(color=TEXT_COLOR),
                    legend=dict(
                        bgcolor=LEGEND_BG,
                        bordercolor=LEGEND_BORDER,
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=-0.12,
                        yanchor="top",
                        entrywidth=170,
                        entrywidthmode="pixels",
                    ),
                    title="Per-Class Precision / Recall / F1 Radar",
                    margin=dict(l=10, r=10, t=50, b=90),
                    height=460,
                )
                render_plotly(radar_fig, 460, "radar_fig")

                f1_vals = [per[c]["f1"] for c in cls_names]
                colors = [CLASS_COLORS.get(c, TEXT_COLOR) for c in cls_names]
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
                    paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG,
                    font=dict(color=TEXT_COLOR),
                    xaxis=dict(gridcolor=GRID_COLOR),
                    yaxis=dict(range=[0, 1.1], gridcolor=GRID_COLOR),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=280,
                )
                render_plotly(fig_f1, 280, "fig_f1")

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
                                color=CLASS_COLORS.get(cls, TEXT_COLOR),
                                size=5,
                                opacity=0.75,
                                line=dict(width=0),
                            ),
                        )
                    )
                fig_tsne.update_layout(
                    title="t-SNE Feature Cluster (validation set)",
                    paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG,
                    font=dict(color=TEXT_COLOR),
                    xaxis=dict(
                        title="t-SNE Dim 1",
                        gridcolor=GRID_COLOR,
                        zeroline=False,
                        showticklabels=False,
                    ),
                    yaxis=dict(
                        title="t-SNE Dim 2",
                        gridcolor=GRID_COLOR,
                        zeroline=False,
                        showticklabels=False,
                    ),
                    legend=dict(
                        bgcolor=LEGEND_BG,
                        bordercolor=LEGEND_BORDER,
                        itemsizing="constant",
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=-0.08,
                        yanchor="top",
                        entrywidth=170,
                        entrywidthmode="pixels",
                    ),
                    margin=dict(l=10, r=10, t=50, b=90),
                    height=600,
                )
                render_plotly(fig_tsne, 600, "fig_tsne")

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
                            CLASS_COLORS.get(c, TEXT_COLOR) for c in dist["Class"]
                        ],
                        text=dist["Count"],
                        textposition="outside",
                    )
                )
                fig_dist.update_layout(
                    paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG,
                    font=dict(color=TEXT_COLOR),
                    xaxis=dict(gridcolor=GRID_COLOR),
                    yaxis=dict(gridcolor=GRID_COLOR),
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=240,
                )
                render_plotly(fig_dist, 240, "fig_dist")
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
                        f'<p style="color:{TEXT_COLOR};margin:4px 0;">'
                        f'{CLASS_DESC.get(pred_class, "")}</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<p style="color:{ACCENT_BLUE};font-size:2rem;font-weight:700;margin:0;">'
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
                        paper_bgcolor=PAPER_BG,
                        plot_bgcolor=PLOT_BG,
                        font=dict(color=TEXT_COLOR),
                        xaxis=dict(title="Window Index", gridcolor=GRID_COLOR),
                        yaxis=dict(
                            title="Probability",
                            tickformat=".0%",
                            gridcolor=GRID_COLOR,
                            range=[0, 1],
                        ),
                        legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BORDER),
                        margin=dict(l=10, r=10, t=20, b=10),
                        height=280,
                    )
                    render_plotly(fig2, 280, "inference_probs")
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
