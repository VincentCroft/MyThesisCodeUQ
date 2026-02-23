"""
PMU Fault Classifier — Entry Point
Usage:
    python app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).parent.resolve()
DASHBOARD = CODE_DIR / "GUI" / "dashboard.py"
PYTHON_EXE = sys.executable


def main():
    print("=" * 55)
    print("  ⚡ PMU Fault Classifier GUI")
    print("  Launching Streamlit dashboard...")
    print("=" * 55)
    cmd = [
        PYTHON_EXE,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD),
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "dark",
        "--theme.primaryColor",
        "#38bdf8",
        "--theme.backgroundColor",
        "#0f172a",
        "--theme.secondaryBackgroundColor",
        "#1e293b",
        "--theme.textColor",
        "#e2e8f0",
    ]
    try:
        subprocess.run(cmd, cwd=str(CODE_DIR), check=True)
    except KeyboardInterrupt:
        print("\n  GUI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"  ❌  Failed to start GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
