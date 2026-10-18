"""
_gpu_bootstrap.py
=================
CPU bootstrap module.

Sets up environment variables for CPU execution.
"""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

# Auto-dispatch to project .venv if invoked with system python
_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

# Limit thread pools to keep CPU usage manageable
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["MKL_NUM_THREADS"]      = "4"

import onnxruntime as ort

# ── Execution providers (CPU only) ───────────────────────────────────────────
BEST_ORT_PROVIDERS = ["CPUExecutionProvider"]

# ── YOLO device ──────────────────────────────────────────────────────────────
YOLO_DEVICE = "cpu"

try:
    import torch as _torch
    TORCH_VERSION = _torch.__version__
except ImportError:
    TORCH_VERSION = "not installed"

if __name__ == "__main__":
    print("CPU Bootstrap Status")
    print(f"  ORT version       : {ort.__version__}")
    print(f"  ORT providers     : {ort.get_available_providers()}")
    print(f"  torch version     : {TORCH_VERSION}")
    print(f"  YOLO device       : {YOLO_DEVICE}")
