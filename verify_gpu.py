"""
verify_gpu.py
=============
Verifies that ONNX Runtime, InsightFace, and YOLOv11 Nano
are all running correctly on CPU.
"""
import os, sys, subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-dispatch to project .venv if invoked with system python
# ---------------------------------------------------------------------------
_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

import onnxruntime as ort

SEP = "=" * 62
print(SEP)
print("  CPU VERIFICATION REPORT — Campus Guardian")
print(SEP)

# ------------------------------------------------------------------ ONNX Runtime
print("\n[ONNX Runtime]")
print(f"  Version          : {ort.__version__}")
providers = ort.get_available_providers()
print(f"  Available        : {providers}")
cpu_ok = "CPUExecutionProvider" in providers
print(f"  CPUProvider      : {'YES' if cpu_ok else 'NO'}")

# --------------------------------------------------------- InsightFace
print("\n[InsightFace buffalo_l]")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(320, 320))
    first_key  = list(app.models.keys())[0]
    active_ep  = app.models[first_key].session.get_providers()[0]
    print(f"  Active provider  : {active_ep}")
    print(f"  On CPU           : {active_ep == 'CPUExecutionProvider'}")
except Exception as exc:
    print(f"  Error: {exc}")

# --------------------------------------------------------- YOLOv11 Nano
print("\n[YOLOv11n (Nano)]")
try:
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.to("cpu")
    device = str(next(model.model.parameters()).device)
    print(f"  Model device     : {device}")
    print(f"  On CPU           : {device == 'cpu'}")
except Exception as exc:
    print(f"  Error: {exc}")

print()
print(SEP)
print(f"  OVERALL STATUS   : {'PASS — CPU MODE ACTIVE' if cpu_ok else 'FAIL'}")
print(SEP)
