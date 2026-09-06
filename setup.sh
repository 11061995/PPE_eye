#!/usr/bin/env bash
# setup.sh — one-shot environment for the TRAINING box (x86_64 + NVIDIA GPU).
# For the Jetson, follow requirements-jetson.txt by hand. Do not run this there.
#
#   bash setup.sh            # auto-detect CUDA, install, verify
#   bash setup.sh cu121      # force a specific torch CUDA build
#   bash setup.sh cpu        # CPU-only (fine for split_no_leak.py, not training)

set -euo pipefail
CUDA_TAG="${1:-auto}"
VENV="${VENV:-.venv}"

echo "=== PPE-EYE reproduction: environment setup ==="

# ---------------------------------------------------------------- python version
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "[py]   python $PYV"
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)' || {
  echo "[fail] need python >= 3.9"; exit 1; }

# ---------------------------------------------------------------- detect CUDA
if [ "$CUDA_TAG" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRV_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "")
    echo "[gpu]  driver supports CUDA ${DRV_CUDA:-unknown}"
    case "${DRV_CUDA%%.*}" in
      13|14) CUDA_TAG="default" ;;   # PyPI torch already ships cu13
      12)    CUDA_TAG="cu121" ;;
      11)    CUDA_TAG="cu118" ;;
      *)     CUDA_TAG="default" ;;
    esac
  else
    echo "[gpu]  no nvidia-smi found -> CPU only."
    echo "       You can still run split_no_leak.py, but training will be unusable."
    CUDA_TAG="cpu"
  fi
fi
echo "[gpu]  torch build: $CUDA_TAG"

# ---------------------------------------------------------------- venv
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  echo "[venv] created $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip wheel

# ---------------------------------------------------------------- torch first
# Installing torch explicitly BEFORE ultralytics stops pip from silently
# resolving a build that mismatches the driver.
case "$CUDA_TAG" in
  cpu)     IDX="https://download.pytorch.org/whl/cpu" ;;
  cu118)   IDX="https://download.pytorch.org/whl/cu118" ;;
  cu121)   IDX="https://download.pytorch.org/whl/cu121" ;;
  default) IDX="" ;;
  *)       IDX="https://download.pytorch.org/whl/$CUDA_TAG" ;;
esac
echo "[pip]  installing torch ..."
if [ -n "$IDX" ]; then
  pip install -q torch torchvision --index-url "$IDX"
else
  pip install -q torch torchvision
fi

# ---------------------------------------------------------------- rest
echo "[pip]  installing ultralytics + export tooling ..."
pip install -q -r requirements-train.txt

# ---------------------------------------------------------------- verify
echo
echo "=== verification ==="
python - <<'EOF'
import sys
ok = True
try:
    import torch
    cu = torch.cuda.is_available()
    print(f"torch        {torch.__version__:<12} cuda={cu}")
    if cu:
        print(f"  device     {torch.cuda.get_device_name(0)}")
        print(f"  vram       {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("  WARNING: CUDA unavailable. Training YOLO11x on CPU is not viable.")
        print("  Fix this before proceeding, or the whole reproduction is pointless.")
except Exception as e:
    print("torch        FAILED:", e); ok = False
for mod, label in [("ultralytics","ultralytics"), ("cv2","opencv"),
                   ("numpy","numpy"), ("yaml","pyyaml"), ("PIL","pillow"),
                   ("onnx","onnx")]:
    try:
        m = __import__(mod)
        print(f"{label:<12} {getattr(m,'__version__','ok')}")
    except Exception as e:
        print(f"{label:<12} FAILED: {e}"); ok = False
sys.exit(0 if ok else 1)
EOF

echo
echo "=== weight download smoke test ==="
python - <<'EOF'
# Confirms Ultralytics can reach its asset host. yolov10*.pt is the one most
# likely to 404 — the v10 configs ship with ultralytics but the pretrained
# checkpoints are hosted separately. If it fails, drop the two yolo10 rows from
# train.py's grid; they are not the interesting cells anyway.
from ultralytics import YOLO
for w in ("yolo11n.pt", "yolo11s.pt", "yolov10n.pt"):
    try:
        YOLO(w)
        print(f"  {w:<14} ok")
    except Exception as e:
        print(f"  {w:<14} FAILED ({type(e).__name__}) -> drop from grid if yolov10")
EOF

cat <<'EOF'

=== next ===
  source .venv/bin/activate

  # 1. build the two splits (the actual experiment)
  python split_no_leak.py --src /data/ppe_raw --dst /data/chvg_clean --val 0.15 --test 0.15
  python split_no_leak.py --src /data/ppe_raw --dst /data/chvg_naive --hamming 999

  # 2. quick smoke model, ~40 min on one GPU
  yolo detect train model=yolo11s.pt data=/data/chvg_clean/data.yaml epochs=50 imgsz=640 batch=16

  # 3. live look
  python live_check.py --weights runs/detect/train/weights/best.pt --source 0 --debug-assoc

Record two numbers from step 3: the UNRESOLVED percentage and the p95 latency.
Everything later gets compared against those.
EOF
