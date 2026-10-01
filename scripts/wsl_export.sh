#!/bin/bash
# Runs inside WSL: installs deps to user site, exports int8 LiteRT models.
set -eo pipefail
BASE="/mnt/c/Users/hibye/Documents/ACTUAL PROJECTS/Thesis"

export OMP_NUM_THREADS=6
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2

try() {
  local i
  for i in 1 2 3 4 5; do
    "$@" && return 0
    echo "RETRY $i for: $*"
    sleep 5
  done
  echo "FAILED after retries: $*"
  return 1
}

echo "== $(date) setup =="
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "bootstrapping pip..."
  try curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python3 /tmp/get-pip.py -q --user --break-system-packages
fi

PIP="python3 -m pip install -q --user --break-system-packages --retries 10 --timeout 60"
try $PIP ultralytics
try $PIP tensorflow-cpu
try $PIP litert-torch ai-edge-litert

echo "== $(date) exporting =="
python3 - "$BASE" <<'PYEOF'
import sys, os, glob, yaml
from ultralytics import YOLO

base = sys.argv[1]
jobs = [
    ("ball", "detect", f"{base}/runs/detect/ball_yolo26s/weights/best.pt",
     {"path": f"{base}/datasets/ball_cal", "train": "images", "val": "images"}, 8),
    ("court", "pose", f"{base}/runs/pose/court_ft/weights/best.pt",
     {"path": f"{base}/datasets/court_ft", "train": "images/val", "val": "images/val",
      "names": {0: "court"}, "kpt_shape": [14, 3]}, 8),
    ("paddle", "detect", f"{base}/runs/detect/paddle_ft/weights/best.pt",
     {"path": f"{base}/datasets/paddle_ft", "train": "images/train", "val": "images/train",
      "names": {0: "paddle"}}, "w8a32"),
]
for name, kind, w, data, quantize in jobs:
    if glob.glob(f"{w.rsplit('/', 1)[0]}/best*.tflite"):
        print(f"== {name} already exported, skipping ==", flush=True)
        continue
    tmp = f"/tmp/{name}_cal.yaml"
    yaml.safe_dump(data, open(tmp, "w"))
    print(f"== exporting {name} ({kind}) quantize={quantize} ==", flush=True)
    m = YOLO(w)
    p = m.export(format="litert", quantize=quantize, imgsz=640, data=tmp)
    print(f"{name} -> {p}", flush=True)
print("ALL_DONE", flush=True)
PYEOF
echo "== $(date) finished =="
