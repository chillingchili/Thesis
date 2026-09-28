#!/bin/bash
# Runs inside WSL: installs deps to user site, exports int8 LiteRT models.
set -eo pipefail
BASE="/mnt/c/Users/hibye/Documents/ACTUAL PROJECTS/Thesis"

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

echo "== $(date) exporting =="
python3 - "$BASE" <<'PYEOF'
import sys, yaml
from ultralytics import YOLO

base = sys.argv[1]
jobs = [
    ("ball", f"{base}/runs/detect/ball_yolo26s/weights/best.pt", f"{base}/datasets/ball"),
    ("court", f"{base}/runs/pose/court_yolo26s/weights/best.pt", f"{base}/datasets/court"),
]
for name, w, ddir in jobs:
    data = yaml.safe_load(open(f"{ddir}/data.yaml"))
    data["path"] = ddir
    tmp = f"/tmp/{name}_data.yaml"
    yaml.safe_dump(data, open(tmp, "w"))
    print(f"== exporting {name} ==", flush=True)
    m = YOLO(w)
    p = m.export(format="litert", int8=True, imgsz=640, data=tmp)
    print(f"{name} -> {p}", flush=True)
print("ALL_DONE", flush=True)
PYEOF
echo "== $(date) finished =="
