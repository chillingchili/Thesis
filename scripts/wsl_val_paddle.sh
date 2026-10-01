#!/bin/bash
# Runs inside WSL: validate float best.pt vs int8 TFLite on the paddle train
# split so quantization deltas are measured on identical data.
set -eo pipefail
BASE="/mnt/c/Users/hibye/Documents/ACTUAL PROJECTS/Thesis"

export OMP_NUM_THREADS=6
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2

cat > /tmp/paddle_train.yaml <<EOF
path: $BASE/datasets/paddle_ft
train: images/train
val: images/train
names:
  0: paddle
EOF

python3 - "$BASE" <<'PYEOF'
import sys
from ultralytics import YOLO

base = sys.argv[1]
data = "/tmp/paddle_train.yaml"
jobs = [
    ("float", f"{base}/runs/detect/paddle_ft/weights/best.pt"),
    ("int8", f"{base}/runs/detect/paddle_ft/weights/best_w8a32.tflite"),
]
for name, w in jobs:
    print(f"== val {name} ==", flush=True)
    m = YOLO(w)
    r = m.val(data=data, imgsz=640, batch=16, device="cpu",
              verbose=False, save=False, plots=False)
    print(f"RESULT {name}: P={r.box.mp:.4f} R={r.box.mr:.4f} "
          f"mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}", flush=True)
print("VAL_DONE", flush=True)
PYEOF
