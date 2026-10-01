"""Export isolated 320px candidates; never overwrite the production 640px models."""
from pathlib import Path
import os
import shutil
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import yaml
from ultralytics import YOLO

base = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=320)
parser.add_argument("--quantize", default="8")
parser.add_argument("--ball-only", action="store_true")
args = parser.parse_args()
quantize = int(args.quantize) if args.quantize.isdigit() else args.quantize
jobs = [
    ("ball", "runs/detect/ball_yolo26s/weights/best.pt", {
        "path": str(base / "datasets/ball_cal"), "train": "images", "val": "images",
        "names": {0: "ball"},
    }),
    ("court", "runs/pose/court_ft/weights/best.pt", {
        "path": str(base / "datasets/court_ft"), "train": "images/val", "val": "images/val",
        "names": {0: "court"}, "kpt_shape": [14, 3],
    }),
]
for name, source, data in jobs:
    if args.ball_only and name != "ball":
        continue
    destination = base / "outputs/mobile_candidates" / f"{name}{args.size}_{args.quantize}"
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / "best.pt"
    shutil.copy2(base / source, weights)
    calibration = destination / "calibration.yaml"
    calibration.write_text(yaml.safe_dump(data))
    model = YOLO(weights)
    print(model.export(format="litert", quantize=quantize, imgsz=args.size, data=str(calibration)), flush=True)
