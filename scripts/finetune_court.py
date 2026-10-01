"""Fine-tune the court pose model on auto-labeled indoor footage (court_ft).

Warm-starts from runs/pose/court_yolo26s/weights/best_orig.pt (backup made
before training), writes to runs/pose/court_ft/ so the original run is intact.

usage: python scripts/finetune_court.py [epochs]
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMGSZ = 640
BATCH = 8


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    yml = ROOT / "datasets" / "court_ft" / "data.yaml"
    data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    data["path"] = str(ROOT / "datasets" / "court_ft")
    yml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    import torch
    print("cuda:", torch.cuda.is_available())

    model = YOLO(str(ROOT / "runs/pose/court_yolo26s/weights/best_orig.pt"))
    model.train(
        data=str(yml),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        name="court_ft",
        exist_ok=True,
        device=0,
    )
    print("FT best:", ROOT / "runs/pose/court_ft/weights/best.pt")


if __name__ == "__main__":
    main()
