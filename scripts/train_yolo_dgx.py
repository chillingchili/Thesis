"""Train YOLOv26s on the DGX (or any machine with ultralytics installed).

Usage:
  python train_yolo_dgx.py ball    # ball detection  -> runs/detect/ball_yolo26s/weights/best.pt
  python train_yolo_dgx.py court   # court keypoints -> runs/pose/court_yolo26s/weights/best.pt
  python train_yolo_dgx.py paddle  # paddle detect   -> runs/detect/paddle_yolo26s/weights/best.pt
  python train_yolo_dgx.py both

Self-locating: chdirs to this file's folder, so you can run it from anywhere.
Self-healing: rewrites data.yaml paths to absolute before training.

Offline note (DGX blocks github.com): keep these files NEXT TO this script
so nothing ever needs the network:
  yolo26s.pt        model weights for ball training
  yolo26s-pose.pt   model weights for court training
  yolo26n.pt        tiny model used by ultralytics' AMP self-check
"""
import os
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

EPOCHS = 100
IMGSZ = 640
BATCH = 32


def prepare(name: str) -> str:
    """Verify dataset exists, fix data.yaml path, return yaml path."""
    ddir = ROOT / "datasets" / name
    yml = ddir / "data.yaml"
    train_imgs = ddir / "images" / "train"
    val_imgs = ddir / "images" / "val"

    print(f"[{name}] ROOT = {ROOT}")
    for need in (yml, train_imgs, val_imgs):
        if not need.exists():
            sys.exit(
                f"MISSING: {need}\n"
                f"Expected layout next to train_yolo_dgx.py:\n"
                f"  yolo26s.pt\n"
                f"  datasets/{name}/data.yaml\n"
                f"  datasets/{name}/images/train/\n"
                f"  datasets/{name}/images/val/\n"
                f"(if images/ are missing, the WinSCP copy was incomplete)"
            )

    n_train = sum(1 for _ in train_imgs.glob("*.jpg"))
    n_val = sum(1 for _ in val_imgs.glob("*.jpg"))
    print(f"[{name}] images: train={n_train} val={n_val}")
    if n_train == 0 or n_val == 0:
        sys.exit(f"[{name}] images folders are empty - re-copy datasets/ with WinSCP")

    data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    data["path"] = str(ddir)  # absolute -> immune to cwd/yaml confusion
    yml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"[{name}] data.yaml path -> {data['path']}")
    return str(yml)


def gpu_check() -> None:
    try:
        import torch

        ok = torch.cuda.is_available()
        print(f"[gpu] torch.cuda.is_available() = {ok}")
        if not ok:
            print("[gpu] WARNING: training will run on CPU (slow). "
                  "Run: pip install --upgrade torch  (then retry)")
    except Exception as e:  # noqa: BLE001
        print(f"[gpu] torch check failed: {e}")


def train_ball() -> None:
    model = YOLO(str(ROOT / "yolo26s.pt"))
    model.train(
        data=prepare("ball"),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name="ball_yolo26s",
        exist_ok=True,
    )
    print("BALL best:", (ROOT / "runs/detect/ball_yolo26s/weights/best.pt"))


def train_court() -> None:
    # yolo26s-pose.pt found next to this script (or downloaded) - no GitHub needed
    try:
        model = YOLO(str(ROOT / "yolo26s-pose.pt")) if (ROOT / "yolo26s-pose.pt").exists() else YOLO("yolo26s-pose.pt")
    except Exception:
        print("yolo26s-pose.pt unavailable, warm-starting a pose model from yolo26s.pt")
        model = YOLO("yolo11-pose.yaml")  # pose task (needed for pose labels)
        model.load(str(ROOT / "yolo26s.pt"))
    model.train(
        data=prepare("court"),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name="court_yolo26s",
        exist_ok=True,
    )
    print("COURT best:", (ROOT / "runs/pose/court_yolo26s/weights/best.pt"))


def train_paddle() -> None:
    model = YOLO(str(ROOT / "yolo26s.pt"))
    model.train(
        data=prepare("paddle"),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name="paddle_yolo26s",
        exist_ok=True,
    )
    print("PADDLE best:", (ROOT / "runs/detect/paddle_yolo26s/weights/best.pt"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what not in ("ball", "court", "paddle", "both"):
        sys.exit(__doc__)
    gpu_check()
    if what in ("ball", "both"):
        train_ball()
    if what in ("court", "both"):
        train_court()
    if what == "paddle":
        train_paddle()
