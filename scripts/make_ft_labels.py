"""Generate YOLO-pose labels for datasets/court_ft by projecting the frozen
calibration homography onto every extracted frame, then assemble train/val
with a subset of original broadcast images mixed in.

usage: python scripts/make_ft_labels.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from balltrack_pipeline import KPT_COURT_FT  # noqa: E402

FT = ROOT / "datasets" / "court_ft"
ORIG = ROOT / "datasets" / "court"
N_MIX_TRAIN = 300
N_MIX_VAL = 40
VAL_EVERY = 7
PAD = 0.015


def label_for(img_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    H = np.array(json.loads((ROOT / "outputs" / "court_calib.json").read_text())["H_canonical_to_image"])
    k = cv2.perspectiveTransform(KPT_COURT_FT.reshape(-1, 1, 2), H).reshape(-1, 2)
    nx, ny = k[:, 0] / w, k[:, 1] / h
    x0, x1 = nx.min() - PAD, nx.max() + PAD
    y0, y1 = ny.min() - PAD, ny.max() + PAD
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cols = [0, cx, cy, x1 - x0, y1 - y0]
    for x, y in zip(nx, ny):
        cols += [x, y, 2.0]
    out_path.write_text(" ".join(f"{c:.6f}" for c in cols) + "\n")


def main() -> None:
    shutil.copyfile(ROOT / "outputs" / "court_calib_f3000.json", ROOT / "outputs" / "court_calib.json")

    for split in ("train", "val"):
        (FT / "images" / split).mkdir(parents=True, exist_ok=True)
        (FT / "labels" / split).mkdir(parents=True, exist_ok=True)

    imgs = sorted((FT / "images" / "train").glob("*.jpg"))
    to_val = []
    for i, p in enumerate(imgs):
        lp = FT / "labels" / "train" / (p.stem + ".txt")
        label_for(p, lp)
        if i % VAL_EVERY == 0:
            to_val.append(p)
    for p in to_val:
        p.rename(FT / "images" / "val" / p.name)
        (FT / "labels" / "train" / (p.stem + ".txt")).rename(FT / "labels" / "val" / (p.stem + ".txt"))
    n_train, n_val = len(list((FT / "images" / "train").glob("*.jpg"))), len(to_val)
    print(f"indoor: train={n_train} val={n_val}")

    rng = random.Random(0)
    for split, n in (("train", N_MIX_TRAIN), ("val", N_MIX_VAL)):
        pool = list((ORIG / "images" / split).glob("*.jpg"))
        if not pool:
            pool = list((ORIG / "images" / split).glob("*.png"))
        for p in rng.sample(pool, min(n, len(pool))):
            dst = FT / "images" / split / p.name
            if dst.exists():
                continue
            shutil.copyfile(p, dst)
            src_lbl = ORIG / "labels" / split / (p.stem + ".txt")
            shutil.copyfile(src_lbl, FT / "labels" / split / (p.stem + ".txt"))
    for split in ("train", "val"):
        ni = len(list((FT / "images" / split).glob("*.*")))
        nl = len(list((FT / "labels" / split).glob("*.txt")))
        print(f"{split}: images={ni} labels={nl}")

    (FT / "data.yaml").write_text(
        "path: datasets/court_ft\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: court\n"
        "kpt_shape:\n"
        "- 14\n"
        "- 3\n"
    )
    print("wrote datasets/court_ft/data.yaml")


if __name__ == "__main__":
    main()
