"""
visualize_tflite_inference.py

Draws paddle detections from float best.pt vs w8a32 TFLite on the same
images, side by side, into one montage. Run inside WSL (tflite is Linux-only).

Usage:
  python3 scripts/visualize_tflite_inference.py [--n 8] [--seed 11]
Output:
  outputs/paddle_tflite_inference/montage.jpg
"""
import argparse
import glob
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[1]
FLOAT_W = BASE / "runs/detect/paddle_ft/weights/best.pt"
TFLITE = BASE / "runs/detect/paddle_ft/weights/best_w8a32.tflite"
OUT = BASE / "outputs/paddle_tflite_inference"
PANEL_W, PANEL_H = 480, 270
COLS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(img, boxes, confs, color, tag):
    vis = img.copy()
    for (x0, y0, x1, y1), c in zip(boxes, confs):
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(vis, p0, p1, color, 2)
        cv2.putText(vis, f"{c:.2f}", (p0[0], max(14, p0[1] - 5)),
                    FONT, 0.5, color, 2, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(vis, tag, (6, 17), FONT, 0.55, color, 2, cv2.LINE_AA)
    vis = cv2.resize(vis, (PANEL_W, PANEL_H))
    return vis


def get(model, path):
    r = model.predict(path, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
    if r.boxes is None:
        return [], []
    return ([tuple(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()],
            [float(c) for c in r.boxes.conf.cpu().numpy()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    images = sorted(glob.glob(str(BASE / "datasets/paddle_ft/images/train/*.jpg")))
    random.Random(args.seed).shuffle(images)
    images = images[: args.n]

    fl, q = YOLO(str(FLOAT_W)), YOLO(str(TFLITE))
    rows = []
    for path in images:
        name = Path(path).stem
        img = cv2.imread(path)
        fb, fc = get(fl, path)
        tb, tc = get(q, path)
        rows.append(draw(img, fb, fc, (255, 0, 0), f"FLOAT {name} ({len(fb)})"))
        rows.append(draw(img, tb, tc, (0, 255, 0), f"W8A32 {name} ({len(tb)})"))

    while len(rows) % COLS:
        rows.append(np.zeros((PANEL_H, PANEL_W, 3), np.uint8))
    grid = [rows[i:i + COLS] for i in range(0, len(rows), COLS)]
    title = np.zeros((40, PANEL_W * COLS, 3), np.uint8)
    cv2.putText(title,
                f"paddle inference: float (blue) vs w8a32 tflite (green), conf>=0.25, "
                f"count in tag  |  n={len(images)} train imgs",
                (10, 28), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    body = np.vstack([np.hstack(r) for r in grid])
    out = np.vstack([title, body])
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "montage.jpg"
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {dst} ({out.shape[1]}x{out.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
