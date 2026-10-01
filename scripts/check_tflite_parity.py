"""
check_tflite_parity.py

Compares float .pt vs int8 .tflite paddle detections on identical images
(IoU matching at conf 0.25). Run inside WSL (tflite runtime is Linux-only).

Usage:
  python3 scripts/check_tflite_parity.py [--n 40]
"""
import argparse
import glob
import random
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[1]
FLOAT_W = BASE / "runs/detect/paddle_ft/weights/best.pt"
TFLITE_CANDIDATES = sorted(glob.glob(str(BASE / "runs/detect/paddle_ft/weights/best*.tflite")))


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    images = sorted(glob.glob(str(BASE / "datasets/paddle_ft/images/train/*.jpg")))
    random.Random(7).shuffle(images)
    images = images[: args.n]

    if not TFLITE_CANDIDATES:
        raise SystemExit("no best*.tflite found in paddle_ft/weights")
    print(f"tflite={TFLITE_CANDIDATES[0]}")

    fl = YOLO(str(FLOAT_W))
    q = YOLO(TFLITE_CANDIDATES[0])

    matched = 0
    float_only = 0
    int8_only = 0
    conf_deltas = []
    ious = []
    for im in images:
        rf = fl.predict(im, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
        rq = q.predict(im, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
        fb = ([] if rf.boxes is None else
              [tuple(map(float, b)) for b in rf.boxes.xyxy.cpu().numpy()])
        qb = ([] if rq.boxes is None else
              [tuple(map(float, b)) for b in rq.boxes.xyxy.cpu().numpy()])
        fc = ([] if rf.boxes is None else
              [float(c) for c in rf.boxes.conf.cpu().numpy()])
        qc = ([] if rq.boxes is None else
              [float(c) for c in rq.boxes.conf.cpu().numpy()])

        used = set()
        for i, b in enumerate(fb):
            best, best_iou = None, 0.5
            for j, c in enumerate(qb):
                if j in used:
                    continue
                v = iou(b, c)
                if v >= best_iou:
                    best, best_iou = j, v
            if best is None:
                float_only += 1
            else:
                used.add(best)
                matched += 1
                ious.append(best_iou)
                conf_deltas.append(abs(fc[i] - qc[best]))
        int8_only += len(qb) - len(used)

    n = max(1, matched)
    print(f"images={len(images)} matched={matched} float_only={float_only} "
          f"int8_only={int8_only}")
    if matched:
        print(f"mean_iou={sum(ious)/len(ious):.4f} "
              f"mean_abs_conf_delta={sum(conf_deltas)/len(conf_deltas):.4f}")
    print("PARITY_DONE", flush=True)


if __name__ == "__main__":
    main()
