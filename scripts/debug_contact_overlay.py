"""Diagnostic overlays: at the estimated contact frame, draw the full BlazePose
skeleton (16 = right wrist highlighted), every YOLO paddle detection with conf,
and the wrist pixel position -- to diagnose box selection vs keypoint quality.

Usage:
  python scripts/debug_contact_overlay.py \
      --videos data/training/raw/coach --keypoints data/training/keypoints \
      --weights runs/paddle_yolo26s/weights/best.pt \
      --out outputs/paddle_angles/debug --clips 5
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trim_after_contact import RIGHT_WRIST, estimate_contact_frame  # noqa: E402

EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
    (8, 9), (9, 10), (7, 11), (11, 12), (12, 13), (0, 14), (14, 15),
    (15, 16),
]


def draw(img, kpts_px):
    for a, b in EDGES:
        pa, pb = kpts_px[a], kpts_px[b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                 (255, 200, 0), 2, cv2.LINE_AA)
    for i, p in enumerate(kpts_px):
        if np.isnan(p).any():
            continue
        c = (0, 0, 255) if i == RIGHT_WRIST else (255, 255, 255)
        cv2.circle(img, tuple(p.astype(int)), 5 if i == RIGHT_WRIST else 3, c, -1)
    w = kpts_px[RIGHT_WRIST]
    if not np.isnan(w).any():
        cv2.putText(img, "Rwrist", (int(w[0]) + 8, int(w[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vids = []
    for root in args.videos:
        vids += sorted(Path(root).rglob("*.mp4"))
    vids = vids[: args.clips]

    for v in vids:
        npy = Path(args.keypoints) / (v.stem + ".npy")
        if not npy.exists():
            print(f"{v.stem}: no keypoints")
            continue
        kpts = np.load(npy)
        contact = estimate_contact_frame(kpts)
        cap = cv2.VideoCapture(str(v))
        cap.set(cv2.CAP_PROP_POS_FRAMES, contact)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        H, W = frame.shape[:2]
        res = model.predict(frame, imgsz=640, conf=0.1, verbose=False, device=0)[0]
        n_det = 0
        if res.boxes is not None and len(res.boxes):
            for b, c in zip(res.boxes.xyxy.cpu().numpy(),
                            res.boxes.conf.cpu().numpy()):
                n_det += 1
                x1, y1, x2, y2 = b.astype(int)
                col = (0, 255, 0) if c >= 0.25 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{c:.2f}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        kpts_px = kpts[contact] * np.array([W, H], dtype=np.float32)
        draw(frame, kpts_px)
        w = kpts_px[RIGHT_WRIST]
        info = (f"{v.stem} f={contact} dets={n_det} "
                f"rwrist_px=({w[0]:.0f},{w[1]:.0f}) "
                f"rwrist_norm=({kpts[contact,RIGHT_WRIST,0]:.2f},"
                f"{kpts[contact,RIGHT_WRIST,1]:.2f})")
        print(info, flush=True)
        cv2.putText(frame, info, (10, H - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"{v.stem}_contact.jpg"), frame)


if __name__ == "__main__":
    main()
