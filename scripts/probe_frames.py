"""Targeted probe: detections at low conf near the wrist, and Otsu fill on close-up paddles.

For sampled clips with empty windows (or no_contour-heavy), prints per window
frame: best det conf + center distance to wrist at several conf thresholds,
and the Otsu/edge fill values when a contour is found.

Usage:
  python scripts/probe_frames.py --clip CoachA_Lob_039 CoachA_Lob_045 \
      --videos data/training/raw/coach --keypoints data/training/keypoints \
      --weights runs/detect/paddle_ft/weights/best.pt --contact 195 201 --halfwin 5
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_paddle_angles import (  # noqa: E402
    contour_rect, edge_rect, expand_box, find_videos, long_axis_angle,
)
from trim_after_contact import RIGHT_WRIST  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", nargs="+", required=True)
    ap.add_argument("--contact", nargs="+", type=int, required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--halfwin", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    paths = {p.stem: p for p in find_videos(args.videos)}
    for clip, contact in zip(args.clip, args.contact):
        video = paths.get(clip)
        kpts = np.load(Path(args.keypoints) / f"{clip}.npy")
        cap = cv2.VideoCapture(str(video))
        i = -1
        print(f"== {clip} contact={contact}")
        while i < contact + args.halfwin:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if abs(i - contact) > args.halfwin:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape
            res = model.predict(frame, imgsz=640, conf=0.05, verbose=False, device=0)[0]
            w = kpts[i, RIGHT_WRIST]
            wx, wy = float(w[0]) * W, float(w[1]) * H
            near = []
            if res.boxes is not None and len(res.boxes):
                for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    d = float(np.hypot(cx - wx, cy - wy))
                    near.append((float(c), d, b))
            near.sort(key=lambda t: -t[0])
            best = [(c, d) for c, d, _ in near[:3]]
            line = f"f{i - contact:+d} wrist=({wx:.0f},{wy:.0f}) top3(conf,dist)=" + \
                   " ".join(f"({c:.2f},{d:.0f})" for c, d in best)
            print(line)
            hit = [t for t in near if t[0] >= 0.10 and t[1] <= 300]
            if hit:
                c, d, b = max(hit, key=lambda t: t[0])
                x1, y1, x2, y2 = expand_box(b, gray.shape)
                crop = gray[y1:y2, x1:x2]
                h, w = crop.shape[:2]
                _, th1 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, th2 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                cands = []
                for pol, th in enumerate((th1, th2)):
                    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cc in cnts:
                        area = cv2.contourArea(cc)
                        if area < 16:
                            continue
                        rect = cv2.minAreaRect(cc)
                        fill = rect[1][0] * rect[1][1] / (h * w)
                        cover = area / max(1.0, rect[1][0] * rect[1][1])
                        cands.append((round(fill, 2), round(cover, 2),
                                      round(long_axis_angle(rect), 1), pol))
                blur_ = cv2.GaussianBlur(crop, (3, 3), 0)
                edges = cv2.Canny(blur_, 50, 150)
                ys, xs = np.nonzero(edges)
                efill = None
                eang = None
                if len(xs) >= 30:
                    er = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
                    efill = round(er[1][0] * er[1][1] / (h * w), 2)
                    eang = round(long_axis_angle(er), 1)
                top = sorted(cands, key=lambda t: -(t[1] * 0.7 + min(t[0], 1) * 0.3))[:4]
                print(f"  PICK conf={c:.2f} d={d:.0f} crop={w}x{h} "
                      f"otsu(fill,cov,ang,pol)={top} edge(fill,ang)=({efill},{eang})")
        cap.release()


if __name__ == "__main__":
    main()
