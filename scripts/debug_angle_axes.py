"""Diagnose paddle-angle degeneracy (clip scalars exactly 0 or 90).

At each sampled clip's contact frame: detect -> expand crop -> contour_rect,
then draw the winning minAreaRect + long axis on an upscaled crop and tally
metadata (border pass, fill_crop, polarity) against degenerate outputs.

Usage:
  python scripts/debug_angle_axes.py --summary outputs/paddle_angles/ft_train/summary.csv \
      --videos data/training/raw/coach data/training/raw/beg1 data/training/raw/beg2 \
      --keypoints data/training/keypoints --weights runs/detect/paddle_ft/weights/best.pt \
      --out outputs/paddle_angles/ft_train/axes_debug --per-class 8 --all
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_paddle_angles import contour_rect, expand_box, find_videos, long_axis_angle  # noqa: E402


def read_contact_frame(video: Path, contact: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    i = -1
    frame = None
    while i < contact:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        frame = f
    cap.release()
    return frame


def draw_tile(crop, rect, meta, angle, label, tile=340):
    h, w = crop.shape[:2]
    scale = max(2.0, tile / max(h, w, 1))
    up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    up = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    cx, cy = rect[0][0] * scale, rect[0][1] * scale
    pts = cv2.boxPoints((rect[0], (rect[1][0] * scale, rect[1][1] * scale), rect[2]))
    pts = np.round(pts).astype(np.int32)
    cv2.polylines(up, [pts], True, (255, 0, 0), 1)
    (p1, p2, p3, p4) = pts
    sides = [(np.linalg.norm(p1 - p2), p1, p2), (np.linalg.norm(p2 - p3), p2, p3)]
    sides.sort(key=lambda s: -s[0])
    _, a, b = sides[0]
    col = (0, 255, 0) if angle not in (0.0, 90.0) else (0, 0, 255)
    cv2.line(up, tuple(a), tuple(b), col, 3)
    canvas = np.zeros((tile + 60, tile, 3), np.uint8)
    oy = (tile + 60 - up.shape[0]) // 2
    ox = (tile - up.shape[1]) // 2
    oy = max(oy, 0)
    ox = max(ox, 0)
    canvas[oy:oy + up.shape[0], ox:ox + up.shape[1]] = up[:canvas.shape[0] - oy, :canvas.shape[1] - ox]
    cv2.putText(canvas, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    m = f"ang={angle:.1f} fill={meta['fill_crop']} cov={meta['cover']} pol={meta['polarity']}"
    cv2.putText(canvas, m, (4, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def bucket(scalar):
    if scalar == 0.0:
        return "zero"
    if scalar == 90.0:
        return "ninety"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--all", action="store_true", help="stats over every clip, no montage")
    args = ap.parse_args()

    from ultralytics import YOLO

    rows = list(csv.DictReader(open(args.summary, encoding="utf-8")))
    for r in rows:
        r["contact_frame"] = int(r["contact_frame"])
        r["scalar"] = float(r["contact_angle"]) if r["contact_angle"] else None
    paths = {p.stem: p for p in find_videos(args.videos)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.all:
        sample = rows
    else:
        by = defaultdict(list)
        for r in rows:
            if r["scalar"] is not None:
                by[bucket(r["scalar"])].append(r)
        sample = []
        for b in ("zero", "ninety", "other"):
            pool = sorted(by.get(b, []), key=lambda r: r["clip_id"])
            step = max(1, len(pool) // args.per_class)
            sample += pool[::step][: args.per_class]
        print(f"sampled {len(sample)} clips: "
              f"{Counter(bucket(r['scalar']) for r in sample)}")

    model = YOLO(args.weights)
    stats = defaultdict(Counter)
    tiles = []
    for idx, r in enumerate(sample, 1):
        video = paths.get(r["clip_id"])
        if video is None:
            stats["misc"]["no_video"] += 1
            continue
        frame = read_contact_frame(video, r["contact_frame"])
        if frame is None:
            stats["misc"]["unreadable"] += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = model.predict(frame, imgsz=640, conf=0.25, verbose=False, device=0)[0]
        bkt = bucket(r["scalar"]) if r["scalar"] is not None else "?"
        if res.boxes is None or len(res.boxes) == 0:
            stats[bkt]["no_det"] += 1
            continue
        k = int(res.boxes.conf.argmax())
        b = res.boxes.xyxy[k].cpu().numpy()
        x1, y1, x2, y2 = expand_box(b, gray.shape)
        crop = gray[y1:y2, x1:x2]
        got = contour_rect(crop)
        if got is None:
            stats[bkt]["no_contour"] += 1
            continue
        rect, meta = got
        angle = long_axis_angle(rect)
        stats[bkt][f"angle={'deg' if angle in (0.0, 90.0) else 'cont'}"] += 1
        if meta["fill_crop"] >= 0.7:
            stats[bkt]["fill>=0.7"] += 1
        if not args.all:
            label = f"{r['clip_id']} scalar={r['scalar']:.1f} [{bkt}]"
            tiles.append(draw_tile(crop, rect, meta, angle, label))
        if idx % 50 == 0:
            print(f"[{idx}/{len(sample)}]", flush=True)

    for b in ("zero", "ninety", "other"):
        if stats[b]:
            n = sum(stats[b].values())
            print(f"{b:7s} n={n}: {dict(stats[b])}")
    if stats["misc"]:
        print("misc:", dict(stats["misc"]))

    if tiles:
        cols = 4
        rows_n = (len(tiles) + cols - 1) // cols
        th, tw = tiles[0].shape[:2]
        grid = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(tiles):
            r_, c_ = divmod(i, cols)
            grid[r_ * th:(r_ + 1) * th, c_ * tw:(c_ + 1) * tw] = t
        dst = out / "axes_debug_montage.jpg"
        cv2.imwrite(str(dst), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"montage -> {dst} ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
