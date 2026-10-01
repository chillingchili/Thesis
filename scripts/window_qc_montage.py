"""QC montage of contact-window frames: drawn axis on ok frames, fail reason otherwise.

Samples clips per group, reads contact +/-3 frames, runs the extraction gates,
and renders one grid per group: crop upscaling, minAreaRect + long axis when the
frame produced an angle, else the failing gate label.

Usage:
  python scripts/window_qc_montage.py --summary outputs/paddle_angles/ft_train/summary.csv \
      --videos data/training/raw/coach data/training/raw/beg1 data/training/raw/beg2 \
      --keypoints data/training/keypoints --weights runs/detect/paddle_ft/weights/best.pt \
      --out outputs/paddle_angles/ft_train/window_qc --per-group 6 --seed 7
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_paddle_angles import (  # noqa: E402
    crop_angle, expand_box, find_videos, pick_det,
)
from trim_after_contact import RIGHT_WRIST  # noqa: E402


def tile(img_gray, rect, angle, label, fail, size=300):
    h, w = img_gray.shape[:2]
    scale = min(size / max(h, w, 1), 8.0)
    up = cv2.resize(img_gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_NEAREST)
    up = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    if rect is not None:
        pts = np.round(cv2.boxPoints((rect[0], (rect[1][0] * scale, rect[1][1] * scale),
                                      rect[2]))).astype(np.int32)
        cv2.polylines(up, [pts], True, (255, 0, 0), 1)
        sides = sorted([(np.linalg.norm(pts[0] - pts[1]), pts[0], pts[1]),
                        (np.linalg.norm(pts[1] - pts[2]), pts[1], pts[2])],
                       key=lambda s: -s[0])
        _, a, b = sides[0]
        col = (0, 255, 0) if angle is not None and angle not in (0.0, 90.0) else (0, 0, 255)
        cv2.line(up, tuple(a), tuple(b), col, 3)
    canvas = np.zeros((size + 56, size, 3), np.uint8)
    oy = max((size - up.shape[0]) // 2, 0)
    ox = max((size - up.shape[1]) // 2, 0)
    hh = min(up.shape[0], canvas.shape[0] - oy)
    ww = min(up.shape[1], canvas.shape[1] - ox)
    canvas[oy:oy + hh, ox:ox + ww] = up[:hh, :ww]
    cv2.putText(canvas, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    if fail:
        cv2.putText(canvas, fail, (3, canvas.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 140, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--halfwin", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO

    rows = list(csv.DictReader(open(args.summary, encoding="utf-8")))
    rng = np.random.default_rng(args.seed)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    sample = []
    for g in sorted(groups):
        pool = groups[g]
        idx = rng.permutation(len(pool))[: args.per_group]
        sample += [pool[i] for i in idx]
    print(f"sampled {len(sample)} clips")

    paths = {p.stem: p for p in find_videos(args.videos)}
    model = YOLO(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_group = {}
    for r in sample:
        by_group.setdefault(r["group"], []).append(r)

    for g, clips in sorted(by_group.items()):
        tiles = []
        for r in clips:
            video = paths.get(r["clip_id"])
            if video is None:
                continue
            contact = int(r["contact_frame"])
            kpt = Path(args.keypoints) / f"{r['clip_id']}.npy"
            kpts = np.load(kpt) if kpt.exists() else None
            want = list(range(max(0, contact - args.halfwin),
                              contact + args.halfwin + 1))
            want_set = set(want)
            cap = cv2.VideoCapture(str(video))
            i = -1
            while want_set:
                ok, frame = cap.read()
                if not ok:
                    break
                i += 1
                if i not in want_set:
                    continue
                want_set.discard(i)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                res = model.predict(frame, imgsz=640, conf=0.25, verbose=False, device=0)[0]
                rel = i - contact
                head = f"{r['clip_id'][-18:]} f{rel:+d}"
                if res.boxes is None or len(res.boxes) == 0:
                    tiles.append(tile(gray, None, None, head, "no_det"))
                    continue
                b, _c = pick_det(res, gray.shape, kpts, i)
                if b is None:
                    tiles.append(tile(gray, None, None, head, "far_wrist"))
                    continue
                x1, y1, x2, y2 = expand_box(b, gray.shape)
                crop = gray[y1:y2, x1:x2]
                angle, rect, meta, status = crop_angle(crop)
                if status == "no_contour":
                    tiles.append(tile(crop, None, None, head, "no_contour"))
                elif status == "deg":
                    tiles.append(tile(crop, rect, None, head,
                                      f"deg fill={meta['fill_crop']} pol={meta['polarity']}"))
                else:
                    tiles.append(tile(crop, rect, angle, head,
                                      f"pol={meta['polarity']}"))
            cap.release()

        cols = 7
        rows_n = (len(tiles) + cols - 1) // cols
        th, tw = tiles[0].shape[:2]
        grid = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
        for k, t in enumerate(tiles):
            rr, cc = divmod(k, cols)
            grid[rr * th:(rr + 1) * th, cc * tw:(cc + 1) * tw] = t
        dst = out / f"window_qc_{g}.jpg"
        cv2.imwrite(str(dst), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{g}: {len(tiles)} tiles -> {dst}")


if __name__ == "__main__":
    main()
