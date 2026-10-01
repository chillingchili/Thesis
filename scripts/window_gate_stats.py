"""Why do contact-window frames fail? Gate-level stats on ±3 window frames.

Per window frame: detection present? box near wrist (<=300px)? contour found
(fill 0.25-0.75)? angle degenerate (exact 0/90)? Tally the first failing gate
per frame, grouped by dataset, to see which rule empties the contact window.

Usage:
  python scripts/window_gate_stats.py --summary outputs/paddle_angles/ft_train/summary.csv \
      --videos data/training/raw/coach data/training/raw/beg1 data/training/raw/beg2 \
      --keypoints data/training/keypoints --weights runs/detect/paddle_ft/weights/best.pt \
      --sample 12 --seed 7
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_paddle_angles import (  # noqa: E402
    WRIST_MAX_PX, crop_angle, expand_box, find_videos, pick_det,
)
from trim_after_contact import RIGHT_WRIST  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--sample", type=int, default=12, help="clips per group x serve_type")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--halfwin", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO

    rows = list(csv.DictReader(open(args.summary, encoding="utf-8")))
    rng = np.random.default_rng(args.seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["group"], r["serve_type"])].append(r)
    sample = []
    for key in sorted(buckets):
        pool = buckets[key]
        idx = rng.permutation(len(pool))[: args.sample]
        sample += [pool[i] for i in idx]
    print(f"sampled {len(sample)} clips from {len(buckets)} group x type buckets")

    paths = {p.stem: p for p in find_videos(args.videos)}
    model = YOLO(args.weights)
    tally = defaultdict(Counter)
    ok_angles = defaultdict(list)

    for n_cl, r in enumerate(sample, 1):
        video = paths.get(r["clip_id"])
        if video is None:
            continue
        contact = int(r["contact_frame"])
        kpt_path = Path(args.keypoints) / f"{r['clip_id']}.npy"
        kpts = np.load(kpt_path) if kpt_path.exists() else None
        want = set(range(max(0, contact - args.halfwin), contact + args.halfwin + 1))
        cap = cv2.VideoCapture(str(video))
        i = -1
        while want:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i not in want:
                continue
            want.discard(i)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape
            g = r["group"]
            res = model.predict(frame, imgsz=640, conf=0.25, verbose=False, device=0)[0]
            if res.boxes is None or len(res.boxes) == 0:
                tally[g]["1_no_det"] += 1
                continue
            if kpts is not None and i < kpts.shape[0]:
                w = kpts[i, RIGHT_WRIST]
                if np.isnan(w).any():
                    tally[g]["2_no_wrist"] += 1
                    continue
                wx, wy = float(w[0]) * W, float(w[1]) * H
                centers = [((float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2)
                           for b, _ in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
                dmin = min(((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5 for cx, cy in centers)
                if dmin > WRIST_MAX_PX:
                    tally[g][f"3_far_wrist({int(dmin//100)*100})"] += 1
                    continue
            b, _c = pick_det(res, gray.shape, kpts, i)
            if b is None:
                tally[g]["3_far_wrist"] += 1
                continue
            x1, y1, x2, y2 = expand_box(b, gray.shape)
            angle, _rect, _meta, status = crop_angle(gray[y1:y2, x1:x2])
            if status == "no_contour":
                tally[g]["4_no_contour"] += 1
                continue
            if status == "deg":
                tally[g]["5_deg_reject"] += 1
                continue
            tally[g]["6_ok"] += 1
            ok_angles[g].append(round(angle, 1))
        cap.release()
        if n_cl % 20 == 0:
            print(f"[{n_cl}/{len(sample)}]", flush=True)

    totals = Counter()
    for g in sorted(tally):
        c = tally[g]
        tot = sum(c.values())
        totals.update(c)
        parts = " ".join(f"{k}={v}({100*v/tot:.0f}%)" for k, v in sorted(c.items()))
        print(f"{g:7s} n={tot}: {parts}")
    tot = sum(totals.values())
    parts = " ".join(f"{k}={v}({100*v/tot:.0f}%)" for k, v in sorted(totals.items()))
    print(f"ALL     n={tot}: {parts}")
    for g in sorted(ok_angles):
        v = ok_angles[g]
        if v:
            import statistics as st
            print(f"ok angles {g}: n={len(v)} med={st.median(v):.1f} "
                  f"p10={st.quantiles(v, n=10)[0]:.1f} p90={st.quantiles(v, n=10)[-1]:.1f}")


if __name__ == "__main__":
    main()
