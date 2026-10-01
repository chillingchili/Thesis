"""Contact-window detection audit (thesis 4.3.2 pre-flight).

Runs the paddle detector only on frames around each clip's estimated contact
frame and records every box (xyxy + conf) together with the right-wrist pixel.
Purpose: measure how often detections near contact are trustworthy (near the
wrist, moving) vs false positives (static cart/net/shirt), before committing
to the angle-extraction protocol.

Outputs <out>/<clip_id>.json:
  {clip_id, contact, frames: [{f, wrist: [x,y]|null,
    boxes: [[x1,y1,x2,y2,conf], ...]}], n}

Usage:
  python scripts/audit_contact_windows.py \
      --videos data/training/raw/coach --keypoints data/training/keypoints \
      --weights runs/paddle_yolo26s/weights/best.pt \
      --out outputs/paddle_angles/audit --halfwin 10
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trim_after_contact import RIGHT_WRIST, estimate_contact_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--halfwin", type=int, default=10)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vids = []
    for root in args.videos:
        vids += sorted(Path(root).rglob("*.mp4"))
    print(f"{len(vids)} clips", flush=True)

    for idx, v in enumerate(vids, 1):
        npy = Path(args.keypoints) / (v.stem + ".npy")
        if not npy.exists():
            continue
        kpts = np.load(npy)
        contact = estimate_contact_frame(kpts)
        f0 = max(0, contact - args.halfwin)
        f1 = contact + args.halfwin
        cap = cv2.VideoCapture(str(v))
        frames = []
        for f in range(f0, min(f1 + 1, kpts.shape[0])):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]
            res = model.predict(frame, imgsz=640, conf=0.1, verbose=False, device=0)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes):
                for b, c in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.conf.cpu().numpy()):
                    boxes.append([round(float(v_), 1) for v_ in b] + [round(float(c), 3)])
            w = kpts[f, RIGHT_WRIST]
            wrist = [round(float(w[0]) * W, 1), round(float(w[1]) * H, 1)] \
                if not np.isnan(w).any() else None
            frames.append({"f": f, "wrist": wrist, "boxes": boxes})
        cap.release()
        rec = {"clip_id": v.stem, "contact": contact, "n_frames": kpts.shape[0],
               "frames": frames}
        (out / f"{v.stem}.json").write_text(json.dumps(rec), encoding="utf-8")
        if idx % 25 == 0 or idx == len(vids):
            print(f"[{idx}/{len(vids)}] last={v.stem} contact={contact}", flush=True)


if __name__ == "__main__":
    main()
