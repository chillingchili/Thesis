"""Bootstrap paddle fine-tune labels by seed detection + CSRT tracking.

Per clip:
  1. contact frame from keypoints (peak right-wrist speed)
  2. seed = detector box in [contact-45,contact-18] U [contact+18,contact+45]
     with conf>=0.40 within 250px of the right wrist (sharp-ish frame away
     from peak motion blur), highest conf wins
  3. CSRT tracks the seed box from the seed frame across the contact zone;
     boxes are saved for the seed frame and for contact +/- 6 frames
  4. writes YOLO-format images/labels to a fine-tune dataset folder

Failed seeds/tracks are listed at the end (those clips need manual labels).

Outputs:
  <out>/images/train/<clip>_f<frame>.jpg
  <out>/labels/train/<clip>_f<frame>.txt
  <out>/seeds/<clip>.jpg            verification overlay of the seed box
  <out>/contact/<clip>.jpg          verification overlay of tracked contact boxes
  <out>/report.csv                  per-clip status

Usage:
  python scripts/bootstrap_paddle_labels.py \
      --videos data/training/raw/coach data/training/raw/beg1 data/training/raw/beg2 \
      --keypoints data/training/keypoints \
      --weights runs/paddle_yolo26s/weights/best.pt \
      --out datasets/paddle_ft --halfwin 6
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trim_after_contact import RIGHT_WRIST, estimate_contact_frame  # noqa: E402

SEED_NEAR, SEED_FAR = 18, 45
SEED_CONF, SEED_GATE = 0.40, 250.0
SEED_AREA_MAX = 0.06
ANCHOR_CONF, ANCHOR_IOU = 0.45, 0.25


def find_videos(roots):
    vids = []
    for root in roots:
        vids += sorted(Path(root).rglob("*.mp4"))
    return vids


def detect(model, frame, conf=0.30):
    res = model.predict(frame, imgsz=640, conf=conf, verbose=False, device=0)[0]
    out = []
    if res.boxes is not None and len(res.boxes):
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            out.append((b.astype(float), float(c)))
    return out


def pick_seed(model, cap, kpts, contact, H, W):
    ranges = list(range(max(0, contact - SEED_FAR), max(0, contact - SEED_NEAR + 1))) + \
             list(range(contact + SEED_NEAR, min(kpts.shape[0], contact + SEED_FAR + 1)))
    ranges.sort(key=lambda f: abs(f - contact))
    best = None
    for f in ranges:
        w = kpts[f, RIGHT_WRIST]
        if np.isnan(w).any():
            continue
        wp = np.array([w[0] * W, w[1] * H])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        for box, conf in detect(model, frame):
            if conf < SEED_CONF:
                continue
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw * bh > SEED_AREA_MAX * H * W or bw < 8 or bh < 8:
                continue
            if max(bw, bh) / min(bw, bh) > 5.0:
                continue
            c = (box[:2] + box[2:]) / 2
            if np.linalg.norm(c - wp) > SEED_GATE:
                continue
            if best is None or conf > best[2]:
                best = (f, box, conf, frame)
            if conf >= 0.55:
                return best
    return best


def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def track_span(seed_frame, seed_box, frames, cap, start_frame, model):
    tracker = cv2.TrackerCSRT_create()
    cap.set(cv2.CAP_PROP_POS_FRAMES, seed_frame)
    ok, img = cap.read()
    if not ok:
        return {}
    x, y, w, h = [float(v) for v in seed_box]
    tracker.init(img, (int(round(x)), int(round(y)), int(round(w)), int(round(h))))
    sw = w * h
    boxes = {seed_frame: (x, y, w, h)}
    step = 1 if frames and frames[-1] > seed_frame else -1
    f = seed_frame
    for target in frames:
        while f != target:
            f += step
            ok, img = cap.read()
            if not ok:
                return boxes
            ok, b = tracker.update(img)
            if not ok:
                return boxes
            bx, by, bw, bh = [float(v) for v in b]
            if bw * bh < 0.25 * sw or bw * bh > 4.0 * sw:
                return boxes
            if bx < -5 or by < -5 or bx + bw > img.shape[1] + 5 or by + bh > img.shape[0] + 5:
                return boxes
            for dbox, dconf in detect(model, img, conf=0.35):
                db = (float(dbox[0]), float(dbox[1]),
                      float(dbox[2] - dbox[0]), float(dbox[3] - dbox[1]))
                if dconf >= ANCHOR_CONF and iou((bx, by, bw, bh), db) >= ANCHOR_IOU \
                        and db[2] * db[3] <= 0.10 * img.shape[0] * img.shape[1]:
                    bx, by, bw, bh = db
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(img, (int(round(bx)), int(round(by)),
                                       int(round(bw)), int(round(bh))))
                    sw = bw * bh
                    break
            boxes[f] = (bx, by, bw, bh)
    return boxes


def save_label(img_dir, txt_dir, clip_id, f, frame, box):
    stem = f"{clip_id}_f{f:04d}"
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    H, W = frame.shape[:2]
    x, y, w, h = box
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    (txt_dir / f"{stem}.txt").write_text(
        f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}\n", encoding="utf-8")


def overlay(frame, box, label, color=(0, 255, 0)):
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--halfwin", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = Path(args.out)
    img_dir = out / "images" / "train"
    txt_dir = out / "labels" / "train"
    seed_dir = out / "seeds"
    contact_dir = out / "contact"
    for d in (img_dir, txt_dir, seed_dir, contact_dir):
        d.mkdir(parents=True, exist_ok=True)

    vids = find_videos(args.videos)
    if args.limit:
        vids = vids[: args.limit]
    print(f"{len(vids)} clips", flush=True)

    rows = []
    for idx, v in enumerate(vids, 1):
        clip_id = v.stem
        npy = Path(args.keypoints) / f"{clip_id}.npy"
        if not npy.exists():
            rows.append((clip_id, "no_keypoints", 0))
            continue
        kpts = np.load(npy)
        if kpts.shape[0] < 30:
            rows.append((clip_id, "too_short", 0))
            continue
        contact = estimate_contact_frame(kpts)
        cap = cv2.VideoCapture(str(v))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first = cap.read()
        if not ok:
            cap.release()
            rows.append((clip_id, "unreadable", 0))
            continue
        H, W = first.shape[:2]
        seed = pick_seed(model, cap, kpts, contact, H, W)
        if seed is None:
            cap.release()
            rows.append((clip_id, "seed_fail", 0))
            continue
        sf, sbox, sconf, sframe = seed
        sbox = (float(sbox[0]), float(sbox[1]),
                float(sbox[2] - sbox[0]), float(sbox[3] - sbox[1]))

        win = [f for f in range(max(0, contact - args.halfwin),
                                min(kpts.shape[0], contact + args.halfwin + 1))
               if f != sf]
        fwd = [f for f in win if f > sf]
        bwd = sorted([f for f in win if f < sf], reverse=True)
        boxes = {sf: tuple(sbox)}
        if fwd:
            boxes.update(track_span(sf, tuple(sbox), sorted(fwd), cap, sf, model))
        if bwd:
            boxes.update(track_span(sf, tuple(sbox), bwd, cap, sf, model))
        cap.release()

        save_label(img_dir, txt_dir, clip_id, sf, sframe, tuple(sbox))
        sv = overlay(sframe.copy(), tuple(sbox), f"{clip_id} seed c={sconf:.2f}")
        cv2.imwrite(str(seed_dir / f"{clip_id}.jpg"), sv)

        n_lab = 1
        contact_vis = None
        for f in sorted(boxes):
            if abs(f - contact) > args.halfwin:
                continue
            cap = cv2.VideoCapture(str(v))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            save_label(img_dir, txt_dir, clip_id, f, frame, boxes[f])
            n_lab += 1
            if abs(f - contact) <= 2:
                vis = overlay(frame.copy(), boxes[f], f"f={f}", (0, 255, 255))
                contact_vis = vis if contact_vis is None else np.hstack(
                    [cv2.resize(contact_vis, (480, 270)), cv2.resize(vis, (480, 270))])
        if contact_vis is not None:
            cv2.imwrite(str(contact_dir / f"{clip_id}.jpg"), contact_vis)

        rows.append((clip_id, "ok", n_lab))
        if idx % 25 == 0 or idx == len(vids):
            print(f"[{idx}/{len(vids)}] {clip_id} contact={contact} seed_f={sf} "
                  f"c={sconf:.2f} labels={n_lab}", flush=True)

    with open(out / "report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "status", "n_labels"])
        w.writerows(rows)
    ok = sum(1 for r in rows if r[1] == "ok")
    print(f"DONE ok={ok}/{len(rows)} -> {out}/report.csv", flush=True)
    fails = [r[0] for r in rows if r[1] != "ok"]
    if fails:
        print(f"failed ({len(fails)}): {', '.join(fails[:20])}", flush=True)


if __name__ == "__main__":
    main()
