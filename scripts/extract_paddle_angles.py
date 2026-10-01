"""Paddle-angle stream (thesis Section 4.3.2).

Per frame: YOLOv26s detections -> pick box nearest the right wrist (<=300px;
highest conf if wrist unknown) -> expanded crop -> Otsu contour (both
polarities, best coverage; rect must cover 25-90% of the crop); if that fails
or is axis-aligned, a Canny-edge fallback takes a minAreaRect over all edge
pixels (20-96% fill) -> cv2.minAreaRect -> 90-degree axis ambiguity
resolved by aspect ratio -> long-axis orientation in [0, 180). Exactly 0 or 90
(axis-aligned quantization artifacts) are treated as undeterminable and set to
None.

Per clip: contact frame = peak right-wrist speed (trim_after_contact
heuristic, shared with keypoint pipeline); the clip scalar is the median
angle over contact +/- 5 frames (falls back to the whole series if the
window is empty).

Writes <out>/<clip_id>.json (full series + scalar), <out>/summary.csv, and
optionally contact-frame overlays to <out>/peek/ for visual verification.

Usage:
  python scripts/extract_paddle_angles.py \
      --videos data/training/raw/coach data/training/raw/beg1 data/training/raw/beg2 \
      --keypoints data/training/keypoints \
      --weights runs/detect/paddle_yolo26s/weights/best.pt \
      --out outputs/paddle_angles/train

  # smoke test on a few clips with contact-frame overlays
  python scripts/extract_paddle_angles.py ... --max-clips 5 --peek 5
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trim_after_contact import RIGHT_WRIST, estimate_contact_frame  # noqa: E402

SERVE_RE = re.compile(r"(drive|lob|topspin)", re.I)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
WRIST_MAX_PX = 300.0


def find_videos(roots):
    out = []
    for root in roots:
        root = Path(root)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS:
                out.append(p)
    return out


def group_of(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for key in ("coach", "beg1", "beg2", "beginner3", "beginner4"):
        if any(key in p for p in parts):
            return {"coach": "CoachA", "beg1": "Beg1", "beg2": "Beg2",
                    "beginner3": "Beg3", "beginner4": "Beg4"}[key]
    return "unknown"


def long_axis_angle(rect) -> float:
    (_, _), (w, h), ang = rect
    a = (ang % 180.0) if w >= h else ((ang - 90.0) % 180.0)
    if a >= 180.0 - 1e-9 or abs(a) < 1e-9:
        a = 0.0
    return float(a)


def contour_rect(crop: np.ndarray):
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None
    _, th1 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best = None
    best_score = 0.0
    best_meta = None
    for th_i, th in enumerate((th1, th2)):
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 16:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _ = rect
            rect_area = rw * rh
            fill_crop = rect_area / (h * w)
            if fill_crop < 0.25 or fill_crop > 0.90:
                continue
            cover = area / rect_area
            if cover < 0.5:
                continue
            centered = 1.0 - min(
                1.0,
                ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5 / (max(w, h) / 2),
            )
            score = cover * (0.7 + 0.3 * centered)
            if score > best_score:
                best_score = score
                best = rect
                best_meta = {"fill_crop": round(fill_crop, 3),
                             "cover": round(cover, 3),
                             "polarity": th_i,
                             "score": round(score, 3)}
    if best is not None:
        return best, best_meta
    return None


def edge_rect(crop: np.ndarray):
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) < 30:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), _ = rect
    fill_crop = (rw * rh) / (h * w)
    if fill_crop < 0.20 or fill_crop > 0.96:
        return None
    return rect, {"fill_crop": round(fill_crop, 3),
                  "cover": round(len(xs) / (rw * rh), 3),
                  "polarity": 2,
                  "score": 0.0}


def crop_angle(crop: np.ndarray):
    """(angle, rect, meta, status): status ok | deg | no_contour."""
    got = contour_rect(crop)
    if got is not None:
        angle = long_axis_angle(got[0])
        if angle not in (0.0, 90.0):
            return angle, got[0], got[1], "ok"
    eg = edge_rect(crop)
    if eg is not None:
        angle = long_axis_angle(eg[0])
        if angle not in (0.0, 90.0):
            return angle, eg[0], eg[1], "ok"
        return None, eg[0], eg[1], "deg"
    if got is not None:
        return None, got[0], got[1], "deg"
    return None, None, None, "no_contour"


def expand_box(box, shape, pad=0.12):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(shape[1], int(x2))
    y2 = min(shape[0], int(y2))
    return x1, y1, x2, y2


def pick_det(res, frame_shape, kpts, frame_idx):
    dets = list(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()))
    if not dets:
        return None, None
    H, W = frame_shape
    if kpts is not None and frame_idx < kpts.shape[0]:
        w = kpts[frame_idx, RIGHT_WRIST]
        if not np.isnan(w).any():
            wx, wy = float(w[0]) * W, float(w[1]) * H
            best = None
            best_d = None
            for b, c in dets:
                cx, cy = (float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2
                d = ((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5
                if best_d is None or d < best_d:
                    best_d = d
                    best = (b, float(c))
            if best_d <= WRIST_MAX_PX:
                return best
            return None, None
    b, c = max(dets, key=lambda t: float(t[1]))
    return b, float(c)


def process_clip(video: Path, keypoints_dir: Path, model, peek_dir=None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    npy = keypoints_dir / (video.stem + ".npy")
    kpts = np.load(npy) if npy.exists() else None
    frames_angles = []
    confs = []
    n_deg = 0
    peek_drawn = False
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = model.predict(frame, imgsz=640, conf=0.25, verbose=False, device=0)[0]
        angle = None
        conf = None
        if res.boxes is not None and len(res.boxes) > 0:
            b, conf = pick_det(res, gray.shape, kpts, i)
            if b is not None:
                x1, y1, x2, y2 = expand_box(b, gray.shape)
                angle, _rect, _meta, status = crop_angle(gray[y1:y2, x1:x2])
                if status == "deg":
                    n_deg += 1
        frames_angles.append(angle)
        confs.append(conf)
    cap.release()
    n = i + 1

    contact_frame = None
    if kpts is not None and kpts.shape[0] >= 3:
        contact_frame = estimate_contact_frame(kpts)
    if contact_frame is None:
        contact_frame = int(0.7 * n)

    valid = [(f, a) for f, a in enumerate(frames_angles) if a is not None]
    win = [a for f, a in valid if abs(f - contact_frame) <= 5]
    n_window = len(win)
    used = "window"
    if not win:
        win = [a for _, a in valid]
        used = "full_series"
    contact_angle = float(np.median(win)) if win else None

    if peek_dir is not None and not peek_drawn:
        peek_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(contact_frame, max(n - 1, 0)))
        ok, frame = cap.read()
        cap.release()
        if ok:
            res = model.predict(frame, imgsz=640, conf=0.25, verbose=False, device=0)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                b = res.boxes.xyxy[int(res.boxes.conf.argmax())].cpu().numpy().astype(int)
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{video.stem} f{contact_frame} ang={contact_angle}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imwrite(str(peek_dir / f"{video.stem}.jpg"), frame)
            peek_drawn = True

    return {
        "clip_id": video.stem,
        "video": str(video),
        "group": group_of(video),
        "serve_type": (SERVE_RE.search(video.stem).group(1).lower()
                       if SERVE_RE.search(video.stem) else None),
        "n_frames": n,
        "fps": fps,
        "contact_frame": contact_frame,
        "n_valid": len(valid),
        "n_window": n_window,
        "n_deg": n_deg,
        "scalar_rule": used,
        "contact_angle": contact_angle,
        "frames": list(range(n)),
        "angles": frames_angles,
        "confs": confs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--peek", type=int, default=0, help="dump contact-frame overlays for first N clips")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    videos = find_videos(args.videos)
    if args.max_clips:
        videos = videos[: args.max_clips]
    print(f"{len(videos)} clips from {args.videos}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    peek_dir = out / "peek" if args.peek else None

    rows = []
    n_peek = 0
    row_keys = ("clip_id", "group", "serve_type", "n_frames",
                "contact_frame", "n_valid", "scalar_rule", "contact_angle")
    for idx, video in enumerate(videos, 1):
        existing = out / f"{video.stem}.json"
        if existing.exists():
            prev = json.loads(existing.read_text(encoding="utf-8"))
            rows.append({k: prev[k] for k in row_keys})
            continue
        rec = process_clip(video, Path(args.keypoints), model,
                           peek_dir=peek_dir if (args.peek and n_peek < args.peek) else None)
        if rec is None:
            print(f"[{idx}/{len(videos)}] {video.name}: UNREADABLE", flush=True)
            continue
        if peek_dir is not None and (out / "peek" / f"{video.stem}.jpg").exists():
            n_peek += 1
        (out / f"{rec['clip_id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        rows.append({k: rec[k] for k in ("clip_id", "group", "serve_type", "n_frames",
                                         "contact_frame", "n_valid", "scalar_rule",
                                         "contact_angle")})
        print(f"[{idx}/{len(videos)}] {rec['clip_id']}: f={rec['n_frames']} "
              f"contact={rec['contact_frame']} valid={rec['n_valid']} "
              f"angle={rec['contact_angle']}", flush=True)

    fields = list(rows[0].keys()) if rows else []
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"DONE -> {out}/summary.csv ({len(rows)} clips)", flush=True)


if __name__ == "__main__":
    main()
