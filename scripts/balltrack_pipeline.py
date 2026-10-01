"""Ball tracking + court homography pipeline.

process_video(): annotated side-by-side video (broadcast view | bird's-eye court),
ball landing detection in court feet, and the thesis 7-zone classification.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
BALL_WEIGHTS = ROOT / "runs/detect/ball_yolo26s/weights/best.pt"
COURT_WEIGHTS = ROOT / "runs/pose/court_ft/weights/best.pt"

# 14 keypoints -> canonical court coords in feet (x: 0..20 sideline-to-sideline,
# y: 0..44 far baseline to near baseline; net at y=22, NVZ lines at y=15/29)
KPT_COURT_FT = np.array(
    [
        (0, 0),      # 0 far baseline left
        (20, 0),     # 1 far baseline right
        (20, 44),    # 2 near baseline right
        (0, 44),     # 3 near baseline left
        (0, 15),     # 4 far NVZ left
        (20, 15),    # 5 far NVZ right
        (0, 29),     # 6 near NVZ left
        (20, 29),    # 7 near NVZ right
        (10, 0),     # 8 far baseline center
        (10, 22),    # 9 net center
        (10, 29),    # 10 near NVZ center (center-T)
        (10, 44),    # 11 near baseline center
        (0, 22),     # 12 net left post
        (20, 22),    # 13 net right post
    ],
    dtype=np.float32,
)

SKELETON_EDGES = [
    (0, 8), (8, 1), (0, 4), (4, 12), (12, 6), (6, 3),
    (1, 5), (5, 13), (13, 7), (7, 2), (4, 5),
    (6, 10), (10, 7), (12, 9), (9, 13), (3, 11), (11, 2),
]

BEV_SCALE = 14
BEV_MARGIN = 30
BEV_W = int(20 * BEV_SCALE + 2 * BEV_MARGIN)
BEV_H = int(44 * BEV_SCALE + 2 * BEV_MARGIN)

_MODELS = {}


def get_models() -> tuple[YOLO, YOLO]:
    if not _MODELS:
        _MODELS["ball"] = YOLO(str(BALL_WEIGHTS))
        _MODELS["court"] = YOLO(str(COURT_WEIGHTS))
    return _MODELS["ball"], _MODELS["court"]


def classify_zone(x_ft: float, y_ft: float) -> str:
    if x_ft < 0 or x_ft > 20:
        return "Fault-Wide"
    if y_ft < 0 or y_ft > 44:
        return "Fault-Long"
    if 15 <= y_ft <= 29:
        return "Fault-Short"
    if y_ft < 15:
        side = "Left" if x_ft < 10 else "Right"
        depth = "Deep" if y_ft < 7.5 else "Short"
    else:
        side = "Left" if x_ft < 10 else "Right"
        depth = "Deep" if y_ft > 36.5 else "Short"
    return f"{depth}-{side}"


V_OFF = 0


def ft_to_bev(x_ft: float, y_ft: float) -> tuple[int, int]:
    return (
        int(BEV_MARGIN + x_ft * BEV_SCALE),
        int(V_OFF + BEV_MARGIN + y_ft * BEV_SCALE),
    )


def draw_bev_base(height: int) -> np.ndarray:
    global V_OFF
    V_OFF = max((height - BEV_H) // 2, 0)
    img = np.full((height, BEV_W, 3), 28, dtype=np.uint8)

    def p(x, y):
        return ft_to_bev(x, y)

    cv2.rectangle(img, p(0, 0), p(20, 44), (240, 240, 240), 2)
    for y in (15.0, 29.0):
        cv2.line(img, p(0, y), p(20, y), (240, 240, 240), 2)
    cv2.line(img, p(10, 0), p(10, 15), (240, 240, 240), 2)
    cv2.line(img, p(10, 29), p(10, 44), (240, 240, 240), 2)
    kitchen = np.array([p(0, 15), p(20, 15), p(20, 29), p(0, 29)], np.int32)
    cv2.fillPoly(img, [kitchen], (44, 62, 48))
    cv2.line(img, p(0, 15), p(20, 15), (240, 240, 240), 2)
    cv2.line(img, p(0, 29), p(20, 29), (240, 240, 240), 2)
    cv2.line(img, p(0, 22), p(20, 22), (60, 60, 220), 4)
    for y in (7.5, 36.5):
        for x0, x1 in ((0, 10), (10, 20)):
            for t in np.arange(x0, x1, 1.0):
                a = p(t, y)
                b = p(min(t + 0.5, x1), y)
                cv2.line(img, a, b, (150, 150, 150), 1)
    cv2.putText(img, "BIRD EYE", (BEV_MARGIN, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_skeleton(frame: np.ndarray, kpts_px: np.ndarray, valid: np.ndarray | None = None) -> None:
    if valid is None:
        valid = np.ones(len(kpts_px), dtype=bool)
    for a, b in SKELETON_EDGES:
        cv2.line(frame, tuple(kpts_px[a].astype(int)), tuple(kpts_px[b].astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    if len(kpts_px) > 8:
        mid = (kpts_px[4] + kpts_px[5]) / 2
        cv2.line(frame, tuple(kpts_px[8].astype(int)), tuple(mid.astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    for i, pt in enumerate(kpts_px):
        color = (0, 200, 255) if valid[i] else (60, 60, 60)
        cv2.circle(frame, tuple(pt.astype(int)), 4, color, -1, cv2.LINE_AA)


def detect_bounce(ball_xy: np.ndarray, world_xy: np.ndarray, min_i: int = 0, fps: float = 30.0) -> tuple[int, float, float] | None:
    cy = ball_xy[:, 1].copy()
    valid = ~np.isnan(cy)
    idx = np.arange(len(cy))
    runs = []
    a = b = None
    for f in idx[valid]:
        if a is None:
            a = b = f
        elif f == b + 1:
            b = f
        else:
            runs.append((a, b))
            a = b = f
    if a is not None:
        runs.append((a, b))
    for a, b in runs:
        j = a
        while j <= b:
            k = j
            while k + 1 <= b and abs(float(cy[k + 1]) - float(cy[j])) <= 0.6:
                k += 1
            if k - j + 1 >= 8:
                valid[j:k + 1] = False
                j = k + 1
            else:
                j += 1
    if valid.sum() < 20:
        return None
    vidx = idx[valid]
    cy_s = np.interp(idx, idx[valid], cy[valid])
    vwx = (~np.isnan(world_xy[:, 0])) & valid
    vwy = (~np.isnan(world_xy[:, 1])) & valid
    wx_s = np.interp(idx, idx[vwx], world_xy[vwx, 0])
    wy_s = np.interp(idx, idx[vwy], world_xy[vwy, 1])
    s = fps / 30.0
    win = max(3, round(5 * s)) | 1
    ker = np.ones(win) / win
    cy_s = np.convolve(cy_s, ker, mode="same")
    vy = np.gradient(cy_s)
    pre_n = max(4, round(8 * s))
    post_n = max(4, round(6 * s))
    thr = 0.6 / s
    start_i = max(12, round(12 * s), min_i)
    for i in range(start_i, len(vy) - max(6, round(8 * s))):
        pre = vy[i - pre_n:i]
        post = vy[i:i + post_n]
        if pre.mean() < thr or post.mean() > -thr:
            continue
        before = vidx[vidx < i]
        after = vidx[vidx > i]
        if before.size == 0 or after.size == 0:
            continue
        if i - before[-1] > 2 * pre_n or after[0] - i > 2 * post_n:
            continue
        if (before >= i - 2 * pre_n).sum() < 3 or (after <= i + 2 * post_n).sum() < 2:
            continue
        if not ((before >= i - 2 * pre_n) & (before <= i - pre_n)).any():
            continue
        wx, wy = wx_s[i], wy_s[i]
        if -4 <= wx <= 24 and -4 <= wy <= 48:
            return i, float(wx), float(wy)
    return None


def process_video(
    in_path: str,
    out_path: str | None = None,
    stride: int = 1,
    court_every: int = 5,
    conf: float = 0.3,
    progress=None,
    fixture: str | None = None,
    min_bounce_frame: int | None = None,
) -> dict:
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else ROOT / "outputs" / f"{in_path.stem}_out.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.stem + "_pass1.mp4")

    ball_model, court_model = get_models()

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fps = max(fps / stride, 1.0)
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w + BEV_W, h))
    if not writer.isOpened():
        raise RuntimeError("cannot open writer")

    bev_base = draw_bev_base(h)
    trail = deque(maxlen=80)
    last_bp = None
    last_bi = -10
    prev_bp = None
    static_n = 0
    prev_wp = None
    last_wp_i = -10
    kpts_ema = None
    kpts_last = None
    kpts_valid = None
    ball_xy = np.full((max(total, 1), 2), np.nan, dtype=np.float32)
    world_xy = np.full((max(total, 1), 2), np.nan, dtype=np.float32)
    court_frames = 0
    kpt_counts = []
    t0 = time.time()
    frame_i = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % stride:
            continue

        if frame_i % (court_every * stride) == 0:
            pres = court_model.predict(frame, imgsz=640, conf=0.5, verbose=False)
            r = pres[0]
            if r.keypoints is not None and len(r.keypoints.data) and float(r.boxes.conf[0]) >= 0.5:
                kd = r.keypoints.data[0].cpu().numpy()
                k = kd[:, :2].astype(np.float32)
                m = kd[:, 2] >= 0.4 if kd.shape[1] > 2 else np.ones(len(k), dtype=bool)
                kpts_ema = k if kpts_ema is None else 0.5 * kpts_ema + 0.5 * k
                kpts_last = kpts_ema
                kpts_valid = m
                court_frames += 1
                kpt_counts.append(int(m.sum()))

        H = None
        if kpts_last is not None:
            H, _ = cv2.findHomography(
                kpts_last.reshape(-1, 1, 2), KPT_COURT_FT.reshape(-1, 1, 2),
                method=cv2.RANSAC, ransacReprojThreshold=3.0,
            )

        bres = ball_model.predict(frame, imgsz=640, conf=conf, verbose=False)
        boxes = bres[0].boxes
        best = None
        if boxes is not None and len(boxes):
            ci = int(np.argmax(boxes.conf.cpu().numpy()))
            if float(boxes.conf[ci]) >= conf:
                x0, y0, x1, y1 = boxes.xyxy[ci].cpu().numpy()
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if prev_bp is not None and abs(cx - prev_bp[0]) < 2 and abs(cy - prev_bp[1]) < 2:
                    static_n += 1
                else:
                    static_n = 0
                if static_n > fps:
                    prev_bp = (cx, cy)
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 140, 255), 2)
                    cv2.putText(frame, f"static {float(boxes.conf[ci]):.2f} (ignored)", (int(x0), int(y0) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
                else:
                    prev_bp = (cx, cy)
                    best = (cx, cy)
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Ball {float(boxes.conf[ci]):.2f}", (int(x0), int(y0) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if kpts_last is not None:
            draw_skeleton(frame, kpts_last, kpts_valid)

        bev = bev_base.copy()
        if H is not None:
            if best is not None:
                wp = cv2.perspectiveTransform(np.array([[[best[0], best[1]]]], dtype=np.float32), H)[0, 0]
                jump = False
                if prev_wp is not None and frame_i - last_wp_i <= 2:
                    if abs(float(wp[0]) - prev_wp[0]) > 6 or abs(float(wp[1]) - prev_wp[1]) > 6:
                        jump = True
                if jump:
                    last_bp, last_bi = None, -10
                else:
                    ball_xy[frame_i] = best
                    world_xy[frame_i] = wp
                    prev_wp, last_wp_i = (float(wp[0]), float(wp[1])), frame_i
                    if -4 <= wp[0] <= 24 and -4 <= wp[1] <= 48:
                        trail.append((tuple(ft_to_bev(float(wp[0]), float(wp[1]))), frame_i))
                    if last_bp is not None and frame_i - last_bi <= 2 and \
                            (abs(best[0] - last_bp[0]) ** 2 + abs(best[1] - last_bp[1]) ** 2) ** 0.5 <= 150:
                        cv2.line(frame, (int(last_bp[0]), int(last_bp[1])), (int(best[0]), int(best[1])), (255, 255, 0), 2, cv2.LINE_AA)
                    last_bp, last_bi = best, frame_i
                cv2.circle(frame, (int(best[0]), int(best[1])), 5, (0, 0, 255), -1)
        tp = list(trail)
        for (a, ia), (b, ib) in zip(tp, tp[1:]):
            if ib - ia <= 2 and (abs(a[0] - b[0]) ** 2 + abs(a[1] - b[1]) ** 2) ** 0.5 <= 90:
                cv2.line(bev, a, b, (0, 255, 255), 2, cv2.LINE_AA)
        if tp:
            cv2.circle(bev, tp[-1][0], 6, (0, 0, 255), -1)

        writer.write(np.hstack([frame, bev]))
        if progress and total and frame_i % 30 == 0:
            progress(min(frame_i / total, 1.0), desc=f"frame {frame_i}/{total}")

    cap.release()
    writer.release()

    n = frame_i + 1
    # Optional same-view serve contact bound; callers without contact keep prior behavior.
    bounce_start = max(int(fps * 0.5), min_bounce_frame or 0)
    hit = detect_bounce(ball_xy[:n], world_xy[:n], min_i=bounce_start, fps=fps)
    summary = {
        "frames": n,
        "stride": stride,
        "seconds": round(time.time() - t0, 1),
        "court_rate": round(court_frames / max(n / (court_every * stride), 1), 3),
        "kpt_conf_avg": round(float(np.mean(kpt_counts)), 2) if kpt_counts else 0.0,
        "landing": None,
    }
    if hit is not None:
        bounce, lx, ly = hit
        zone = classify_zone(lx, ly)
        summary["landing"] = {
            "frame": bounce,
            "x_ft": round(lx, 2),
            "y_ft": round(ly, 2),
            "zone": zone,
        }
        stamp_landing(tmp_path, out_path, bounce, zone, lx, ly, ball_xy, stride)
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.replace(out_path)
        summary["note"] = "no landing detected"

    if progress:
        progress(1.0, desc="done")
    summary["out"] = str(out_path)
    if fixture:
        fp = Path(fixture) if Path(fixture).is_absolute() else ROOT / fixture
        fp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {
                "fps": fps,
                "stride": stride,
                "min_i": bounce_start,
                "frames": n,
                "landing": summary.get("landing"),
                "ball_xy": [
                    [float(p[0]) if np.isfinite(p[0]) else None, float(p[1]) if np.isfinite(p[1]) else None]
                    for p in ball_xy[:n]
                ],
                "world_xy": [
                    [float(p[0]) if np.isfinite(p[0]) else None, float(p[1]) if np.isfinite(p[1]) else None]
                    for p in world_xy[:n]
                ],
            },
            open(fp, "w"),
        )
        summary["fixture"] = str(fp)
    return summary


def stamp_landing(src: Path, dst: Path, bounce: int, zone: str, lx: float, ly: float, ball_xy, stride: int = 1) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_w = w - BEV_W
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bi = bounce // max(stride, 1)
    rect = None
    if zone.startswith(("Deep", "Short")):
        x0, x1 = (0.0, 10.0) if lx < 10 else (10.0, 20.0)
        if ly < 15:
            y0, y1 = (0.0, 7.5) if ly < 7.5 else (7.5, 15.0)
        else:
            y0, y1 = (29.0, 36.5) if ly < 36.5 else (36.5, 44.0)
        rect = (x0, y0, x1, y1)
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        cam = frame[:, :cam_w]
        bez = frame[:, cam_w:]
        if i >= bi and rect:
            p0 = ft_to_bev(rect[0], rect[1])
            p1 = ft_to_bev(rect[2], rect[3])
            overlay = bez.copy()
            cv2.rectangle(overlay, p0, p1, (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.35, bez, 0.65, 0, bez)
            cv2.rectangle(bez, p0, p1, (0, 140, 255), 3)
            cv2.putText(bez, zone, (p0[0] + 6, p0[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(bez, ft_to_bev(lx, ly), 8, (0, 0, 255), -1, cv2.LINE_AA)
        if i == bi:
            j = bounce
            while j >= 0 and np.isnan(ball_xy[j, 0]):
                j -= 1
            if j >= 0 and bounce - j <= 4:
                bx, by = ball_xy[j]
                cv2.circle(cam, (int(bx), int(by)), 14, (0, 0, 255), 3)
                cv2.putText(cam, "LANDING", (int(bx) + 18, int(by) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if i >= bi:
            cv2.putText(cam, f"Landing zone: {zone}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        writer.write(np.hstack([cam, bez]))
    cap.release()
    writer.release()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python scripts/balltrack_pipeline.py <video> [stride]")
        raise SystemExit(1)
    src = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(process_video(src, stride=stride))
