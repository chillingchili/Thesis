"""Ball tracking + court homography pipeline.

process_video(): annotated side-by-side video (broadcast view | bird's-eye court),
ball landing detection in court feet, and the thesis 7-zone classification.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
BALL_WEIGHTS = ROOT / "runs/detect/ball_yolo26s/weights/best.pt"
COURT_WEIGHTS = ROOT / "runs/pose/court_yolo26s/weights/best.pt"

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


def draw_skeleton(frame: np.ndarray, kpts_px: np.ndarray) -> None:
    for a, b in SKELETON_EDGES:
        cv2.line(frame, tuple(kpts_px[a].astype(int)), tuple(kpts_px[b].astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    mid = (kpts_px[4] + kpts_px[5]) / 2
    cv2.line(frame, tuple(kpts_px[8].astype(int)), tuple(mid.astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    for i, pt in enumerate(kpts_px):
        cv2.circle(frame, tuple(pt.astype(int)), 4, (0, 200, 255), -1, cv2.LINE_AA)


def detect_bounce(ball_xy: np.ndarray, world_xy: np.ndarray, min_i: int = 0) -> int | None:
    cy = ball_xy[:, 1].copy()
    valid = ~np.isnan(cy)
    if valid.sum() < 20:
        return None
    idx = np.arange(len(cy))
    cy_s = np.interp(idx, idx[valid], cy[valid])
    vwx = ~np.isnan(world_xy[:, 0])
    vwy = ~np.isnan(world_xy[:, 1])
    wx_s = np.interp(idx, idx[vwx], world_xy[vwx, 0])
    wy_s = np.interp(idx, idx[vwy], world_xy[vwy, 1])
    win = 5
    ker = np.ones(win) / win
    cy_s = np.convolve(cy_s, ker, mode="same")
    vy = np.gradient(cy_s)
    for i in range(max(12, min_i), len(vy) - 8):
        pre = vy[i - 8:i]
        post = vy[i:i + 6]
        if pre.mean() < 0.6 or post.mean() > -0.6:
            continue
        wx, wy = wx_s[i], wy_s[i]
        if -4 <= wx <= 24 and -4 <= wy <= 48:
            return i
    return None


def process_video(
    in_path: str,
    out_path: str | None = None,
    stride: int = 1,
    court_every: int = 5,
    conf: float = 0.3,
    progress=None,
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
    img_trail = deque(maxlen=40)
    kpts_ema = None
    kpts_last = None
    ball_xy = np.full((max(total, 1), 2), np.nan, dtype=np.float32)
    world_xy = np.full((max(total, 1), 2), np.nan, dtype=np.float32)
    court_frames = 0
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
                k = r.keypoints.data[0].cpu().numpy()[:, :2].astype(np.float32)
                kpts_ema = k if kpts_ema is None else 0.5 * kpts_ema + 0.5 * k
                kpts_last = kpts_ema
                court_frames += 1

        H = None
        if kpts_last is not None:
            H, _ = cv2.findHomography(
                kpts_last.reshape(-1, 1, 2), KPT_COURT_FT.reshape(-1, 1, 2), method=0
            )

        bres = ball_model.predict(frame, imgsz=640, conf=conf, verbose=False)
        boxes = bres[0].boxes
        best = None
        if boxes is not None and len(boxes):
            ci = int(np.argmax(boxes.conf.cpu().numpy()))
            if float(boxes.conf[ci]) >= conf:
                x0, y0, x1, y1 = boxes.xyxy[ci].cpu().numpy()
                best = ((x0 + x1) / 2, (y0 + y1) / 2)
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball {float(boxes.conf[ci]):.2f}", (int(x0), int(y0) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if kpts_last is not None:
            draw_skeleton(frame, kpts_last)

        bev = bev_base.copy()
        if H is not None:
            if best is not None:
                ball_xy[frame_i] = best
                wp = cv2.perspectiveTransform(np.array([[[best[0], best[1]]]], dtype=np.float32), H)[0, 0]
                world_xy[frame_i] = wp
                trail.append(tuple(ft_to_bev(float(wp[0]), float(wp[1]))))
            img_trail.append(best)
            pts = [p for p in img_trail if p is not None]
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 0), 2, cv2.LINE_AA)
            if best is not None:
                cv2.circle(frame, (int(best[0]), int(best[1])), 5, (0, 0, 255), -1)
        tp = list(trail)
        for a, b in zip(tp, tp[1:]):
            cv2.line(bev, a, b, (0, 255, 255), 2, cv2.LINE_AA)
        if tp:
            cv2.circle(bev, tp[-1], 6, (0, 0, 255), -1)

        writer.write(np.hstack([frame, bev]))
        if progress and total and frame_i % 30 == 0:
            progress(min(frame_i / total, 1.0), desc=f"frame {frame_i}/{total}")

    cap.release()
    writer.release()

    n = frame_i + 1
    bounce = detect_bounce(ball_xy[:n], world_xy[:n], min_i=int(fps * 0.5))
    summary = {
        "frames": n,
        "stride": stride,
        "seconds": round(time.time() - t0, 1),
        "court_rate": round(court_frames / max(n / (court_every * stride), 1), 3),
        "landing": None,
    }
    if bounce is not None:
        lx, ly = world_xy[bounce]
        zone = classify_zone(float(lx), float(ly))
        summary["landing"] = {
            "frame": bounce,
            "x_ft": round(float(lx), 2),
            "y_ft": round(float(ly), 2),
            "zone": zone,
        }
        stamp_landing(tmp_path, out_path, bounce, zone, ball_xy, world_xy)
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.replace(out_path)
        summary["note"] = "no landing detected"

    if progress:
        progress(1.0, desc="done")
    summary["out"] = str(out_path)
    return summary


def stamp_landing(src: Path, dst: Path, bounce: int, zone: str, ball_xy, world_xy) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bez = draw_bev_base(h)
    rect = None
    wx0, wy0 = world_xy[bounce]
    if zone.startswith(("Deep", "Short")):
        x0, x1 = (0.0, 10.0) if wx0 < 10 else (10.0, 20.0)
        if wy0 < 15:
            y0, y1 = (0.0, 7.5) if wy0 < 7.5 else (7.5, 15.0)
        else:
            y0, y1 = (29.0, 36.5) if wy0 < 36.5 else (36.5, 44.0)
        rect = (x0, y0, x1, y1)
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        bev = bez.copy()
        if i >= bounce and rect:
            p0 = ft_to_bev(rect[0], rect[1])
            p1 = ft_to_bev(rect[2], rect[3])
            overlay = bev.copy()
            cv2.rectangle(overlay, p0, p1, (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.35, bev, 0.65, 0, bev)
            cv2.rectangle(bev, p0, p1, (0, 140, 255), 3)
            cv2.putText(bev, zone, (p0[0] + 6, p0[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            wx, wy = world_xy[bounce]
            cv2.circle(bev, ft_to_bev(float(wx), float(wy)), 8, (0, 0, 255), -1, cv2.LINE_AA)
        if i == bounce:
            bx, by = ball_xy[bounce]
            cv2.circle(frame, (int(bx), int(by)), 14, (0, 0, 255), 3)
            cv2.putText(frame, "LANDING", (int(bx) + 18, int(by) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if i >= bounce:
            cv2.putText(frame, f"Landing zone: {zone}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        writer.write(np.hstack([frame, bev]))
    cap.release()
    writer.release()


if __name__ == "__main__":
    import sys

    src = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(process_video(src, stride=stride))
