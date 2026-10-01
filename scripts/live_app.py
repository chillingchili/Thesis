"""Real-time tracker: webcam or video file -> skeleton + ball + bird's-eye homography.

Run:      python scripts/live_app.py                 (webcam 0)
          python scripts/live_app.py 1               (camera index)
          python scripts/live_app.py path\to\vid.mp4 (video, source-speed playback)
Drag-drop: drop a video onto "Play with Tracker.bat"
Keys: q/ESC quit, s screenshot, r reset bounce & trails, SPACE pause (video)
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from balltrack_pipeline import (  # noqa: E402
    BEV_W,
    KPT_COURT_FT,
    classify_zone,
    draw_bev_base,
    draw_skeleton,
    ft_to_bev,
    get_models,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN = "Pickleball Live Tracker"
COURT_EVERY = 5
WINDOW = 150


def smooth_flip(cy: np.ndarray, wx: np.ndarray, wy: np.ndarray, fps: float = 30.0):
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
    s = fps / 30.0
    win = max(3, round(5 * s)) | 1
    cy_s = np.convolve(cy_s, np.ones(win) / win, mode="same")
    vy = np.gradient(cy_s)
    vwx = (~np.isnan(wx)) & valid
    vwy = (~np.isnan(wy)) & valid
    if not vwx.any() or not vwy.any():
        return None
    wx_s = np.interp(idx, idx[vwx], wx[vwx])
    wy_s = np.interp(idx, idx[vwy], wy[vwy])
    pre_n = max(4, round(8 * s))
    post_n = max(4, round(6 * s))
    thr = 0.6 / s
    for i in range(max(12, round(12 * s)), len(vy) - max(6, round(8 * s))):
        if vy[i - pre_n:i].mean() < thr or vy[i:i + post_n].mean() > -thr:
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
        if -4 <= wx_s[i] <= 24 and -4 <= wy_s[i] <= 48:
            return i, float(wx_s[i]), float(wy_s[i])
    return None


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    is_video = bool(arg) and Path(arg).exists()
    if is_video:
        cap = cv2.VideoCapture(str(arg))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        mode = f"video: {Path(arg).name}"
    else:
        cam_i = int(arg) if arg and arg.isdigit() else 0
        cap = cv2.VideoCapture(cam_i)
        src_fps = 30.0
        total = 0
        mode = f"camera {cam_i}"
    if not cap.isOpened():
        raise SystemExit(f"cannot open {arg}")
    if not is_video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, first = cap.read()
    if not ok:
        raise SystemExit("read failed")
    h, w = first.shape[:2]
    ball_model, court_model = get_models()
    if is_video:
        warm = np.zeros_like(first)
        ball_model.predict(warm, imgsz=640, conf=0.3, verbose=False, device=DEVICE)
        court_model.predict(warm, imgsz=640, conf=0.5, verbose=False, device=DEVICE)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(WIN, mode)
    t_play = time.time()
    paused = False

    bev_base = draw_bev_base(h)
    img_trail = deque(maxlen=60)
    bev_trail = deque(maxlen=90)
    kpts_ema = None
    kpts_last = None
    kpts_valid = None
    court_seen = -999
    court_conf = -1.0
    H = None
    prev_bp = None
    static_n = 0
    prev_wp = None
    prev_wp_i = -10
    bounce = None
    bounce_zone = None
    bounce_pt = None
    wlen = max(WINDOW, int(COURT_EVERY * 3))
    cy_arr = np.full(wlen, np.nan)
    wx_arr = np.full(wlen, np.nan)
    wy_arr = np.full(wlen, np.nan)
    abs_base = 0
    frames = 0
    t_fps = time.time()
    fps_disp = 0.0
    shot_n = 0

    while True:
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = False
                t_play = time.time() - frames / src_fps
            continue
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        i = frames

        wait_ms = 1
        if is_video:
            wall = time.time() - t_play
            video_pos = frames / src_fps
            if video_pos - wall < -2.0 / src_fps:
                continue
            wait_ms = max(1, min(int((video_pos - wall) * 1000), 120))

        if (i - abs_base) >= wlen:
            cy_arr[:] = np.nan
            wx_arr[:] = np.nan
            wy_arr[:] = np.nan
            abs_base = i - 1
        j = i - abs_base - 1

        if i % COURT_EVERY == 0:
            r = court_model.predict(frame, imgsz=640, conf=0.3, verbose=False, device=DEVICE)[0]
            court_conf = float(r.boxes.conf[0]) if r.boxes is not None and len(r.boxes) else -1.0
            if court_conf >= 0.5:
                court_seen = i
            if r.keypoints is not None and len(r.keypoints.data) and court_conf >= 0.5:
                kd = r.keypoints.data[0].cpu().numpy()
                k = kd[:, :2].astype(np.float32)
                m = kd[:, 2] >= 0.4 if kd.shape[1] > 2 else np.ones(len(kd), dtype=bool)
                if kpts_ema is None:
                    kpts_ema = k.copy()
                else:
                    kpts_ema = 0.5 * kpts_ema + 0.5 * k
                kpts_last = kpts_ema
                kpts_valid = m
                H, _ = cv2.findHomography(
                    kpts_last.reshape(-1, 1, 2), KPT_COURT_FT.reshape(-1, 1, 2),
                    method=cv2.RANSAC, ransacReprojThreshold=3.0,
                )

        bres = ball_model.predict(frame, imgsz=640, conf=0.3, verbose=False, device=DEVICE)[0]
        best = None
        if bres.boxes is not None and len(bres.boxes):
            ci = int(np.argmax(bres.boxes.conf.cpu().numpy()))
            if float(bres.boxes.conf[ci]) >= 0.3:
                x0, y0, x1, y1 = bres.boxes.xyxy[ci].cpu().numpy()
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if prev_bp is not None and abs(cx - prev_bp[0]) < 2 and abs(cy - prev_bp[1]) < 2:
                    static_n += 1
                else:
                    static_n = 0
                prev_bp = (cx, cy)
                if static_n > src_fps:
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 140, 255), 2)
                    cv2.putText(frame, f"static {float(bres.boxes.conf[ci]):.2f} (ignored)", (int(x0), int(y0) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
                else:
                    best = (cx, cy)
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Ball {float(bres.boxes.conf[ci]):.2f}", (int(x0), int(y0) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if best is not None:
            jump = False
            wp = None
            if H is not None:
                wp = cv2.perspectiveTransform(
                    np.array([[[best[0], best[1]]]], dtype=np.float32), H
                )[0, 0]
                if prev_wp is not None and i - prev_wp_i <= 2 and (
                        abs(float(wp[0]) - prev_wp[0]) > 6 or abs(float(wp[1]) - prev_wp[1]) > 6):
                    jump = True
            if jump:
                img_trail.pop()
            else:
                cy_arr[j] = best[1]
                if wp is not None:
                    wx_arr[j], wy_arr[j] = wp
                    prev_wp, prev_wp_i = (float(wp[0]), float(wp[1])), i
                    if bounce is None:
                        hit = smooth_flip(cy_arr[: j + 1], wx_arr[: j + 1], wy_arr[: j + 1], fps=src_fps)
                        if hit is not None:
                            bounce = abs_base + hit[0]
                            bounce_pt = (float(hit[1]), float(hit[2]))
                            bounce_zone = classify_zone(hit[1], hit[2])
                    if -4 <= wp[0] <= 24 and -4 <= wp[1] <= 48:
                        bev_trail.append((tuple(ft_to_bev(float(wp[0]), float(wp[1]))), i))
                if img_trail and i - img_trail[-1][1] <= 2:
                    a = img_trail[-1][0]
                    if (abs(best[0] - a[0]) ** 2 + abs(best[1] - a[1]) ** 2) ** 0.5 <= 150:
                        cv2.line(frame, (int(a[0]), int(a[1])), (int(best[0]), int(best[1])), (255, 255, 0), 2, cv2.LINE_AA)
                img_trail.append((best, i))
                cv2.circle(frame, (int(best[0]), int(best[1])), 5, (0, 0, 255), -1)

        if kpts_last is not None:
            draw_skeleton(frame, kpts_last, kpts_valid)

        bev = bev_base.copy()
        for (a, ia), (b2, ib) in zip(list(bev_trail), list(bev_trail)[1:]):
            if ib - ia <= 2 and (abs(a[0] - b2[0]) ** 2 + abs(a[1] - b2[1]) ** 2) ** 0.5 <= 90:
                cv2.line(bev, a, b2, (0, 255, 255), 2, cv2.LINE_AA)
        if bev_trail:
            cv2.circle(bev, list(bev_trail)[-1][0], 6, (0, 0, 255), -1)

        if bounce is not None and bounce_zone and bounce_zone.startswith(("Deep", "Short")):
            lx, ly = bounce_pt
            x0, x1 = (0.0, 10.0) if lx < 10 else (10.0, 20.0)
            if ly < 15:
                y0, y1 = (0.0, 7.5) if ly < 7.5 else (7.5, 15.0)
            else:
                y0, y1 = (29.0, 36.5) if ly < 36.5 else (36.5, 44.0)
            p0 = ft_to_bev(x0, y0)
            p1 = ft_to_bev(x1, y1)
            overlay = bev.copy()
            cv2.rectangle(overlay, p0, p1, (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.35, bev, 0.65, 0, bev)
            cv2.rectangle(bev, p0, p1, (0, 140, 255), 3)
            cv2.putText(bev, bounce_zone, (p0[0] + 6, p0[1] + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(bev, ft_to_bev(lx, ly), 8, (0, 0, 255), -1, cv2.LINE_AA)

        if bounce is not None:
            cv2.putText(frame, f"Landing zone: {bounce_zone}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

        conf_col = (0, 255, 0) if court_conf >= 0.5 else (0, 160, 255)
        cv2.putText(frame, f"court {court_conf:.2f}", (20, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_col, 2, cv2.LINE_AA)
        if i - court_seen > 60:
            cv2.putText(frame, "No court detected", (20, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

        now = time.time()
        if frames % 15 == 0:
            fps_disp = 15 / max(now - t_fps, 1e-6)
            t_fps = now
            pos = f" | {i}/{total}" if is_video and total else ""
            cv2.setWindowTitle(WIN, f"{mode}{pos} | {fps_disp:.0f} fps")
        cv2.putText(frame, f"{fps_disp:.0f} fps | {DEVICE}", (20, h - 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(bev, "BIRD EYE (LIVE)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WIN, np.hstack([frame, bev]))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = True
            continue
        if key == ord("s"):
            shot_n += 1
            out = ROOT_SHOTS / f"live_shot_{shot_n}.png"
            cv2.imwrite(str(out), np.hstack([frame, bev]))
            print("saved", out)
        if key == ord("r"):
            bounce = bounce_zone = bounce_pt = None
            bev_trail.clear()
            img_trail.clear()
            print("reset")
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    print(f"played {frames} frames | zone: {bounce_zone}")
    if is_video and bounce_zone:
        print(f"landing: {bounce_zone} at {bounce_pt}")
    if is_video:
        while True:
            key = cv2.waitKey(500) & 0xFF
            if key != -1:
                break
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    cap.release()
    cv2.destroyAllWindows()


ROOT_SHOTS = Path(__file__).resolve().parent.parent / "outputs"

if __name__ == "__main__":
    main()
