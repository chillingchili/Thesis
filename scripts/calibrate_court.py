"""One-time court calibration from near-side lines (unoccluded): fits the two
sidelines + near baseline + near NVZ, solves image<->court homography, projects
all 14 canonical kpts. For camera-static footage.

usage: python scripts/calibrate_court.py <video> <frame_i> [out_png]
Writes: outputs/court_calib.json + overlay png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from balltrack_pipeline import KPT_COURT_FT, SKELETON_EDGES  # noqa: E402


def white_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > 170) & (hsv[..., 1] < 60)).astype(np.uint8)


def hough(mask: np.ndarray):
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, minLineLength=170, maxLineGap=40)
    return [] if segs is None else [tuple(map(int, s)) for s in segs[:, 0]]


def fit_line_trim(pts, rounds=3, tol=20.0):
    p = np.array(pts, dtype=np.float64)
    for _ in range(rounds):
        A = np.vstack([p[:, 0], np.ones(len(p))]).T
        a, b = np.linalg.lstsq(A, p[:, 1], rcond=None)[0]
        r = np.abs(p[:, 0] * a + b - p[:, 1]) / np.sqrt(a * a + 1)
        keep = r < tol
        if keep.sum() >= 4:
            p = p[keep]
    A = np.vstack([p[:, 0], np.ones(len(p))]).T
    a, b = np.linalg.lstsq(A, p[:, 1], rcond=None)[0]
    return a, b, len(p)


def sideline(segs, y_at, x_lo, x_hi, slope_lo, slope_hi, x950_rng, y2, x2_rng):
    pts = []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) > abs(dy) * 1.4 or max(abs(dx), abs(dy)) < 60:
            continue
        slope = dx / dy
        if not (slope_lo <= slope <= slope_hi):
            continue
        xm = (x1 + x2) / 2
        if not (x_lo <= xm <= x_hi):
            continue
        pts.append((y1, x1))
        pts.append((y2, x2))
    if len(pts) < 4:
        return None
    a, b, n = fit_line_trim(pts)
    if not (x950_rng[0] <= a * y_at + b <= x950_rng[1]):
        return None
    if not (x2_rng[0] <= a * y2 + b <= x2_rng[1]):
        return None
    return a, b, n


def h_at(segs, y_lo, y_hi, xc, tol=170):
    ys, ws = [], []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) > abs(dx) * 0.35 or abs(dx) < 220:
            continue
        if max(x1, x2) < xc - tol or min(x1, x2) > xc + tol:
            continue
        if not (y_lo <= y1 <= y_hi or y_lo <= y2 <= y_hi):
            continue
        t = (xc - x1) / dx if dx else 0
        if not (0 <= t <= 1):
            continue
        ys.append(y1 + t * dy)
        ws.append(abs(dx))
    if not ys:
        return None
    ys, ws = np.array(ys), np.array(ws)
    med = np.median(ys)
    keep = np.abs(ys - med) < 25
    return float(np.average(ys[keep], weights=ws[keep])) if keep.any() else None


def main() -> None:
    video = sys.argv[1]
    fi = int(sys.argv[2])
    out_png = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "outputs" / "court_calib_overlay.png")
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("read failed")
    mask = white_mask(frame)
    Hm, Wm = mask.shape
    xc = Wm // 2
    segs = hough(mask)

    near_nvz_y = h_at(segs, 505, 615, xc)
    near_base_y = h_at(segs, 870, 1020, xc)
    left = sideline(segs, near_base_y, 0, xc - 150, -1.8, -0.18, (40, 560),
                    near_nvz_y, (350, 750))
    right = sideline(segs, near_base_y, xc + 150, Wm, 0.18, 1.8, (1400, 1930),
                     near_nvz_y, (1250, 1650))
    print("nearNVZ_y:", near_nvz_y and round(near_nvz_y, 1),
          "nearBASE_y:", near_base_y and round(near_base_y, 1),
          "left:", left and (round(left[0], 3), round(left[1], 1), left[2]),
          "right:", right and (round(right[0], 3), round(right[1], 1), right[2]))
    if not all([near_nvz_y, near_base_y, left, right]):
        raise SystemExit("detection failed")

    def inter(y, hv):
        a, b = hv[0], hv[1]
        return (a * y + b, y)

    nearL = inter(near_base_y, left)
    nearR = inter(near_base_y, right)
    nvzL = inter(near_nvz_y, left)
    nvzR = inter(near_nvz_y, right)
    src = np.array([nearL, nearR, nvzR, nvzL], dtype=np.float32)
    dst = KPT_COURT_FT[[3, 2, 7, 6]].astype(np.float32)
    H_ci = cv2.getPerspectiveTransform(dst, src)

    kpts = cv2.perspectiveTransform(KPT_COURT_FT.reshape(-1, 1, 2), H_ci).reshape(-1, 2)
    vis = frame.copy()
    for a, b in SKELETON_EDGES:
        cv2.line(vis, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    mid = (kpts[4] + kpts[5]) / 2
    cv2.line(vis, tuple(kpts[8].astype(int)), tuple(mid.astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    for i, p in enumerate(kpts):
        cv2.circle(vis, tuple(p.astype(int)), 7, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (int(p[0]) + 10, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(out_png, vis)
    calib = {
        "video": video, "frame": fi,
        "H_canonical_to_image": H_ci.tolist(),
        "nearL": nearL, "nearR": nearR, "nvzL": nvzL, "nvzR": nvzR,
    }
    (ROOT / "outputs" / "court_calib.json").write_text(json.dumps(calib, indent=1))
    print("wrote", out_png)


if __name__ == "__main__":
    main()
