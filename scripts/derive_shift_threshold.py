"""
derive_shift_threshold.py

Derives the body-weight-shift threshold from Coach B's explicitly good-form
CoachA training clips. Does not access evaluation clips.

Metric (must match ShiftDetector.kt exactly):
  hip_x  = (x23 + x24) / 2 per valid frame, then 5-frame moving average
  body_h = median over frames of |y_nose - mean(y_ankle27, y_ankle28)|
  shift  = (p95(hip_x) - p5(hip_x)) / body_h        (dimensionless)
  verdict = shift >= threshold, threshold = p10 of eligible annotated clips

Usage (from project root):
  python scripts/derive_shift_threshold.py

Writes: android/app/src/main/assets/shift_config.json
"""

import json
import os

import numpy as np
from coach_annotations import ROOT, baseline_ids, provenance

CLASSES_NOTE = "Coach B explicit Good Form labels, CoachA training clips only; not a full-form classifier."

SMOOTH_WINDOW = 5
P_LOW, P_HIGH = 5.0, 95.0
THRESHOLD_PERCENTILE = 10.0
OUT_JSON = "android/app/src/main/assets/shift_config.json"


def clip_shift(kpts):
    """kpts: (T, 33, 2) normalized image coords -> shift value or None."""
    need = [0, 23, 24, 27, 28]  # nose, hips, ankles
    ok = np.isfinite(kpts[:, need, :]).all(axis=(1, 2))
    if ok.sum() < 10:
        return None
    k = kpts[ok]
    hipx = (k[:, 23, 0] + k[:, 24, 0]) / 2.0
    if len(hipx) >= SMOOTH_WINDOW:
        hipx = np.convolve(hipx, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode="valid")
    rng = float(np.percentile(hipx, P_HIGH) - np.percentile(hipx, P_LOW))
    ankle_y = (k[:, 27, 1] + k[:, 28, 1]) / 2.0
    body_h = float(np.median(np.abs(k[:, 0, 1] - ankle_y)))
    if body_h < 1e-6:
        return None
    return rng / body_h


def main():
    selected, missing, values = [], [], []
    for cid in baseline_ids():
        path = ROOT / "data/training/keypoints" / (cid + ".npy")
        value = clip_shift(np.load(path).astype(np.float32)) if path.exists() else None
        if value is None or not np.isfinite(value):
            missing.append(cid)
        else:
            selected.append(cid)
            values.append(value)
    if missing:
        print(f"Excluded missing/invalid baseline keypoints: {missing}")
    if len(values) < 10:
        raise ValueError("Insufficient annotated good-form training baseline")
    coach = np.array(values)
    threshold = float(np.percentile(coach, THRESHOLD_PERCENTILE))
    coach_rate = float((coach >= threshold).mean())
    cfg = {
        **provenance(),
        "baseline_clip_ids": selected,
        "excluded_baseline_clip_ids": missing,
        "metric": "hip_x_range_body_height_norm",
        "smooth_window": SMOOTH_WINDOW,
        "p_low": P_LOW,
        "p_high": P_HIGH,
        "threshold_rule": "coach_p10",
        "threshold": threshold,
        "n_coach_clips": int(len(coach)),
        "coach_detection_rate": round(coach_rate, 3),
        "hip_landmarks": [23, 24],
        "nose_landmark": 0,
        "ankle_landmarks": [27, 28],
        "note": CLASSES_NOTE,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    print(f"Annotated baseline n={len(coach)}, threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
