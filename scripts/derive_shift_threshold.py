"""
derive_shift_threshold.py

Derives the empirical body-weight-shift threshold from CoachA's clips and
validates it against every other subject (train + holdout).

Metric (must match ShiftDetector.kt exactly):
  hip_x  = (x23 + x24) / 2 per valid frame, then 5-frame moving average
  body_h = median over frames of |y_nose - mean(y_ankle27, y_ankle28)|
  shift  = (p95(hip_x) - p5(hip_x)) / body_h        (dimensionless)
  verdict = shift >= threshold,  threshold = p10 of CoachA clips

Usage (from project root):
  python scripts/derive_shift_threshold.py

Writes: android/app/src/main/assets/shift_config.json
"""

import csv
import json
import os

import numpy as np

CLASSES_NOTE = "coach good-form clips = all CoachA clips (no good/bad annotation exists)"

SETS = [
    ("train", "data/training/keypoints", "data/training/keypoints/manifest.csv"),
    ("beg3", "data/testing/keypoints", "data/testing/keypoints/manifest.csv"),
    ("beg4", "data/testing/keypoints_beg4", "data/testing/keypoints_beg4/manifest.csv"),
]

SMOOTH_WINDOW = 5
P_LOW, P_HIGH = 5.0, 95.0
THRESHOLD_PERCENTILE = 10.0
OUT_JSON = "android/app/src/main/assets/shift_config.json"


def clip_shift(kpts):
    """kpts: (T, 33, 2) normalized image coords -> shift value or None."""
    need = [0, 23, 24, 27, 28]  # nose, hips, ankles
    ok = ~np.isnan(kpts[:, need, :]).any(axis=(1, 2))
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


def load_set(kp_dir, manifest):
    rows = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            cid = os.path.splitext(os.path.basename(r["clip_path"]))[0]
            p = os.path.join(kp_dir, cid + ".npy")
            if not os.path.exists(p):
                continue
            v = clip_shift(np.load(p).astype(np.float32))
            if v is not None:
                subj = "Beg4" if cid.startswith("Nested") else cid.split("_")[0]
                rows.append((subj, v))
    return rows


def main():
    all_rows = []
    for name, kp, man in SETS:
        rows = load_set(kp, man)
        print(f"{name}: n={len(rows)}")
        all_rows += rows

    subjects = sorted({s for s, _ in all_rows})
    values = {s: np.array([v for ss, v in all_rows if ss == s]) for s in subjects}
    coach = values["CoachA"]
    threshold = float(np.percentile(coach, THRESHOLD_PERCENTILE))

    print(f"\n=== metric: hip_x_range / body_height  (smooth={SMOOTH_WINDOW}, "
          f"p{P_LOW:g}-p{P_HIGH:g}) ===")
    print(f"{'subject':<12} {'n':>4} {'min':>7} {'p10':>7} {'med':>7} {'max':>7}")
    for s in subjects:
        v = values[s]
        print(f"{s:<12} {len(v):>4} {v.min():>7.3f} {np.percentile(v,10):>7.3f} "
              f"{np.median(v):>7.3f} {v.max():>7.3f}")

    print(f"\nthreshold = coach_p10 = {threshold:.4f}  ({CLASSES_NOTE})")
    print(f"{'subject':<12} {'n':>4} {'detected':>9} {'rate':>6}")
    for s in subjects:
        v = values[s]
        n_ok = int((v >= threshold).sum())
        print(f"{s:<12} {len(v):>4} {n_ok:>9} {n_ok/len(v):>6.2f}")

    coach_rate = float((coach >= threshold).mean())
    fp = {s: float((values[s] >= threshold).mean())
          for s in subjects if s != "CoachA"}
    print(f"\nCoach detection rate: {coach_rate:.2f}")
    print("Beginner false-positive rates:",
          "  ".join(f"{s}={r:.2f}" for s, r in fp.items()))

    cfg = {
        "metric": "hip_x_range_body_height_norm",
        "smooth_window": SMOOTH_WINDOW,
        "p_low": P_LOW,
        "p_high": P_HIGH,
        "threshold_rule": "coach_p10",
        "threshold": round(threshold, 4),
        "n_coach_clips": int(len(coach)),
        "coach_detection_rate": round(coach_rate, 3),
        "beginner_false_positive_rates": {s: round(r, 3) for s, r in fp.items()},
        "hip_landmarks": [23, 24],
        "nose_landmark": 0,
        "ankle_landmarks": [27, 28],
        "note": CLASSES_NOTE,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
