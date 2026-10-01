"""
derive_paddle_threshold.py

Derives the empirical paddle-orientation threshold band from Coach A's
contact-window scalars and validates it against every training beginner.

Metric (must match the Android rule engine exactly):
  angle   = per-clip scalar = median of minAreaRect long-axis angles
            (deg, [0,180)) over frames in the contact window (+-5)
  verdict = too_closed  if angle <  band_low
            too_open    if angle >  band_high
            optimal     otherwise
  band    = [p5, p95] of Coach A clips

Usage (from project root):
  python scripts/derive_paddle_threshold.py

Writes: android/app/src/main/assets/paddle_config.json
"""

import csv
import json
import os

import numpy as np
from coach_annotations import baseline_ids, provenance

SUMMARY_CSV = "outputs/paddle_angles/ft_train/summary.csv"
OUT_JSON = "android/app/src/main/assets/paddle_config.json"

P_LOW, P_HIGH = 5.0, 95.0

NOTE = (
    "Direction convention (stated heuristic): side-view camera with the "
    "supination that opens the face also raising the paddle tip, so angles "
    "farther from horizontal-toward-90deg read as too_open and toward "
    "0/180deg as too_closed; no per-clip open/closed ground truth exists to "
    "verify the mapping."
)


def load_scalars(path):
    pools = {}
    meta = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r["contact_angle"]:
                continue
            pools.setdefault(r["group"], []).append(float(r["contact_angle"]))
            meta.append(r)
    return {g: np.array(v) for g, v in pools.items()}, meta


def main():
    pools, meta = load_scalars(SUMMARY_CSV)
    eligible = set(baseline_ids())
    baseline = [r for r in meta if r["clip_id"] in eligible and r["scalar_rule"] == "window"]
    if len({r["clip_id"] for r in baseline}) != len(baseline):
        raise ValueError("Duplicate baseline paddle clip IDs")
    if len(baseline) < 10:
        raise ValueError("Insufficient contact-window measurements in annotated baseline")
    coach = np.array([float(r["contact_angle"]) for r in baseline])
    if not np.isfinite(coach).all():
        raise ValueError("Non-finite baseline paddle angles")
    low = float(np.percentile(coach, P_LOW))
    high = float(np.percentile(coach, P_HIGH))

    print(f"=== paddle long-axis angle at contact (window median, deg) ===")
    print(f"{'group':<10} {'n':>4} {'p5':>7} {'p25':>7} {'med':>7} "
          f"{'p75':>7} {'p95':>7}")
    for g in sorted(pools):
        v = pools[g]
        print(f"{g:<10} {len(v):>4} {np.percentile(v,5):>7.1f} "
              f"{np.percentile(v,25):>7.1f} {np.median(v):>7.1f} "
              f"{np.percentile(v,75):>7.1f} {np.percentile(v,95):>7.1f}")

    print(f"\nband = coach_p{P_LOW:g}-p{P_HIGH:g} = [{low:.1f}, {high:.1f}]  "
          f"(below -> too_closed, above -> too_open)")
    print(f"{'group':<10} {'n':>4} {'in_band':>8} {'rate':>6} {'too_closed':>11} "
          f"{'too_open':>9}")
    rates = {}
    for g in sorted(pools):
        v = pools[g]
        n_in = int(((v >= low) & (v <= high)).sum())
        r = n_in / len(v)
        rates[g] = round(r, 3)
        print(f"{g:<10} {len(v):>4} {n_in:>8} {r:>6.2f} "
              f"{int((v < low).sum()):>11} {int((v > high).sum()):>9}")

    n_window = sum(1 for r in meta if r["scalar_rule"] == "window")
    cfg = {
        **provenance(),
        "baseline_clip_ids": [r["clip_id"] for r in baseline],
        "excluded_baseline_clip_ids": sorted(eligible - {r["clip_id"] for r in baseline}),
        "direction_validated": False,
        "metric": "paddle_long_axis_angle_deg",
        "angle_domain": "[0,180)",
        "scalar_rule": "median_over_contact_window",
        "window_half_frames": 5,
        "threshold_rule": "coach_p5_p95",
        "band_low": low,
        "band_high": high,
        "labels": {"below": "too_closed", "above": "too_open",
                   "within": "optimal"},
        "n_coach_clips": int(len(coach)),
        "n_scalars_total": int(sum(len(v) for v in pools.values())),
        "n_window_rule": n_window,
        "in_band_rates": rates,
        "note": NOTE,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
