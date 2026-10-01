"""
gen_paddle_stream_fixtures.py

Exports golden fixtures for the Android PaddleStream JVM unit tests:
per-clip wrist speed series + per-frame angles with the expected contact
frame and window scalar (as produced by the offline python pipeline), plus
long_axis_angle synthetic cases.

Usage (from project root):
  python scripts/gen_paddle_stream_fixtures.py

Writes: android/app/src/test/resources/fixtures/paddle_stream.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_paddle_angles import long_axis_angle  # noqa: E402

SUMMARY = Path("outputs/paddle_angles/ft_train/summary.csv")
FRAMES_DIR = Path("outputs/paddle_angles/ft_train")
KPTS_DIR = Path("data/training/keypoints")
OUT = Path("android/app/src/test/resources/fixtures/paddle_stream.json")
RIGHT_WRIST = 16


def speeds_of(npy_path: Path):
    kpts = np.load(npy_path)
    wrist = kpts[:, RIGHT_WRIST, :]
    speed = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    speed = np.nan_to_num(speed, nan=0.0)
    return kpts.shape[0], [float(v) for v in speed]


def pick_rows(rows):
    window = [r for r in rows if r["scalar_rule"] == "window"]
    full = [r for r in rows if r["scalar_rule"] == "full_series"]
    chosen = []
    groups = ("CoachA", "Beg1", "Beg2")
    for g in groups:
        gw = [r for r in window if r["group"] == g]
        chosen += [gw[i] for i in range(0, len(gw), max(1, len(gw) // 4))][:4]
    chosen += [full[i] for i in range(0, len(full), max(1, len(full) // 3))][:4]
    seen, out = set(), []
    for r in chosen:
        if r["clip_id"] not in seen:
            seen.add(r["clip_id"])
            out.append(r)
    return out


def main():
    with open(SUMMARY, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["contact_angle"]]
    chosen = pick_rows(rows)

    clips = []
    for r in chosen:
        clip_id = r["clip_id"]
        npy = KPTS_DIR / f"{clip_id}.npy"
        if not npy.exists():
            continue
        t, speeds = speeds_of(npy)
        rec = json.loads((FRAMES_DIR / f"{clip_id}.json").read_text(encoding="utf-8"))
        angles = [[fr, a] for fr, a in zip(rec["frames"], rec["angles"])
                  if a is not None]
        clips.append({
            "clip_id": clip_id,
            "group": r["group"],
            "rule": r["scalar_rule"],
            "t": t,
            "speeds": speeds,
            "angles": angles,
            "contact": int(r["contact_frame"]),
            "scalar": float(r["contact_angle"]),
        })
        print(f"{clip_id}: T={t} speeds={len(speeds)} angles={len(angles)} "
              f"contact={r['contact_frame']} rule={r['scalar_rule']}")

    axis = []
    cases = [(10, 5, 0.0), (5, 10, 0.0), (10, 5, 90.0), (10, 5, 45.0),
             (10, 5, -45.0), (5, 10, 45.0), (10, 5, 180.0), (10, 5, 179.5),
             (10, 5, 90.5), (20, 20, 30.0), (10, 5, -90.0), (10, 5, 89.999),
             (3, 40, 12.25), (40, 3, 177.75), (7, 7, -0.0001)]
    for w, h, ang in cases:
        expected = long_axis_angle(((0, 0), (w, h), ang))
        axis.append({"w": w, "h": h, "ang": ang, "expected": expected})
        print(f"axis w={w} h={h} ang={ang} -> {expected}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"clips": clips, "axis": axis}), encoding="utf-8")
    print(f"Wrote {OUT} ({len(clips)} clips, {len(axis)} axis cases)")


if __name__ == "__main__":
    main()
