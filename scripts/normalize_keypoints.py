"""
normalize_keypoints.py

Re-centers and rescales already-extracted BlazePose keypoints so the GRU
sees body-relative motion instead of screen position. Without this, the
model can pick up on where someone stood in frame or how far they were
from the camera -- cues that have nothing to do with serve type and don't
transfer to a new person or a new recording setup.

For every frame:
  1. Subtract the mid-hip point (average of landmarks 23, 24) from every
     keypoint, so absolute position in the frame no longer matters.
  2. Divide by the clip's median torso length (mid-shoulder to mid-hip), so
     standing closer to or farther from the camera no longer matters. A
     single per-clip scale is used rather than per-frame so torso tracking
     noise can't jitter the whole pose frame-to-frame.

Frames where hip/shoulder landmarks are themselves NaN stay NaN --
train_gru.py's pad_or_truncate already zero-fills those.

Usage:
  # normalize AFTER trimming, so train and holdout share the same pipeline
  python normalize_keypoints.py --in_dir data/training/keypoints_trimmed \
      --out_dir data/training/keypoints_norm
  python normalize_keypoints.py --in_dir data/testing/keypoints_trimmed \
      --out_dir data/testing/keypoints_norm
"""

import argparse
import csv
import os
import shutil

import numpy as np

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24


def normalize_clip(kpts: np.ndarray) -> np.ndarray:
    mid_hip = (kpts[:, LEFT_HIP, :] + kpts[:, RIGHT_HIP, :]) / 2  # (frames, 2)
    mid_shoulder = (kpts[:, LEFT_SHOULDER, :] + kpts[:, RIGHT_SHOULDER, :]) / 2

    torso_len = np.linalg.norm(mid_shoulder - mid_hip, axis=1)  # (frames,)

    # One scale for the whole clip (median torso length) instead of per-frame:
    # per-frame torso wobbles ~5% from tracking noise and would jitter the
    # entire pose every frame. Also stops a single tiny-torso frame from
    # exploding coordinates.
    valid = torso_len[(torso_len > 0.02) & np.isfinite(torso_len)]
    if valid.size == 0:
        scale = np.nan
    else:
        scale = float(np.median(valid))

    centered = kpts - mid_hip[:, np.newaxis, :]
    normalized = centered / scale
    return normalized.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.in_dir, "manifest.csv")

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_done = 0
    for row in rows:
        clip_id = os.path.splitext(os.path.basename(row["clip_path"]))[0]
        in_path = os.path.join(args.in_dir, f"{clip_id}.npy")
        if not os.path.exists(in_path):
            continue
        kpts = np.load(in_path)
        np.save(os.path.join(args.out_dir, f"{clip_id}.npy"), normalize_clip(kpts))
        n_done += 1

    shutil.copy(manifest_path, os.path.join(args.out_dir, "manifest.csv"))
    print(f"Normalized {n_done} clips -> {args.out_dir}")


if __name__ == "__main__":
    main()
