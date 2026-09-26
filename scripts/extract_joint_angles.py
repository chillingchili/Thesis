"""
extract_joint_angles.py

Converts BlazePose keypoints (ideally already run through
normalize_keypoints.py) into a small set of joint angles instead of raw
(x, y) coordinates. Angles describe the shape of the motion -- how bent
the elbow is, how far the arm has swung relative to the torso -- without
encoding a specific person's limb proportions the way raw coordinates do.
This is a much stronger person-invariance guarantee than centering and
scaling alone.

Each angle is encoded as (sin, cos) rather than a raw radian value, so a
small physical change in angle never looks like a huge numeric jump (the
wraparound problem you'd get right at +-180 degrees).

Angles computed (right arm, since your subjects are right-handed):
  - elbow    : interior angle at the elbow (shoulder-elbow-wrist)
  - shoulder : interior angle at the shoulder (hip-shoulder-elbow)
  - wrist    : interior angle at the wrist (elbow-wrist-index finger)
  - torso    : lean of the torso relative to vertical
  - twist    : rotation of the shoulder line relative to the hip line

Output shape per clip: (frames, 5, 2) -- same nested-array shape
extract_keypoints.py produces, so train_gru.py works unchanged, just
point --keypoints_dir at this script's --out_dir instead.

Usage:
  python extract_joint_angles.py --in_dir data/keypoints_norm --out_dir data/keypoints_angles
"""

import argparse
import csv
import os
import shutil

import numpy as np

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_INDEX = 20
LEFT_HIP, RIGHT_HIP = 23, 24


def unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < eps, eps, n)


def interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex b formed by points a-b-c. Shape (frames,)."""
    v1, v2 = unit(a - b), unit(c - b)
    cos_theta = np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)
    return np.arccos(cos_theta)


def signed_angle_from_vertical(v: np.ndarray) -> np.ndarray:
    """Signed angle between v and 'up' in image coordinates (y grows down)."""
    return np.arctan2(v[..., 0], -v[..., 1])


def signed_angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = np.sum(v1 * v2, axis=-1)
    return np.arctan2(cross, dot)


def clip_angles(kpts: np.ndarray) -> np.ndarray:
    mid_shoulder = (kpts[:, LEFT_SHOULDER, :] + kpts[:, RIGHT_SHOULDER, :]) / 2
    mid_hip = (kpts[:, LEFT_HIP, :] + kpts[:, RIGHT_HIP, :]) / 2

    elbow = interior_angle(kpts[:, RIGHT_SHOULDER], kpts[:, RIGHT_ELBOW], kpts[:, RIGHT_WRIST])
    shoulder = interior_angle(kpts[:, RIGHT_HIP], kpts[:, RIGHT_SHOULDER], kpts[:, RIGHT_ELBOW])
    wrist = interior_angle(kpts[:, RIGHT_ELBOW], kpts[:, RIGHT_WRIST], kpts[:, RIGHT_INDEX])
    torso = signed_angle_from_vertical(mid_shoulder - mid_hip)
    twist = signed_angle_between(
        kpts[:, RIGHT_SHOULDER] - kpts[:, LEFT_SHOULDER],
        kpts[:, RIGHT_HIP] - kpts[:, LEFT_HIP],
    )

    angles = np.stack([elbow, shoulder, wrist, torso, twist], axis=-1)  # (frames, 5)
    return np.stack([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)  # (frames, 5, 2)


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
        np.save(os.path.join(args.out_dir, f"{clip_id}.npy"), clip_angles(kpts))
        n_done += 1

    shutil.copy(manifest_path, os.path.join(args.out_dir, "manifest.csv"))
    print(f"Converted {n_done} clips to joint-angle features -> {args.out_dir}")


if __name__ == "__main__":
    main()
