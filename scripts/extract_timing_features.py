"""
extract_timing_features.py

From joint-angle sequences, extract peak/trough timing + stroke-shape
features (option 4). Times are normalized to [0,1] over the valid clip so
they are length-invariant; amplitudes are absolute angle differences.

Features per clip (for each of 5 angles):
  - t_peak, t_trough: when max/min angle occurs (normalized)
  - t_maxvel: when max |angular velocity| occurs (normalized)
  - amp: max - min angle (radians)
  - mean, std of the angle

Plus cross-angle timing:
  - lag(shoulder_peak - elbow_peak), lag(shoulder_trough - elbow_trough)

Output: (n_clips, n_features) float32 matrix + labels/clip_ids/groups CSV.

Usage:
  python scripts/extract_timing_features.py --in_dir data/training/keypoints_angles \
      --out features/timing_train.npz
"""

import argparse
import csv
import os

import numpy as np

ANGLE_NAMES = ["elbow", "shoulder", "wrist", "torso", "twist"]


def valid_len(arr: np.ndarray) -> int:
    """Length of non-all-zero rows (trailing pad)."""
    nonzero = np.any(arr != 0.0, axis=tuple(range(1, arr.ndim)))
    if not nonzero.any():
        return arr.shape[0]
    return int(np.max(np.nonzero(nonzero)[0]) + 1)


def timing_features(seq: np.ndarray) -> np.ndarray:
    """seq: (frames, K, 2) sin/cos angles -> 1-D feature vector."""
    L = valid_len(seq)
    if L < 4:
        return np.zeros(5 * 6 + 2, dtype=np.float32)
    seq = seq[:L]
    ang = np.arctan2(seq[..., 0], seq[..., 1])  # (L, K)
    K = ang.shape[1]
    feats = []
    peaks = []
    troughs = []
    for k in range(K):
        a = ang[:, k]
        i_pk = int(np.argmax(a))
        i_tr = int(np.argmin(a))
        vel = np.diff(a, prepend=a[0])
        i_vel = int(np.argmax(np.abs(vel)))
        peaks.append(i_pk / max(L - 1, 1))
        troughs.append(i_tr / max(L - 1, 1))
        feats.extend(
            [
                i_pk / max(L - 1, 1),
                i_tr / max(L - 1, 1),
                i_vel / max(L - 1, 1),
                float(a.max() - a.min()),
                float(a.mean()),
                float(a.std()),
            ]
        )
    # cross-angle lags: shoulder (idx 1) vs elbow (idx 0)
    lag_pk = peaks[1] - peaks[0]
    lag_tr = troughs[1] - troughs[0]
    feats.extend([lag_pk, lag_tr])
    return np.asarray(feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out", required=True, help="Output .npz path")
    args = parser.parse_args()

    manifest_path = os.path.join(args.in_dir, "manifest.csv")
    rows = list(csv.DictReader(open(manifest_path, newline="")))

    X, y, clip_ids, groups = [], [], [], []
    for row in rows:
        clip_id = os.path.splitext(os.path.basename(row["clip_path"]))[0]
        npy_path = os.path.join(args.in_dir, f"{clip_id}.npy")
        if not os.path.exists(npy_path):
            continue
        seq = np.load(npy_path)
        feat = timing_features(seq)
        X.append(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0))
        y.append(row["serve_type"].strip().lower())
        clip_ids.append(clip_id)
        groups.append(clip_id.split("_")[0])

    X = np.stack(X)
    np.savez(
        args.out,
        X=X,
        y=np.array(y),
        clip_ids=np.array(clip_ids),
        groups=np.array(groups),
    )
    print(f"Extracted {X.shape[0]} clips, {X.shape[1]} features -> {args.out}")


if __name__ == "__main__":
    main()
