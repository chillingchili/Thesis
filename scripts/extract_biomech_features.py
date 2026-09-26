"""Quick win 3: biomechanical feature enrichment on top of timing features.

Adds shape/kinematic features from joint-angle sequences that are more
subject-invariant than raw timing peaks:
  - angular velocity stats (mean/max std of d/dt)
  - ROM ratios between joints
  - cross-joint correlation
  - skew/kurtosis of angle distribution
  - number of velocity sign changes (smoothness proxy)
  - normalized path length of angle trajectory
"""
import argparse
import csv
import os
import numpy as np
from scipy import stats

ANGLE_NAMES = ["elbow", "shoulder", "wrist", "torso", "twist"]

def valid_len(arr):
    nonzero = np.any(arr != 0.0, axis=tuple(range(1, arr.ndim)))
    if not nonzero.any():
        return arr.shape[0]
    return int(np.max(np.nonzero(nonzero)[0]) + 1)

def biomech_features(seq):
    L = valid_len(seq)
    if L < 4:
        return np.zeros(5 * 4 + 10 + 5 + 5, dtype=np.float32)  # placeholder size
    seq = seq[:L]
    ang = np.arctan2(seq[..., 0], seq[..., 1])  # (L, K)
    K = ang.shape[1]
    vel = np.diff(ang, axis=0)  # (L-1, K)
    acc = np.diff(vel, axis=0)

    feats = []
    # per-joint: vel mean/max/std, acc max
    for k in range(K):
        v = vel[:, k]
        a = acc[:, k] if acc.shape[0] > 0 else np.zeros(1)
        feats.extend([
            float(np.mean(np.abs(v))),
            float(np.max(np.abs(v))),
            float(np.std(v)),
            float(np.max(np.abs(a))) if len(a) else 0.0,
        ])
    # ROM ratios (shoulder/elbow, wrist/elbow) - subject-scale-ish invariant
    rom = ang.max(0) - ang.min(0)
    rom_safe = rom + 1e-8
    feats.append(float(rom[1] / rom_safe[0]))  # shoulder/elbow
    feats.append(float(rom[2] / rom_safe[0]))  # wrist/elbow
    feats.append(float(rom[3] / rom_safe[1]))  # torso/shoulder
    feats.append(float(np.std(rom)))            # ROM dispersion

    # cross-joint correlation: shoulder-elbow, wrist-shoulder
    feats.append(float(np.corrcoef(ang[:, 0], ang[:, 1])[0, 1]))
    feats.append(float(np.corrcoef(ang[:, 1], ang[:, 2])[0, 1]))
    feats.append(float(np.corrcoef(ang[:, 0], ang[:, 2])[0, 1]))

    # skew/kurtosis per joint (5)
    for k in range(K):
        feats.append(float(stats.skew(ang[:, k])))
    for k in range(K):
        feats.append(float(stats.kurtosis(ang[:, k])))

    # smoothness: velocity sign changes per joint
    for k in range(K):
        s = np.sign(vel[:, k])
        feats.append(float(np.sum(s[1:] != s[:-1]) / max(len(s) - 1, 1)))

    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = os.path.join(args.in_dir, "manifest.csv")
    rows = list(csv.DictReader(open(manifest, newline="")))
    X, y, clip_ids, groups = [], [], [], []
    for row in rows:
        clip_id = os.path.splitext(os.path.basename(row["clip_path"]))[0]
        npy = os.path.join(args.in_dir, f"{clip_id}.npy")
        if not os.path.exists(npy):
            continue
        seq = np.load(npy)
        feat = biomech_features(seq)
        X.append(feat)
        y.append(row["serve_type"].strip().lower())
        clip_ids.append(clip_id)
        # Beg4-style ids have no underscore subject prefix; fallback
        gid = clip_id.split("_")[0] if "_" in clip_id else clip_id
        groups.append(gid)
    X = np.stack(X)
    np.savez(args.out, X=X, y=np.array(y), clip_ids=np.array(clip_ids),
             groups=np.array(groups))
    print(f"Extracted {X.shape[0]} clips, {X.shape[1]} biomech features -> {args.out}")

if __name__ == "__main__":
    main()
