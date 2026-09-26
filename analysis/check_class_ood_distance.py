"""
check_class_ood_distance.py

For each holdout clip, measures how far it sits (in your timing-feature
space) from its own true class's nearest training exemplars. If a class's
holdout distances are systematically much larger than the other classes'
in that SAME run -- and that pattern differs by subject (e.g. lob is
far-out for Beg3, but not for Beg4) -- that's direct evidence the failing
class is out-of-distribution for that specific person given how few
training subjects exist, not that the feature set structurally can't
separate that class from its neighbor.

Distance used: mean distance to the k nearest training clips of the same
true class (Euclidean, after z-scoring by the training set's own
mean/std so no single feature dominates just from having larger units).

Expected input:
  --train_features / --train_labels     : your training feature matrix
                                           and serve_type labels
  --holdout_features / --holdout_labels : ONE holdout subject's features
                                           and labels (run once per
                                           subject -- once for Beg3, once
                                           for Beg4 -- and compare output)
  --k : number of nearest training neighbors to average (default 5)

Usage:
  python check_class_ood_distance.py \
      --train_features train_feats.npy --train_labels train_labels.csv \
      --holdout_features beg3_feats.npy --holdout_labels beg3_labels.csv
"""

import argparse
import csv

import numpy as np

CLASSES = ["drive", "lob", "topspin"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_labels(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([CLASS_TO_IDX[r["serve_type"].strip().lower()] for r in rows])


def knn_distance(query: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    dists = np.linalg.norm(pool[np.newaxis, :, :] - query[:, np.newaxis, :], axis=-1)
    k = min(k, pool.shape[0])
    nearest = np.sort(dists, axis=1)[:, :k]
    return nearest.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--train_labels", required=True)
    parser.add_argument("--holdout_features", required=True)
    parser.add_argument("--holdout_labels", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    X_train = np.load(args.train_features)
    y_train = load_labels(args.train_labels)
    X_hold = np.load(args.holdout_features)
    y_hold = load_labels(args.holdout_labels)

    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_train_z = (X_train - mean) / std
    X_hold_z = (X_hold - mean) / std

    print(f"{'class':10s} {'holdout n':>10s} {'mean dist':>10s} {'median dist':>12s}")
    per_class_dists = {}
    for c, idx in CLASS_TO_IDX.items():
        train_pool = X_train_z[y_train == idx]
        hold_pts = X_hold_z[y_hold == idx]
        if len(hold_pts) == 0 or len(train_pool) == 0:
            continue
        dists = knn_distance(hold_pts, train_pool, args.k)
        per_class_dists[c] = dists
        print(f"{c:10s} {len(hold_pts):10d} {dists.mean():10.3f} {np.median(dists):12.3f}")

    if per_class_dists:
        overall = np.concatenate(list(per_class_dists.values()))
        print(f"\noverall mean distance: {overall.mean():.3f}")

    print("\nA class whose mean/median distance is well above the others here --")
    print("and whose ranking differs from this same check on your OTHER holdout")
    print("subject -- is out-of-distribution for THIS person specifically: a")
    print("subject-diversity problem, not evidence the feature set can't")
    print("separate that class in general.")


if __name__ == "__main__":
    main()
