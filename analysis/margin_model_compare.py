"""Margin + model comparison: SVM vs kNN on both holdouts."""
import csv
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

CLASSES = ["drive", "lob", "topspin"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def load_labels(path):
    return np.array([C2I[r["serve_type"].strip().lower()] for r in csv.DictReader(open(path))])


def knn_mean(Q, P, k=5):
    d = np.linalg.norm(P[None, :, :] - Q[:, None, :], axis=-1)
    kk = min(k, P.shape[0])
    return np.sort(d, axis=1)[:, :kk].mean(1)


Xtr = np.load("analysis/timing_train_feats.npy")
ytr = load_labels("analysis/timing_train_labels.csv")
mean, std = Xtr.mean(0), Xtr.std(0)
std = np.where(std < 1e-8, 1.0, std)
Xtr_z = (Xtr - mean) / std

with open("models/timing_svm.pkl", "rb") as f:
    svm = pickle.load(f)["clf"]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtr, ytr)

print("=" * 70)
print("MODEL COMPARISON (per-class recall)")
print("=" * 70)
print(f'{"holdout":8s} {"model":6s} {"acc":>6}  ' + "  ".join(f"{c:>8}" for c in CLASSES))
for name, fx, fy in [
    ("Beg3", "analysis/timing_beg3_feats.npy", "analysis/timing_beg3_labels.csv"),
    ("Beg4", "analysis/timing_beg4_o_feats.npy", "analysis/timing_beg4_o_labels.csv"),
]:
    X = np.load(fx)
    y = load_labels(fy)
    for mname, model in [("SVM", svm), ("kNN5", knn)]:
        pred = model.predict(X)
        # handle both string labels (SVM) and int labels (kNN)
        if pred.dtype.kind in "i":
            pred_idx = pred.astype(int)
        else:
            pred_idx = np.array([C2I[str(p)] for p in pred])
        acc = accuracy_score(y, pred_idx)
        rec = [(pred_idx[y == ci] == ci).mean() for ci in range(3)]
        print(f"{name:8s} {mname:6s} {acc:6.3f}  " + "  ".join(f"{r:8.3f}" for r in rec))
    print()

print("=" * 70)
print("MARGIN ANALYSIS (own-class dist vs nearest wrong-class dist)")
print("=" * 70)
for name, fx, fy in [
    ("Beg3", "analysis/timing_beg3_feats.npy", "analysis/timing_beg3_labels.csv"),
    ("Beg4", "analysis/timing_beg4_o_feats.npy", "analysis/timing_beg4_o_labels.csv"),
]:
    X = np.load(fx)
    y = load_labels(fy)
    X_z = (X - mean) / std

    dists = {c: knn_mean(X_z, Xtr_z[ytr == C2I[c]]) for c in CLASSES}
    D = np.stack([dists[c] for c in CLASSES], axis=1)

    own = D[np.arange(len(y)), y]
    other = D.copy()
    other[np.arange(len(y)), y] = np.inf
    nearest_wrong = other.min(axis=1)
    margin = own - nearest_wrong
    near = D.argmin(axis=1)

    pred = svm.predict(X)
    if pred.dtype.kind in "i":
        pred_idx = pred.astype(int)
    else:
        pred_idx = np.array([C2I[str(p)] for p in pred])
    wrong = pred_idx != y

    print(f"\n== {name} ==  (SVM wrong n = {wrong.sum()})")
    if wrong.any():
        print(f"  margin on WRONG: mean={margin[wrong].mean():.3f}  frac<0={(margin[wrong] < 0).mean():.3f}")
        print(f"  margin on RIGHT: mean={margin[~wrong].mean():.3f}  frac<0={(margin[~wrong] < 0).mean():.3f}")
        from collections import Counter
        cls_arr = np.array(CLASSES)
        print("  nearest-train-class among WRONG:", dict(Counter(cls_arr[near[wrong]])))
        print("  nearest-train-class among RIGHT:", dict(Counter(cls_arr[near[~wrong]])))

    print(f'  {"class":10s} {"n":>4} {"own":>7} {"nearWrong":>9} {"margin":>7} {"m<0":>6} {"1NNacc":>7}')
    for ci, c in enumerate(CLASSES):
        m = y == ci
        if m.sum() == 0:
            continue
        acc = (near[m] == ci).mean()
        print(
            f"  {c:10s} {m.sum():4d} {own[m].mean():7.3f} {nearest_wrong[m].mean():9.3f} "
            f"{margin[m].mean():7.3f} {(margin[m] < 0).mean():6.3f} {acc:7.3f}"
        )
    print(f"  overall: margin<0={(margin < 0).mean():.3f}  1NN acc={(near == y).mean():.3f}")
