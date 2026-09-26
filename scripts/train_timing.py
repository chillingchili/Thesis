"""
train_timing.py

Trains a small classifier on peak/trough timing features (option 4) with
StratifiedGroupKFold (leave-one-participant-out style) and evaluates on a
sealed holdout. Uses sklearn only — fast, no deep-net overfitting.

Usage:
  # CV on training set
  python scripts/train_timing.py --train features/timing_train.npz --cv

  # Fit on all train, evaluate holdout
  python scripts/train_timing.py --train features/timing_train.npz --holdout features/timing_test.npz \
      --model timing_model.joblib
"""

import argparse
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["drive", "lob", "topspin"]


def make_clf(kind: str, seed: int = 42):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=-1
        )
    if kind == "gb":
        return GradientBoostingClassifier(random_state=seed)
    if kind == "svm":
        return make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
        )
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    raise ValueError(kind)


def cv_eval(X, y, groups, kind: str, n_folds: int = 3):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        clf = make_clf(kind)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[va])
        acc = accuracy_score(y[va], pred)
        accs.append(acc)
        print(f"  Fold {fold + 1} ({sorted(set(groups[va]))}): acc={acc:.3f}")
        print(
            classification_report(
                y[va], pred, labels=CLASSES, target_names=CLASSES, zero_division=0, digits=3
            )
        )
    print(f"Mean {kind} group-CV acc: {np.mean(accs):.3f}  folds={[round(a, 3) for a in accs]}")
    return float(np.mean(accs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="timing features .npz for train")
    parser.add_argument("--holdout", default=None, help="timing features .npz for holdout")
    parser.add_argument("--cv", action="store_true", help="Run group-CV only")
    parser.add_argument(
        "--kind", default="rf", choices=["rf", "gb", "svm", "lr"], help="Classifier"
    )
    parser.add_argument("--model", default=None, help="Where to save final model (.pkl)")
    parser.add_argument("--n_folds", type=int, default=3)
    args = parser.parse_args()

    d = np.load(args.train, allow_pickle=True)
    X, y, groups = d["X"], d["y"].astype(str), d["groups"].astype(str)
    print(f"Train: {X.shape} classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Participants: {sorted(set(groups))}")

    if args.cv:
        cv_eval(X, y, groups, args.kind, args.n_folds)
        if not args.holdout:
            return

    clf = make_clf(args.kind)
    clf.fit(X, y)
    train_pred = clf.predict(X)
    print(f"Train acc (fit all): {accuracy_score(y, train_pred):.3f}")

    if args.model:
        with open(args.model, "wb") as f:
            pickle.dump({"clf": clf, "classes": CLASSES, "kind": args.kind}, f)
        print(f"Saved model -> {args.model}")

    if args.holdout:
        h = np.load(args.holdout, allow_pickle=True)
        Xh, yh = h["X"], h["y"].astype(str)
        pred = clf.predict(Xh)
        acc = accuracy_score(yh, pred)
        print(f"\nHoldout acc ({args.kind}): {acc:.3f}  n={len(yh)}")
        print(
            classification_report(
                yh, pred, labels=CLASSES, target_names=CLASSES, zero_division=0, digits=3
            )
        )
        cm = confusion_matrix(yh, pred, labels=CLASSES)
        print("Confusion (rows=true, cols=pred):")
        print("            " + "  ".join(f"{c:>8s}" for c in CLASSES))
        for i, row in enumerate(cm):
            print(f"  {CLASSES[i]:>8s}  " + "  ".join(f"{v:8d}" for v in row))


if __name__ == "__main__":
    main()
