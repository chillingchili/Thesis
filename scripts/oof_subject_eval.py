"""
oof_subject_eval.py

Answers: does the phone's model identify each training subject's serve classes
— especially CoachA lob — on our own recorded tests?

Two protocols (both on the 438 train clips only, no holdout involved):
  ENSEMBLE - phone config: probs = mean of all 5 strat5 folds, evaluated on
             every clip. (4 of 5 folds trained on any given clip → measures
             in-distribution capacity, what the app mostly sees live)
  OOF      - each clip predicted only by its fold's val model (honest
             never-trained-on-this-clip view; matches thesis CV protocol)
  HYBRID-OOF - OOF GRU probs averaged with OOF kNN5 timing probs (phone HYBRID)

Prints per-subject x class recall + where true lob clips go.

Usage (from project root):
  python scripts/oof_subject_eval.py
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from train_gru import (  # noqa: E402
    CLASS_TO_IDX,
    CLASSES,
    load_dataset,
)

KEYPOINTS_DIR = "data/training/keypoints_angles"
SEQ_LEN = 128
N_FOLDS = 5
SEED = 42


def load_models():
    import tensorflow as tf
    from train_gru import SequenceAugment

    models = []
    for i in range(1, N_FOLDS + 1):
        p = f"models/gru_runs_angles_strat5/gru_fold{i}.keras"
        m = tf.keras.models.load_model(
            p, custom_objects={"SequenceAugment": SequenceAugment}, compile=False
        )
        models.append(m)
    return models


def per_subject_table(clip_ids, y, probs, title):
    preds = probs.argmax(axis=1)
    subjects = sorted({c.split("_")[0] for c in clip_ids})
    subj = np.array([c.split("_")[0] for c in clip_ids])

    print(f"\n=== {title} ===")
    print(f"overall acc: {(preds == y).mean():.3f}  (n={len(y)})")
    hdr = f"{'subject':<10} {'n':>4} " + " ".join(f"{c:>9}" for c in CLASSES)
    print(hdr)
    for s in subjects:
        m = subj == s
        row = f"{s:<10} {m.sum():>4} "
        for ci in range(len(CLASSES)):
            cm = m & (y == ci)
            rec = (preds[cm] == ci).mean() if cm.sum() else float("nan")
            row += f"{rec:>9.2f} "
        print(row)

    # where do true lob clips go?
    lob = y == CLASS_TO_IDX["lob"]
    print("true lob -> predicted:", end=" ")
    parts = []
    for ci, c in enumerate(CLASSES):
        parts.append(f"{c}={int((preds[lob] == ci).sum())}")
    print("  ".join(parts))
    for s in subjects:
        sm = lob & (subj == s)
        if sm.sum():
            sub = "  ".join(
                f"{c}={int((preds[sm] == ci).sum())}"
                for ci, c in enumerate(CLASSES)
            )
            mean_plob = probs[sm, CLASS_TO_IDX["lob"]].mean()
            print(f"  {s} lob: n={int(sm.sum())}  {sub}  mean p(lob)={mean_plob:.2f}")


def main():
    X, y, clip_ids, groups = load_dataset(KEYPOINTS_DIR, SEQ_LEN)
    print(f"Loaded {len(X)} clips; subjects={sorted(set(groups))}")

    models = load_models()
    ens_probs = np.mean(
        [m.predict(X, batch_size=64, verbose=0) for m in models], axis=0
    )

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y))

    oof_probs = np.zeros_like(ens_probs)
    for i, (tr, va) in enumerate(folds):
        oof_probs[va] = models[i].predict(X[va], batch_size=64, verbose=0)

    per_subject_table(clip_ids, y, ens_probs,
                      "ENSEMBLE (phone config, resubstitution-ish)")
    per_subject_table(clip_ids, y, oof_probs,
                      "OOF GRU (each clip from a fold that never saw it)")

    # Hybrid OOF: kNN5 timing features, same folds
    td = np.load("features/timing_train.npz", allow_pickle=True)
    t_ids = [str(c) for c in td["clip_ids"]]
    t_X = td["X"].astype(np.float32)
    pos = {c: i for i, c in enumerate(t_ids)}
    if all(c in pos for c in clip_ids):
        align = np.array([pos[c] for c in clip_ids])
        t_X = t_X[align]
        knn_oof = np.zeros_like(ens_probs)
        for tr, va in folds:
            knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
            knn.fit(t_X[tr], y[tr])
            knn_oof[va] = knn.predict_proba(t_X[va])
        classes_idx = list(knn.classes_)  # knn may omit a class in tiny splits
        full = np.zeros_like(knn_oof)
        for j, ci in enumerate(classes_idx):
            full[:, int(ci)] = knn_oof[:, j]
        hybrid = 0.5 * oof_probs + 0.5 * full
        per_subject_table(clip_ids, y, hybrid,
                          "HYBRID-OOF (0.5*GRU_OOF + 0.5*kNN5_OOF) = phone HYBRID")
    else:
        print("\n(kNN alignment failed — skipping hybrid)")


if __name__ == "__main__":
    main()
