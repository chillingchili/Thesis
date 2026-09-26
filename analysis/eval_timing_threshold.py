"""
eval_timing_threshold.py

Diagnostic #1: confidence/abstention analysis for the timing-SVM holdout.
Answers: are lob errors near the threshold (abstain fixes them) or
confidently wrong (need better features)?

Also dumps GRU-vs-SVM disagreement for a cheap ensemble-complementarity check.
"""

import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

CLASSES = ["drive", "lob", "topspin"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_svm_probs(model_path, X):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["clf"]
    proba = clf.predict_proba(X)
    return proba, clf.classes_


def threshold_sweep(y, proba, classes, thresholds):
    preds_idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    print(f"\n{'thr':>5} {'coverage':>9} {'acc_all':>8} {'acc_conf':>9} "
          f"{'abstain':>8} {'lob_recall_conf':>16}")
    for thr in thresholds:
        keep = conf >= thr
        cov = keep.mean()
        # all clips: abstain counted wrong
        pred_abstain = np.where(keep, preds_idx, -1)
        acc_all = (pred_abstain == y).mean()
        if keep.sum() == 0:
            print(f"{thr:5.2f} {cov:9.3f} {acc_all:8.3f} {'n/a':>9} "
                  f"{1-keep.sum()/len(y):8.3f} {'n/a':>16}")
            continue
        acc_conf = accuracy_score(y[keep], preds_idx[keep])
        # lob recall among confident clips only
        lob_mask = (y == CLASS_TO_IDX["lob"]) & keep
        if lob_mask.sum() > 0:
            lob_pred = preds_idx[lob_mask]
            lob_rec = (lob_pred == CLASS_TO_IDX["lob"]).mean()
        else:
            lob_rec = float("nan")
        n_abst = int((~keep).sum())
        print(f"{thr:5.2f} {cov:9.3f} {acc_all:8.3f} {acc_conf:9.3f} "
              f"{n_abst:8d} {lob_rec:16.3f}")


def error_confidence_report(y, proba, classes, name):
    preds = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    print(f"\n=== {name}: error confidence breakdown ===")
    for ci, cname in enumerate(classes):
        mask = y == ci
        if mask.sum() == 0:
            continue
        correct = preds[mask] == ci
        conf_c = conf[mask]
        wrong_conf = conf_c[~correct]
        right_conf = conf_c[correct]
        print(f"\n{cname} (n={mask.sum()}): "
              f"correct={correct.sum()} wrong={(~correct).sum()}")
        if len(wrong_conf):
            print(f"  wrong  conf: mean={wrong_conf.mean():.3f} "
                  f"med={np.median(wrong_conf):.3f} "
                  f"p25={np.percentile(wrong_conf,25):.3f} "
                  f"p75={np.percentile(wrong_conf,75):.3f} "
                  f">=0.5:{(wrong_conf>=0.5).mean():.2f} "
                  f">=0.6:{(wrong_conf>=0.6).mean():.2f} "
                  f">=0.7:{(wrong_conf>=0.7).mean():.2f}")
        if len(right_conf):
            print(f"  correct conf: mean={right_conf.mean():.3f} "
                  f"med={np.median(right_conf):.3f}")
        # where do wrong lob go?
        wrong_idx = np.where(mask & (preds != ci))[0]
        if len(wrong_idx) and cname == "lob":
            from collections import Counter
            dest = Counter(classes[preds[i]] for i in wrong_idx)
            dest_conf = {d: [] for d in dest}
            for i in wrong_idx:
                dest_conf[classes[preds[i]]].append(conf[i])
            for d in dest:
                cs = np.asarray(dest_conf[d], dtype=float)
                print(f"  lob -> {d}: n={len(cs)} conf mean={cs.mean():.3f} "
                      f"med={np.median(cs):.3f} min={cs.min():.3f} max={cs.max():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/timing_svm.pkl")
    parser.add_argument("--holdout", default="features/timing_test.npz")
    parser.add_argument("--gru_probs", default=None,
                        help="Optional .npz with gru_probs + y for complementarity")
    args = parser.parse_args()

    h = np.load(args.holdout, allow_pickle=True)
    Xh, y_str = h["X"], h["y"].astype(str)
    y = np.array([CLASS_TO_IDX[s] for s in y_str])
    clip_ids = h["clip_ids"].astype(str)

    proba, classes_order = load_svm_probs(args.model, Xh)
    # ensure column order matches CLASSES
    col = {str(c): i for i, c in enumerate(classes_order)}
    proba_c = np.column_stack([proba[:, col[c]] for c in CLASSES])

    print(f"Loaded {args.model}: n={len(y)}")
    print(f"Train-like sanity: mean conf={proba_c.max(axis=1).mean():.3f}")

    preds = proba_c.argmax(axis=1)
    print(f"\nHoldout acc (no abstain): {accuracy_score(y, preds):.3f}")
    print(classification_report(y, preds, labels=list(range(3)),
                                target_names=CLASSES, zero_division=0, digits=3))

    error_confidence_report(y, proba_c, CLASSES, "SVM timing")

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_sweep(y, proba_c, CLASSES, thresholds)

    # Save per-clip for inspection
    out = "analysis/timing_svm_holdout_detail.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "true", "pred", "conf",
                    "p_drive", "p_lob", "p_topspin", "correct"])
        for i in range(len(y)):
            w.writerow([
                clip_ids[i], CLASSES[y[i]], CLASSES[preds[i]],
                f"{proba_c[i].max():.4f}",
                f"{proba_c[i,0]:.4f}", f"{proba_c[i,1]:.4f}", f"{proba_c[i,2]:.4f}",
                int(preds[i] == y[i]),
            ])
    print(f"\nWrote {out}")

    # Cheap ensemble complementarity if GRU probs provided
    if args.gru_probs and os.path.exists(args.gru_probs):
        g = np.load(args.gru_probs, allow_pickle=True)
        gp = g["gru_probs"]
        gy = g["y"]
        assert np.array_equal(gy, y), "GRU/SVM y mismatch"
        gpred = gp.argmax(axis=1)
        spred = preds
        both_ok = (gpred == y) & (spred == y)
        svm_ok_gru_bad = (spred == y) & (gpred != y)
        gru_ok_svm_bad = (gpred == y) & (spred != y)
        both_bad = (gpred != y) & (spred != y)
        print("\n=== GRU vs SVM complementarity (holdout) ===")
        print(f"  both correct:     {both_ok.sum()}")
        print(f"  SVM right, GRU wrong: {svm_ok_gru_bad.sum()}  "
              f"(ensemble could keep SVM)")
        print(f"  GRU right, SVM wrong: {gru_ok_svm_bad.sum()}  "
              f"(ensemble upside)")
        print(f"  both wrong:       {both_bad.sum()}")
        if gru_ok_svm_bad.sum():
            idx = np.where(gru_ok_svm_bad)[0]
            print("  GRU-only correct clips:")
            for i in idx:
                print(f"    {clip_ids[i]} true={CLASSES[y[i]]} "
                      f"svm_pred={CLASSES[spred[i]]} conf={proba_c[i].max():.3f} "
                      f"gru_pred={CLASSES[gpred[i]]} gru_conf={gp[i].max():.3f}")


if __name__ == "__main__":
    main()
