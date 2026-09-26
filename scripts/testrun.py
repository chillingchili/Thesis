"""
evaluate_holdout.py

Runs one or more trained GRU checkpoints against a sealed holdout set
(clips never touched during training or cross-validation) and reports
accuracy, per-class precision/recall/f1, a confusion matrix, and a list
of exactly which clips were missed or abstained on.

If you pass more than one --model, predictions are averaged across all of
them (a simple ensemble of your 5 CV folds) before applying the confidence
threshold -- usually a bit more robust than trusting a single fold.

Expected input:
  --keypoints_dir : holdout clips extracted with extract_keypoints.py,
                    with manifest.csv (clip_path, serve_type). This should
                    be your SEALED set (Beginner 3/4, Coach A's reserved
                    clips) -- not anything used in train_gru.py.

Usage:
  python scripts/testrun.py --keypoints_dir data/testing/keypoints_angles \
      --model models/gru_runs_angles_strat5/gru_fold1.keras models/gru_runs_angles_strat5/gru_fold2.keras \
              models/gru_runs_angles_strat5/gru_fold3.keras models/gru_runs_angles_strat5/gru_fold4.keras \
              models/gru_runs_angles_strat5/gru_fold5.keras
"""

import argparse
import csv
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Register custom layers used by train_gru.py so load_model can deserialize them.
try:
    from train_gru import SequenceAugment  # noqa: F401
except ImportError:
    pass

CLASSES = ["drive", "lob", "topspin"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CONFIDENCE_THRESHOLD = 0.60


def pad_or_truncate(kpts: np.ndarray, seq_len: int) -> np.ndarray:
    n = kpts.shape[0]
    flat = kpts.reshape(n, -1)
    flat = np.nan_to_num(flat, nan=0.0)
    if n >= seq_len:
        return flat[:seq_len]
    pad = np.zeros((seq_len - n, flat.shape[1]), dtype=np.float32)
    return np.concatenate([flat, pad], axis=0)


def load_holdout(keypoints_dir: str, seq_len: int):
    manifest_path = os.path.join(keypoints_dir, "manifest.csv")
    X, y, clip_ids = [], [], []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            clip_id = os.path.splitext(os.path.basename(row["clip_path"]))[0]
            npy_path = os.path.join(keypoints_dir, f"{clip_id}.npy")
            if not os.path.exists(npy_path):
                continue
            kpts = np.load(npy_path)
            X.append(pad_or_truncate(kpts, seq_len))
            y.append(CLASS_TO_IDX[row["serve_type"].strip().lower()])
            clip_ids.append(clip_id)
    if not X:
        raise RuntimeError("No usable clips found in holdout set.")
    return np.stack(X), np.array(y), clip_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_dir", required=True)
    parser.add_argument(
        "--model", nargs="+", required=True, help="One or more .keras checkpoints"
    )
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Confidence threshold for abstention (lower = fewer abstains)")
    args = parser.parse_args()

    X, y, clip_ids = load_holdout(args.keypoints_dir, args.seq_len)
    print(f"Loaded {len(X)} holdout clips: {dict(zip(CLASSES, np.bincount(y, minlength=3)))}")

    custom_objects = {}
    try:
        from train_gru import SequenceAugment
        custom_objects["SequenceAugment"] = SequenceAugment
    except ImportError:
        pass
    models = [tf.keras.models.load_model(m, custom_objects=custom_objects) for m in args.model]
    all_probs = np.mean([m.predict(X, verbose=0) for m in models], axis=0)

    preds = all_probs.argmax(axis=1)
    max_conf = all_probs.max(axis=1)
    abstained = max_conf < args.threshold

    print(f"\n{abstained.sum()}/{len(y)} clips abstained (below {args.threshold} confidence)")

    print("\n--- Scored on all clips (abstains counted as wrong) ---")
    print(classification_report(y, preds, target_names=CLASSES, zero_division=0))

    if 0 < abstained.sum() < len(y):
        print("--- Scored only on clips the model was confident about ---")
        confident = ~abstained
        print(
            classification_report(
                y[confident], preds[confident], target_names=CLASSES, zero_division=0
            )
        )

    cm = confusion_matrix(y, preds, labels=list(range(len(CLASSES))))
    print("Confusion matrix (rows=true, cols=predicted):")
    print("            " + "  ".join(f"{c:>8s}" for c in CLASSES))
    for true_idx, row in enumerate(cm):
        print(f"  {CLASSES[true_idx]:>8s}  " + "  ".join(f"{v:8d}" for v in row))

    print("\nClips the model got wrong or abstained on:")
    misses = 0
    for clip_id, true_idx, pred_idx, conf, abst in zip(
        clip_ids, y, preds, max_conf, abstained
    ):
        if abst or pred_idx != true_idx:
            misses += 1
            tag = "ABSTAIN" if abst else "WRONG"
            print(
                f"  [{tag:7s}] {clip_id:30s} true={CLASSES[true_idx]:8s} "
                f"pred={CLASSES[pred_idx]:8s} conf={conf:.3f}"
            )
    if misses == 0:
        print("  (none)")


if __name__ == "__main__":
    main()
