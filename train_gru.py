"""
train_gru.py

Trains and stratified-5-fold cross-validates the GRU serve-type classifier
over BlazePose keypoint sequences.

Expected input:
  --keypoints_dir : the folder produced by extract_keypoints.py — one
                    <clip_id>.npy per clip plus a manifest.csv with
                    columns clip_path, serve_type (a "label" good/bad
                    column is fine to have too, but this script ignores it).

Note: serve_type (drive/lob/topspin) is the GRU's classification target.
Form quality (good/bad, from Coach B's annotation) is a separate signal
the rule engine uses later — it is NOT required to train this model, so
you don't need to wait on annotation to start this pipeline.

Usage:
  python train_gru.py --keypoints_dir data/keypoints --seq_len 180

Install (if needed):
  pip install tensorflow scikit-learn numpy --break-system-packages
"""

import argparse
import csv
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

CLASSES = ["drive", "lob", "topspin"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CONFIDENCE_THRESHOLD = 0.60  # softmax abstention threshold at inference time


def pad_or_truncate(kpts: np.ndarray, seq_len: int) -> np.ndarray:
    """Fix every clip to the same sequence length so they can be batched.
    Zero-pads short clips (masked out by the model) or truncates long ones."""
    n = kpts.shape[0]
    flat = kpts.reshape(n, -1)  # (frames, 66)
    flat = np.nan_to_num(flat, nan=0.0)

    if n >= seq_len:
        return flat[:seq_len]
    pad = np.zeros((seq_len - n, flat.shape[1]), dtype=np.float32)
    return np.concatenate([flat, pad], axis=0)


def load_dataset(keypoints_dir: str, seq_len: int):
    manifest_path = os.path.join(keypoints_dir, "manifest.csv")
    X, y, clip_ids = [], [], []

    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            clip_id = os.path.splitext(os.path.basename(row["clip_path"]))[0]
            npy_path = os.path.join(keypoints_dir, f"{clip_id}.npy")
            if not os.path.exists(npy_path):
                continue  # not extracted yet — fine while annotation is rolling

            kpts = np.load(npy_path)
            X.append(pad_or_truncate(kpts, seq_len))
            y.append(CLASS_TO_IDX[row["serve_type"].strip().lower()])
            clip_ids.append(clip_id)

    if not X:
        raise RuntimeError(
            "No usable clips found — check that extract_keypoints.py has "
            "run and that manifest.csv paths match the .npy filenames."
        )
    return np.stack(X), np.array(y), clip_ids


def build_model(seq_len: int, num_features: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len, num_features))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.GRU(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_with_abstention(model, X):
    """Applies the 0.60 confidence threshold: returns predicted class index,
    or -1 for 'abstain / insufficient confidence' when max softmax prob is
    below the threshold."""
    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)
    max_conf = probs.max(axis=1)
    preds = np.where(max_conf >= CONFIDENCE_THRESHOLD, preds, -1)
    return preds, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--out_dir", default="gru_runs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, y, clip_ids = load_dataset(args.keypoints_dir, args.seq_len)
    print(f"Loaded {len(X)} clips: {dict(zip(CLASSES, np.bincount(y, minlength=3)))}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = build_model(args.seq_len, X.shape[-1], len(CLASSES))
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
        )

        preds, probs = predict_with_abstention(model, X_val)
        # For the CV accuracy metric, evaluate on argmax regardless of the
        # abstention threshold (the threshold is an inference-time policy,
        # not part of the classifier's own accuracy).
        hard_preds = probs.argmax(axis=1)

        acc = accuracy_score(y_val, hard_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, hard_preds, labels=list(range(len(CLASSES))), zero_division=0
        )

        print(f"Fold {fold + 1} accuracy: {acc:.3f}")
        for c, p, r, f in zip(CLASSES, precision, recall, f1):
            print(f"  {c:8s}  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")

        fold_accuracies.append(acc)
        model.save(os.path.join(args.out_dir, f"gru_fold{fold + 1}.keras"))

    mean_acc = float(np.mean(fold_accuracies))
    print(f"\nMean stratified 5-fold accuracy: {mean_acc:.3f}")
    print(
        "Meets the 85% target."
        if mean_acc >= 0.85
        else "Below the 85% target — revisit data/augmentation before moving on."
    )


if __name__ == "__main__":
    main()
