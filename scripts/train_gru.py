"""
train_gru.py

Trains and cross-validates the GRU serve-type classifier over joint-angle
sequences (extract_joint_angles.py), using StratifiedGroupKFold so that no
participant's clips ever appear in both the training and validation side
of a fold.

Participant is inferred as the part of the clip filename before the first
underscore (e.g. "Beginner1_Drive_03" -> "Beginner1").

With only 3 participants, use --n_folds 3 (leave-one-participant-out style).
n_splits > n_groups yields empty validation folds.

Expected input:
  --keypoints_dir : folder of <clip_id>.npy + manifest.csv (clip_path, serve_type).
  Joint-angle files are (frames, 5, 2) sin/cos; flattened to 10 features by
  pad_or_truncate. Coordinate files (frames, 33, 2) also work (66 features).

Usage:
  python train_gru.py --keypoints_dir data/training/keypoints_angles \
      --seq_len 128 --n_folds 3 --dropout 0.05 --out_dir gru_runs_angles

Install (if needed):
  pip install tensorflow scikit-learn numpy --break-system-packages
"""

import argparse
import csv
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

CLASSES = ["drive", "lob", "topspin"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CONFIDENCE_THRESHOLD = 0.60  # softmax abstention threshold at inference time


def participant_id_from_clip(clip_id: str) -> str:
    """e.g. 'Beginner1_Drive_03' -> 'Beginner1'. Used to keep each
    participant entirely on one side of a CV fold."""
    return clip_id.split("_")[0]


def pad_or_truncate(kpts: np.ndarray, seq_len: int) -> np.ndarray:
    """Fix every clip to the same sequence length so they can be batched.
    Zero-pads short clips (masked out by the model) or truncates long ones."""
    n = kpts.shape[0]
    flat = kpts.reshape(n, -1)
    flat = np.nan_to_num(flat, nan=0.0)

    if n >= seq_len:
        return flat[:seq_len]
    pad = np.zeros((seq_len - n, flat.shape[1]), dtype=np.float32)
    return np.concatenate([flat, pad], axis=0).astype(np.float32)


def load_dataset(keypoints_dir: str, seq_len: int):
    manifest_path = os.path.join(keypoints_dir, "manifest.csv")
    X, y, clip_ids, groups = [], [], [], []

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
            groups.append(participant_id_from_clip(clip_id))

    if not X:
        raise RuntimeError(
            "No usable clips found — check that extract_keypoints.py / "
            "extract_joint_angles.py has run and manifest paths match .npy names."
        )
    return np.stack(X), np.array(y), clip_ids, np.array(groups)


@tf.keras.utils.register_keras_serializable(package="thesis")
class SequenceAugment(tf.keras.layers.Layer):
    """Training-only augmentation for flattened sin/cos angle sequences.

    - Time warp: resamples the valid region at a random rate (speed jitter).
    - Temporal shift: slides the clip by a few frames.
    - Frame dropout: zeros random valid frames (Masking treats them as pad).
    - Angular jitter: Gaussian noise on each joint angle (radians), re-encoded.
    - Feature noise: small additive noise on flattened features.
    Disabled automatically when training=False (eval / testrun).
    """

    def __init__(
        self,
        angle_noise: float = 0.15,
        feature_noise: float = 0.03,
        rate_min: float = 0.75,
        rate_max: float = 1.35,
        max_shift: int = 8,
        frame_drop: float = 0.05,
        style_offset: float = 0.0,
        style_scale: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.angle_noise = angle_noise
        self.feature_noise = feature_noise
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.max_shift = max_shift
        self.frame_drop = frame_drop
        self.style_offset = style_offset
        self.style_scale = style_scale

    def call(self, inputs, training=None):
        if not training:
            return inputs
        x = tf.cast(inputs, tf.float32)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        F = tf.shape(x)[2]
        t_out = tf.cast(tf.range(T), tf.float32)

        # Valid = not all-zero row (trailing pad is zeros).
        valid = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)  # (B, T)
        idx = tf.cast(tf.range(T), tf.int32)
        last = tf.reduce_max(tf.cast(valid, tf.int32) * idx, axis=1)  # (B,)
        L = last + 1  # (B,)
        L = tf.maximum(L, 2)

        # --- time warp: stretch/compress valid region ---
        rate = tf.random.uniform((B,), self.rate_min, self.rate_max)
        new_L = tf.cast(tf.round(tf.cast(L, tf.float32) * rate), tf.int32)
        new_L = tf.clip_by_value(new_L, 2, T)
        Lf = tf.cast(L, tf.float32)[:, None]
        newLf = tf.cast(new_L, tf.float32)[:, None]
        denom = tf.maximum(newLf - 1.0, 1.0)
        src_f = t_out[None, :] * (Lf - 1.0) / denom
        src = tf.cast(tf.clip_by_value(src_f, 0.0, tf.cast(T - 1, tf.float32)), tf.int32)
        batch_ix = tf.tile(tf.range(B)[:, None], [1, T])
        x = tf.gather_nd(x, tf.stack([batch_ix, src], axis=-1))
        keep_t = tf.cast(t_out[None, :] < tf.cast(new_L, tf.float32)[:, None], tf.float32)
        x = x * keep_t[..., None]

        # --- temporal shift (edge zero-fill; Masking ignores those frames) ---
        shift = tf.random.uniform(
            (B,), -self.max_shift, self.max_shift + 1, dtype=tf.int32
        )
        src_t = tf.cast(t_out[None, :] - tf.cast(shift, tf.float32)[:, None], tf.int32)
        in_range = (src_t >= 0) & (src_t < T)
        src_t_c = tf.clip_by_value(src_t, 0, T - 1)
        x = tf.gather_nd(x, tf.stack([batch_ix, src_t_c], axis=-1))
        x = x * tf.cast(in_range, tf.float32)[..., None]

        # --- frame dropout on valid rows only ---
        if self.frame_drop > 0:
            is_valid = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
            drop = tf.random.uniform((B, T)) < self.frame_drop
            x = x * tf.cast(~(drop & is_valid), tf.float32)[..., None]

        # Keep padding and dropped frames masked through angle encoding/noise.
        # atan2(0, 0) -> 0 -> cos(0) = 1 would otherwise invent valid poses.
        frame_valid = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1, keepdims=True)
        # --- angle noise (stay on unit circle) + feature noise ---
        shape = tf.shape(x)
        n_ang = shape[2] // 2
        x = tf.reshape(x, (shape[0], shape[1], n_ang, 2))
        ang = tf.atan2(x[..., 0], x[..., 1])
        # Subject-style: random per-joint constant offset (posture bias)
        if self.style_offset > 0:
            offset = tf.random.normal((shape[0], 1, n_ang), stddev=self.style_offset)
            ang = ang + offset
        # Subject-style: random per-joint amplitude scale around clip mean
        if self.style_scale > 0:
            mean_a = tf.reduce_mean(ang, axis=1, keepdims=True)
            scale = 1.0 + tf.random.normal((shape[0], 1, n_ang), stddev=self.style_scale)
            scale = tf.clip_by_value(scale, 0.5, 1.8)
            ang = mean_a + (ang - mean_a) * scale
        ang = ang + tf.random.normal(tf.shape(ang), stddev=self.angle_noise)
        x = tf.stack([tf.sin(ang), tf.cos(ang)], axis=-1)
        x = tf.reshape(x, shape)
        x = x + tf.random.normal(tf.shape(x), stddev=self.feature_noise)
        return tf.where(frame_valid, x, tf.zeros_like(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "angle_noise": self.angle_noise,
                "feature_noise": self.feature_noise,
                "rate_min": self.rate_min,
                "rate_max": self.rate_max,
                "max_shift": self.max_shift,
                "frame_drop": self.frame_drop,
                "style_offset": self.style_offset,
                "style_scale": self.style_scale,
            }
        )
        return cfg


def build_model(
    seq_len: int,
    num_features: int,
    num_classes: int,
    hidden_units: int,
    l2: float,
    dropout: float,
    recurrent_dropout: float,
    angle_noise: float,
    feature_noise: float,
    rate_min: float,
    rate_max: float,
    max_shift: int,
    frame_drop: float,
    style_offset: float = 0.0,
    style_scale: float = 0.0,
) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    inputs = tf.keras.Input(shape=(seq_len, num_features))
    x = SequenceAugment(
        angle_noise=angle_noise,
        feature_noise=feature_noise,
        rate_min=rate_min,
        rate_max=rate_max,
        max_shift=max_shift,
        frame_drop=frame_drop,
        style_offset=style_offset,
        style_scale=style_scale,
        name="seq_augment",
    )(inputs)
    x = tf.keras.layers.Masking(mask_value=0.0)(x)
    x = tf.keras.layers.GRU(
        hidden_units,
        return_sequences=False,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        recurrent_dropout=recurrent_dropout,
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", kernel_regularizer=reg
    )(x)

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
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_folds", type=int, default=3,
                        help="Group CV: must be <= participants (3). "
                             "Thesis-style stratified CV uses --cv stratified --n_folds 5.")
    parser.add_argument("--cv", choices=["group", "stratified"], default="group",
                        help="group = StratifiedGroupKFold (subject-held-out, honest). "
                             "stratified = plain StratifiedKFold (thesis Section 4.3.2 metric).")
    parser.add_argument("--hidden_units", type=int, default=32,
                        help="GRU size (32 to reduce overfitting on a small subject pool)")
    parser.add_argument("--l2", type=float, default=1e-4,
                        help="L2 regularization strength")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dense-layer dropout (0.05; raise to 0.2-0.3 if train/val gap stays large)")
    parser.add_argument("--recurrent_dropout", type=float, default=0.0,
                        help="GRU recurrent dropout (0 = off; avoids slowdown unless needed)")
    parser.add_argument("--angle_noise", type=float, default=0.15,
                        help="Aug: Gaussian noise (radians) on joint angles during training")
    parser.add_argument("--feature_noise", type=float, default=0.03,
                        help="Aug: Gaussian noise on flattened features during training")
    parser.add_argument("--rate_min", type=float, default=0.75,
                        help="Aug: min time-warp speed factor")
    parser.add_argument("--rate_max", type=float, default=1.35,
                        help="Aug: max time-warp speed factor")
    parser.add_argument("--max_shift", type=int, default=8,
                        help="Aug: max temporal shift in frames")
    parser.add_argument("--frame_drop", type=float, default=0.05,
                        help="Aug: probability of zeroing a valid frame")
    parser.add_argument("--style_offset", type=float, default=0.0,
                        help="Aug: subject-style random per-joint angle offset (radians std)")
    parser.add_argument("--style_scale", type=float, default=0.0,
                        help="Aug: subject-style random per-joint amplitude scale (std)")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience on val_loss")
    parser.add_argument("--out_dir", default="gru_runs_angles")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, y, clip_ids, groups = load_dataset(args.keypoints_dir, args.seq_len)
    n_groups = len(set(groups))
    print(f"Loaded {len(X)} clips: {dict(zip(CLASSES, np.bincount(y, minlength=3)))}")
    print(f"Participants ({n_groups}): {sorted(set(groups))}")
    print(f"Features/step: {X.shape[-1]}  seq_len: {X.shape[1]}")
    print(
        f"Aug: angle_noise={args.angle_noise} feature_noise={args.feature_noise} "
        f"time_warp=[{args.rate_min},{args.rate_max}] shift={args.max_shift} "
        f"frame_drop={args.frame_drop} style_off={args.style_offset} "
        f"style_scale={args.style_scale} | dropout={args.dropout} l2={args.l2} "
        f"hidden={args.hidden_units}"
    )

    if args.cv == "group":
        if args.n_folds > n_groups:
            raise SystemExit(
                f"n_folds={args.n_folds} > n_participants={n_groups}. "
                f"StratifiedGroupKFold would produce empty validation folds. "
                f"Use --n_folds {n_groups}."
            )
        splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)

    fold_accuracies = []
    fold_val_losses_end = []

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
        print(f"  train participants: {sorted(set(groups[train_idx]))}")
        print(f"  val participants:   {sorted(set(groups[val_idx]))}")
        if len(val_idx) == 0:
            raise SystemExit("Empty validation fold — reduce --n_folds.")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = build_model(
            args.seq_len,
            X.shape[-1],
            len(CLASSES),
            args.hidden_units,
            args.l2,
            args.dropout,
            args.recurrent_dropout,
            args.angle_noise,
            args.feature_noise,
            args.rate_min,
            args.rate_max,
            args.max_shift,
            args.frame_drop,
            args.style_offset,
            args.style_scale,
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop],
            verbose=2,
        )
        stopped_epoch = len(history.history["loss"])
        best_val = float(np.min(history.history["val_loss"]))
        final_train = float(history.history["loss"][-1])
        print(f"  stopped after {stopped_epoch}/{args.epochs} epochs "
              f"(best val_loss={best_val:.4f}, final train_loss={final_train:.4f})")

        # Evaluate without augmentation (training=False inside SequenceAugment)
        _, probs = predict_with_abstention(model, X_val)
        hard_preds = probs.argmax(axis=1)

        # Also report train accuracy without augmentation (overfit gap check)
        _, train_probs = predict_with_abstention(model, X_train)
        train_acc = accuracy_score(y_train, train_probs.argmax(axis=1))

        acc = accuracy_score(y_val, hard_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, hard_preds, labels=list(range(len(CLASSES))), zero_division=0
        )

        print(f"Fold {fold + 1} val accuracy: {acc:.3f}  (train acc without aug: {train_acc:.3f})")
        for c, p, r, f in zip(CLASSES, precision, recall, f1):
            print(f"  {c:8s}  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")

        fold_accuracies.append(acc)
        fold_val_losses_end.append(best_val)
        model.save(os.path.join(args.out_dir, f"gru_fold{fold + 1}.keras"))

    mean_acc = float(np.mean(fold_accuracies))
    cv_label = "stratified" if args.cv == "stratified" else "participant-grouped"
    print(f"\nMean {cv_label} {args.n_folds}-fold accuracy: {mean_acc:.3f}")
    print(f"Fold val accs: {[round(a, 3) for a in fold_accuracies]}")
    print(
        "Meets the 85% target."
        if mean_acc >= 0.85
        else "Below the 85% target."
        + ("" if args.cv == "stratified" else
           " This leave-one-participant-out number is the honest baseline "
           "(holdout on Beginner3/4 will be similar or worse).")
    )


if __name__ == "__main__":
    main()
