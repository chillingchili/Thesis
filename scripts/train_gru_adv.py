"""Real change 2: adversarial subject-invariance GRU (DANN-style).

Trains serve-type classifier while a gradient-reversal adversary tries to
predict participant ID from the same backbone. Encourages embeddings that
discard subject-specific style. Evaluated with group CV + sealed Beg3/Beg4.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import tensorflow as tf

from train_gru import CLASSES, load_dataset, participant_id_from_clip

tf.get_logger().setLevel("ERROR")

NUM_PARTICIPANTS = 3  # Beginner1, Beginner2, CoachA


class GradientReversal(tf.keras.layers.Layer):
    @tf.keras.utils.register_keras_serializable(package="thesis")
    def __init__(self, lambda_, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = float(lambda_)

    def call(self, x, training=None):
        @tf.custom_gradient
        def _rev(z):
            def grad(dy):
                return -self.lambda_ * dy
            return z, grad
        return _rev(x)

    def get_config(self):
        cfg = super().get_config()
        cfg["lambda_"] = self.lambda_
        return cfg


def pad_or_truncate(kpts, seq_len):
    n = kpts.shape[0]
    flat = kpts.reshape(n, -1)
    flat = np.nan_to_num(flat, nan=0.0)
    if n >= seq_len:
        return flat[:seq_len]
    pad = np.zeros((seq_len - n, flat.shape[1]), dtype=np.float32)
    return np.concatenate([flat, pad], axis=0)


def build_adv_model(
    seq_len,
    num_features,
    num_classes,
    hidden,
    dropout,
    lambda_,
    aug_kwargs=None,
):
    inputs = tf.keras.Input(shape=(seq_len, num_features))
    if aug_kwargs:
        from train_gru import SequenceAugment

        x = SequenceAugment(**aug_kwargs, name="seq_augment")(inputs)
    else:
        x = inputs
    x = tf.keras.layers.Masking(mask_value=0.0)(x)
    h = tf.keras.layers.GRU(hidden, return_sequences=False, name="gru")(x)
    h = tf.keras.layers.Dropout(dropout)(h)

    serve_logits = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="serve"
    )(h)

    rev = GradientReversal(lambda_, name="grl")(h)
    subj = tf.keras.layers.Dense(hidden, activation="relu")(rev)
    subj = tf.keras.layers.Dropout(dropout)(subj)
    subj_logits = tf.keras.layers.Dense(
        NUM_PARTICIPANTS, activation="softmax", name="subject"
    )(subj)
    return tf.keras.Model(inputs, [serve_logits, subj_logits], name="adv_gru")


class AdvModel(tf.keras.Model):
    def __init__(self, backbone, lambda_):
        super().__init__()
        self.backbone = backbone
        self.lambda_ = lambda_

    def call(self, x, training=None):
        return self.backbone(x, training=training)

    def train_step(self, data):
        x, (y_serve, y_subj) = data
        with tf.GradientTape() as tape:
            serve_pred, subj_pred = self.backbone(x, training=True)
            loss_s = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_serve, serve_pred, from_logits=False
                )
            )
            loss_j = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_subj, subj_pred, from_logits=False
                )
            )
            # Adversary loss is flipped by GRL; total = Ls + Lj (GRL reverses Lj grads)
            loss = loss_s + loss_j
        grads = tape.gradient(loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.backbone.trainable_variables))
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(serve_pred, 1), tf.cast(y_serve, tf.int64)),
                tf.float32,
            )
        )
        # Subject accuracy: ~1/3 is good (adversary confused), high = still leaks subject
        subj_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(subj_pred, 1), tf.cast(y_subj, tf.int64)),
                tf.float32,
            )
        )
        return {
            "loss": loss,
            "serve_acc": acc,
            "subj_acc": subj_acc,
            "L_serve": loss_s,
            "L_subj": loss_j,
        }

    def test_step(self, data):
        x, (y_serve, y_subj) = data
        serve_pred, subj_pred = self.backbone(x, training=False)
        loss_s = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_serve, serve_pred, from_logits=False
            )
        )
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(serve_pred, 1), tf.cast(y_serve, tf.int64)),
                tf.float32,
            )
        )
        subj_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(subj_pred, 1), tf.cast(y_subj, tf.int64)),
                tf.float32,
            )
        )
        return {"val_loss": loss_s, "val_acc": acc, "val_subj_acc": subj_acc}


def make_ds(X, ys, yj, batch, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, (ys, yj)))
    if training:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True).batch(batch)
    else:
        ds = ds.batch(batch)
    return ds.prefetch(tf.data.AUTOTUNE)


def run_group_cv(X, ys, clip_ids, participants, args, aug_kwargs=None):
    print("======== ADVERSARIAL GROUP CV ========")
    # subject labels: map participant string -> int
    uniq = sorted(set(participants))
    pmap = {p: i for i, p in enumerate(uniq)}
    yj_all = np.array([pmap[p] for p in participants], dtype=np.int64)

    folds = []
    for val_p in uniq:
        val_idx = np.array([i for i, p in enumerate(participants) if p == val_p])
        tr_idx = np.array([i for i, p in enumerate(participants) if p != val_p])
        print(f"\n=== Leave {val_p} out ===")
        print(f"  train participants: {[participants[i] for i in tr_idx[:3]]}...")

        bb = build_adv_model(
            args.seq_len, X.shape[-1], 3, args.hidden_units, args.dropout, args.lambda_,
            aug_kwargs,
        )
        model = AdvModel(bb, args.lambda_)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            run_eagerly=False,
        )
        # Build variables
        model(X[:1], training=False)

        tr_ds = make_ds(
            X[tr_idx], ys[tr_idx], yj_all[tr_idx], args.batch_size, training=True
        )
        val_ds = make_ds(
            X[val_idx], ys[val_idx], yj_all[val_idx], args.batch_size, training=False
        )

        best_w, best_val, bad = None, 1e9, 0
        for epoch in range(args.epochs):
            hist = model.fit(
                tr_ds,
                epochs=1,
                verbose=0,
            )
            vlogs = model.evaluate(val_ds, verbose=0, return_dict=True)
            train_acc = float(hist.history["serve_acc"][0])
            train_subj = float(hist.history["subj_acc"][0])
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  epoch {epoch+1}: train serve={train_acc:.3f} "
                    f"subj={train_subj:.3f} | val serve={vlogs['val_acc']:.3f} "
                    f"subj={vlogs['val_subj_acc']:.3f}"
                )
            if vlogs["val_loss"] < best_val - 1e-4:
                best_val = vlogs["val_loss"]
                best_w = bb.get_weights()
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    if best_w is not None:
                        bb.set_weights(best_w)
                    print(f"  early stop at epoch {epoch+1}, best val_loss={best_val:.4f}")
                    break
        if best_w is not None:
            bb.set_weights(best_w)

        vlogs = model.evaluate(val_ds, verbose=0, return_dict=True)
        acc = float(vlogs["val_acc"])
        print(f"Fold {val_p}: serve acc={acc:.3f}  subj leak={vlogs['val_subj_acc']:.3f}")
        folds.append(acc)

        os.makedirs(args.out_dir, exist_ok=True)
        bb.save(os.path.join(args.out_dir, f"adv_{val_p}.keras"))

    print(f"\nMean adversarial group-CV: {np.mean(folds):.3f} folds={folds}")
    return float(np.mean(folds))


def load_holdout(keypoints_dir, seq_len):
    from train_gru import CLASSES as C
    cmap = {c: i for i, c in enumerate(C)}
    manifest = os.path.join(keypoints_dir, "manifest.csv")
    X, y, ids = [], [], []
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            clip_id = os.path.splitext(os.path.basename(row["clip_path"]))[0]
            npy = os.path.join(keypoints_dir, f"{clip_id}.npy")
            if not os.path.exists(npy):
                continue
            k = np.load(npy)
            X.append(pad_or_truncate(k, seq_len))
            y.append(cmap[row["serve_type"].strip().lower()])
            ids.append(clip_id)
    if not X:
        raise RuntimeError(f"no clips in {keypoints_dir}")
    return np.stack(X), np.array(y, dtype=np.int64), ids


def run_holdout(X, ys, participants, args, aug_kwargs=None):
    """Train one adv model per held-out training participant, ensemble on Beg3/Beg4."""
    print("\n======== ADVERSARIAL HOLDOUT ENSEMBLE ========")
    uniq = sorted(set(participants))
    pmap = {p: i for i, p in enumerate(uniq)}
    yj_all = np.array([pmap[p] for p in participants], dtype=np.int64)

    fold_models = []
    for val_p in uniq:
        tr_idx = np.array([i for i, p in enumerate(participants) if p != val_p])
        bb = build_adv_model(
            args.seq_len, X.shape[-1], 3, args.hidden_units, args.dropout, args.lambda_,
            aug_kwargs,
        )
        model = AdvModel(bb, args.lambda_)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr))
        model(X[:1], training=False)

        tr_ds = make_ds(
            X[tr_idx], ys[tr_idx], yj_all[tr_idx], args.batch_size, training=True
        )
        # simple train for fixed epochs (no val — small data)
        model.fit(tr_ds, epochs=args.epochs, verbose=0)
        fold_models.append(bb)

    os.makedirs(args.out_dir, exist_ok=True)
    for i, m in enumerate(fold_models):
        m.save(os.path.join(args.out_dir, f"adv_holdout_fold{i+1}.keras"))

    for name, dirpath in [
        ("Beg3", args.beg3_dir),
        ("Beg4", args.beg4_dir),
    ]:
        Xh, yh, _ = load_holdout(dirpath, args.seq_len)
        probs = []
        for m in fold_models:
            p, _ = m.predict(Xh, verbose=0)
            probs.append(p)
        avg = np.mean(probs, axis=0)
        preds = avg.argmax(1)
        acc = float(np.mean(preds == yh))
        print(f"{name}: adversarial ensemble acc={acc:.3f} n={len(yh)}")
        from sklearn.metrics import classification_report

        print(classification_report(yh, preds, target_names=CLASSES, zero_division=0))
    return fold_models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoints_dir", default="data/training/keypoints_angles")
    p.add_argument("--beg3_dir", default="data/testing/keypoints_angles")
    p.add_argument("--beg4_dir", default="data/testing/keypoints_beg4_angles")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--hidden_units", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lambda_", type=float, default=1.0, help="GRL strength")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--mode", choices=["cv", "holdout", "both"], default="both")
    p.add_argument("--out_dir", default="gru_adv_runs")
    p.add_argument("--strong_aug", action="store_true",
                   help="Enable strong subject-style augmentation")
    args = p.parse_args()

    aug_kwargs = None
    if args.strong_aug:
        aug_kwargs = dict(
            angle_noise=0.30,
            feature_noise=0.06,
            rate_min=0.65,
            rate_max=1.45,
            max_shift=12,
            frame_drop=0.08,
            style_offset=0.25,
            style_scale=0.20,
        )

    X, ys, clip_ids, groups = load_dataset(args.keypoints_dir, args.seq_len)
    participants = np.array([str(g) for g in groups])
    print(
        f"Loaded {len(X)} clips | participants={sorted(set(participants))} "
        f"| lambda={args.lambda_} hidden={args.hidden_units}"
    )

    if args.mode in ("cv", "both"):
        run_group_cv(X, ys, clip_ids, participants, args, aug_kwargs)
    if args.mode in ("holdout", "both"):
        run_holdout(X, ys, participants, args, aug_kwargs)


if __name__ == "__main__":
    main()
