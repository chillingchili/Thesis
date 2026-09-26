"""
train_gru_metric.py

Metric-learning GRU: embedding head + cosine classifier + center loss
(CE + lambda * center) to pull same-class embeddings together across
subjects. Evaluated with group CV (honest) and on sealed Beg3/Beg4 holdout.

Usage:
  python train_gru_metric.py --keypoints_dir data/training/keypoints_angles \
      --seq_len 128 --n_folds 3 --cv group --out_dir gru_metric_runs
  python train_gru_metric.py ... --cv stratified --n_folds 5   # thesis gate
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from train_gru import (
    CLASSES, CLASS_TO_IDX, SequenceAugment, load_dataset,
    pad_or_truncate, participant_id_from_clip,
)

EMBED_DIM = 32


class CosineHead(tf.keras.layers.Layer):
    """L2-normalized embedding x L2-normalized class weights, scaled."""
    def __init__(self, num_classes, scale=16.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(self.num_classes, int(input_shape[-1])),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, emb):
        emb_n = tf.nn.l2_normalize(emb, axis=1)
        w_n = tf.nn.l2_normalize(self.W, axis=1)
        return self.scale * tf.matmul(emb_n, w_n, transpose_b=True)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "scale": self.scale})
        return cfg


class MetricModel(tf.keras.Model):
    """GRU encoder + cosine head + center loss on embeddings."""
    def __init__(self, backbone, embed_dim, num_classes, center_weight=0.1, scale=16.0, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone          # maps input -> embedding (B, embed_dim)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.center_weight = center_weight
        self.head = CosineHead(num_classes, scale=scale)
        self.center_loss_tracker = tf.keras.metrics.Mean(name="center_loss")

    def call(self, inputs, training=False):
        emb = self.backbone(inputs, training=training)
        logits = self.head(emb)
        return logits

    def get_embedding(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            emb = self.backbone(x, training=True)
            logits = self.head(emb)
            ce = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, tf.nn.softmax(logits), from_logits=False
                )
            )
            # Vectorized center loss: one-hot gather of batch class means.
            y_i = tf.cast(y, tf.int32)
            y_oh = tf.one_hot(y_i, self.num_classes)  # (B, C)
            # class sums -> means (avoid div0 with counts+1 safety)
            emb_exp = tf.expand_dims(emb, 1)          # (B, 1, D)
            y_oh_e = tf.expand_dims(y_oh, 2)          # (B, C, 1)
            summed = tf.reduce_sum(emb_exp * y_oh_e, axis=0)  # (C, D)
            counts = tf.reduce_sum(y_oh, axis=0) + 1e-6      # (C,)
            class_means = summed / counts[:, None]           # (C, D)
            target = tf.gather(class_means, y_i)             # (B, D)
            center = tf.reduce_mean(tf.square(emb - target))
            loss = ce + self.center_weight * center
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.center_loss_tracker.update_state(center)
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64)), tf.float32)
        )
        return {"loss": loss, "accuracy": acc, "center_loss": self.center_loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        emb = self.backbone(x, training=False)
        logits = self.head(emb)
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y, tf.nn.softmax(logits), from_logits=False
            )
        )
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64)), tf.float32)
        )
        return {"loss": ce, "accuracy": acc}


def build_backbone(seq_len, num_features, hidden, dropout, aug_kwargs, l2=1e-4):
    """Returns a model: input -> embedding (EMBED_DIM)."""
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    inputs = tf.keras.Input(shape=(seq_len, num_features))
    x = SequenceAugment(**aug_kwargs, name="seq_augment")(inputs)
    x = tf.keras.layers.Masking(mask_value=0.0)(x)
    x = tf.keras.layers.GRU(
        hidden, return_sequences=False,
        kernel_regularizer=reg, recurrent_regularizer=reg,
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    emb = tf.keras.layers.Dense(
        EMBED_DIM, activation=None, kernel_regularizer=reg, name="embedding"
    )(x)
    return tf.keras.Model(inputs, emb, name="backbone")


def build_metric_model(seq_len, num_features, hidden, dropout, aug_kwargs,
                       center_weight=0.1, scale=16.0, l2=1e-4):
    backbone = build_backbone(seq_len, num_features, hidden, dropout, aug_kwargs, l2)
    model = MetricModel(
        backbone, EMBED_DIM, len(CLASSES),
        center_weight=center_weight, scale=scale,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return model


def predict_probs(model, X):
    """Softmax from cosine logits."""
    logits = model(X, training=False).numpy()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_prototype(model, X, X_ref, y_ref):
    """Classify by mean embedding distance (prototypical) — often more
    subject-robust than the linear cosine head alone."""
    emb_q = model.get_embedding(X, training=False).numpy()
    emb_r = model.get_embedding(X_ref, training=False).numpy()
    # L2 normalize
    emb_q = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-8)
    emb_r = emb_r / (np.linalg.norm(emb_r, axis=1, keepdims=True) + 1e-8)
    protos = []
    for c in range(len(CLASSES)):
        m = y_ref == c
        p = emb_r[m].mean(0)
        protos.append(p / (np.linalg.norm(p) + 1e-8))
    protos = np.stack(protos)  # (C, D)
    # cosine similarity to prototypes
    sim = emb_q @ protos.T
    return sim.argmax(1), sim


def run_cv(args):
    X, y, clip_ids, groups = load_dataset(args.keypoints_dir, args.seq_len)
    aug_kwargs = dict(
        angle_noise=args.angle_noise, feature_noise=args.feature_noise,
        rate_min=args.rate_min, rate_max=args.rate_max,
        max_shift=args.max_shift, frame_drop=args.frame_drop,
    )
    if args.cv == "group":
        if args.n_folds > len(set(groups)):
            raise SystemExit(f"n_folds > n_groups ({len(set(groups))})")
        splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)

    accs_softmax, accs_proto = [], []
    for fold, (tr, va) in enumerate(split_iter):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")
        print(f"  train groups: {sorted(set(groups[tr]))}")
        print(f"  val groups:   {sorted(set(groups[va]))}")
        model = build_metric_model(
            args.seq_len, X.shape[-1], args.hidden_units, args.dropout,
            aug_kwargs, center_weight=args.center_weight, scale=args.scale,
        )
        early = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=args.patience, restore_best_weights=True
        )
        model.fit(
            X[tr], y[tr],
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=[early], verbose=2,
        )
        # softmax head
        probs = predict_probs(model, X[va])
        pred = probs.argmax(1)
        acc_s = accuracy_score(y[va], pred)
        # prototype head (protos from train embeddings)
        pred_p, _ = predict_prototype(model, X[va], X[tr], y[tr])
        acc_p = accuracy_score(y[va], pred_p)
        accs_softmax.append(acc_s)
        accs_proto.append(acc_p)
        print(f"Fold {fold + 1}: softmax={acc_s:.3f}  prototype={acc_p:.3f}")
        model.save(os.path.join(args.out_dir, f"metric_fold{fold + 1}.keras"))

    print(f"\nMean {args.cv} {args.n_folds}-fold softmax: {np.mean(accs_softmax):.3f} "
          f"folds={[round(a,3) for a in accs_softmax]}")
    print(f"Mean {args.cv} {args.n_folds}-fold prototype: {np.mean(accs_proto):.3f} "
          f"folds={[round(a,3) for a in accs_proto]}")
    return accs_softmax, accs_proto


def eval_holdout_ensemble(args):
    """Train group-CV models (or load), ensemble on Beg3/Beg4."""
    X, y, clip_ids, groups = load_dataset(args.keypoints_dir, args.seq_len)
    aug_kwargs = dict(
        angle_noise=args.angle_noise, feature_noise=args.feature_noise,
        rate_min=args.rate_min, rate_max=args.rate_max,
        max_shift=args.max_shift, frame_drop=args.frame_drop,
    )
    # train n_folds group models
    splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    models = []
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        print(f"\n=== Holdout-model fold {fold + 1} (train only on {sorted(set(groups[tr]))}) ===")
        m = build_metric_model(
            args.seq_len, X.shape[-1], args.hidden_units, args.dropout,
            aug_kwargs, center_weight=args.center_weight, scale=args.scale,
        )
        early = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=args.patience, restore_best_weights=True
        )
        m.fit(X[tr], y[tr], epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[early], verbose=2)
        models.append((m, tr))  # keep train idx for prototype ref
    # save
    os.makedirs(args.out_dir, exist_ok=True)
    for i, (m, _) in enumerate(models):
        m.save(os.path.join(args.out_dir, f"metric_holdout_fold{i+1}.keras"))

    holds = {
        "Beg3": ("data/testing/keypoints_angles",),
        "Beg4": ("data/testing/keypoints_beg4_angles",),
    }
    results = {}
    for hname, (kpdir,) in holds.items():
        Xh, yh, cid_h, _ = load_dataset(kpdir, args.seq_len)
        # ensemble softmax probs
        probs = np.mean([predict_probs(m, Xh) for m, _ in models], axis=0)
        pred_s = probs.argmax(1)
        acc_s = accuracy_score(yh, pred_s)
        # ensemble prototype: average embeddings then classify, or majority vote prototypes
        emb_all = []
        for m, tr in models:
            eq = m.get_embedding(Xh, training=False).numpy()
            er = m.get_embedding(X[tr], training=False).numpy()
            emb_all.append((eq, er, y[tr]))
        # average normalized embeddings approach
        preds_p = []
        for eq, er, yr in emb_all:
            pp, _ = predict_prototype(
                type("M", (), {"get_embedding": lambda self, x, training=False: None})(),  # placeholder unused
                Xh, X[tr], yr
            ) if False else (None, None)
        # simpler: vote via per-model prototype
        votes = []
        for m, tr in models:
            pp, _ = predict_prototype(m, Xh, X[tr], y[tr])
            votes.append(pp)
        votes = np.stack(votes)  # (M, N)
        # majority vote
        pred_p = np.array([np.bincount(votes[:, i], minlength=len(CLASSES)).argmax()
                           for i in range(votes.shape[1])])
        acc_p = accuracy_score(yh, pred_p)
        print(f"\n{hname}: softmax ensemble={acc_s:.3f}  prototype vote={acc_p:.3f}  n={len(yh)}")
        print(classification_report(yh, pred_s, labels=list(range(len(CLASSES))),
                                    target_names=CLASSES, zero_division=0, digits=3))
        results[hname] = {"softmax": acc_s, "prototype": acc_p}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoints_dir", default="data/training/keypoints_angles")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--cv", choices=["group", "stratified"], default="group")
    p.add_argument("--hidden_units", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--center_weight", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=16.0)
    p.add_argument("--angle_noise", type=float, default=0.15)
    p.add_argument("--feature_noise", type=float, default=0.03)
    p.add_argument("--rate_min", type=float, default=0.75)
    p.add_argument("--rate_max", type=float, default=1.35)
    p.add_argument("--max_shift", type=int, default=8)
    p.add_argument("--frame_drop", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--out_dir", default="gru_metric_runs")
    p.add_argument("--mode", choices=["cv", "holdout", "both"], default="both")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode in ("cv", "both"):
        print("======== CV ========")
        run_cv(args)
    if args.mode in ("holdout", "both"):
        print("======== HOLDOUT ========")
        eval_holdout_ensemble(args)


if __name__ == "__main__":
    main()
