"""
export_tflite.py

Exports GRU .keras checkpoints to TFLite for on-device (Android) inference.

- Strips SequenceAugment (training-only; identity at inference).
- Keeps Masking chain (GRU needs pad mask for short sequences).
- Exports all 5 strat5 folds for ensemble + a copy as "single" (fold1 best CV).

Usage (from project root):
  python scripts/export_tflite.py
  python scripts/export_tflite.py --src models/gru_runs_angles_strat5 --out models/tflite
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

# Make scripts/ importable when run as file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gru import SequenceAugment  # noqa: F401  (register custom layer)

CLASSES = ["drive", "lob", "topspin"]


def _rebuild_without_augment(m: tf.keras.Model) -> tf.keras.Model:
    """Clone m, skipping SequenceAugment. Match layers by type (names can get _1/_2 suffixes)."""
    import keras

    inputs = tf.keras.Input(shape=(128, 10), dtype=tf.float32, name="serving_input")

    def find(cls):
        for l in m.layers:
            if isinstance(l, cls) and not isinstance(l, type(None)):
                # exclude SequenceAugment if cls is Layer base
                if cls is keras.layers.Layer:
                    if type(l).__name__ == "SequenceAugment" or l.name == "seq_augment":
                        continue
                return l
        return None

    masking = find(keras.layers.Masking)
    not_equal = None
    any_layer = None
    # NotEqual / Any may be keras.ops layers or Lambda - search by class name
    for l in m.layers:
        n = type(l).__name__
        if "NotEqual" in n or n == "NotEqual":
            not_equal = l
        if n in ("Any",) or "Any" == n:
            any_layer = l
    gru = find(keras.layers.GRU)
    dropout = find(keras.layers.Dropout)
    dense = find(keras.layers.Dense)

    if masking is None or gru is None or dense is None:
        raise RuntimeError(f"Missing core layers: masking={masking}, gru={gru}, dense={dense}")

    m_out = masking(inputs)
    if not_equal is not None and any_layer is not None:
        ne = not_equal(inputs)
        any_m = any_layer(ne)
        g = gru([m_out, any_m])
    else:
        g = gru(m_out)
    d = dropout(g) if dropout is not None else g
    out = dense(d)
    return tf.keras.Model(inputs, out, name="gru_serving")


def convert_to_tflite(model: tf.keras.Model, out_path: str, select_ops: bool = False) -> int:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if select_ops:
        # GRU + Masking emits TensorListReserve with dynamic shape —
        # needs Select TF ops on Android (adds ~1.5MB dependency).
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
    # No quantization — keep float32 for parity with Keras; model is tiny (~50KB)
    tflite_bytes = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    return len(tflite_bytes)


def verify_tflite(tflite_path: str, keras_model: tf.keras.Model, n_trials: int = 5) -> bool:
    """Run TFLite interpreter vs Keras on random inputs; max abs diff < 1e-4."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    max_diff = 0.0
    for _ in range(n_trials):
        x = np.random.randn(1, 128, 10).astype(np.float32)
        # half zeros to test masking path
        if _ % 2 == 0:
            cut = np.random.randint(20, 128)
            x[0, cut:] = 0.0

        keras_out = keras_model.predict(x, verbose=0)

        interp.set_tensor(inp["index"], x)
        interp.invoke()
        tfl_out = interp.get_tensor(out["index"])

        d = float(np.max(np.abs(keras_out - tfl_out)))
        max_diff = max(max_diff, d)

    ok = max_diff < 1e-4
    print(f"  verify max_abs_diff={max_diff:.6f}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="models/gru_runs_angles_strat5")
    p.add_argument("--out", default="models/tflite")
    p.add_argument("--verify", action="store_true", default=True)
    p.add_argument("--no-verify", dest="verify", action="store_false")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    keras_files = sorted(
        f for f in os.listdir(args.src) if f.endswith(".keras")
    )
    if not keras_files:
        raise SystemExit(f"No .keras files in {args.src}")

    print(f"Found {len(keras_files)} checkpoints in {args.src}")

    # Export each fold as tflite
    tflite_paths = []
    select_ops = False  # flip to True if default convert fails on TensorListReserve
    for kf in keras_files:
        fold_stem = os.path.splitext(kf)[0]  # e.g. gru_fold1
        keras_path = os.path.join(args.src, kf)
        tflite_path = os.path.join(args.out, fold_stem + ".tflite")

        print(f"\n[{kf}] building inference model (strip SequenceAugment)...")
        infer = _rebuild_without_augment(
            tf.keras.models.load_model(
                keras_path, custom_objects={"SequenceAugment": SequenceAugment}, compile=False
            )
        )
        try:
            size = convert_to_tflite(infer, tflite_path, select_ops=False)
        except Exception as e:
            print(f"  default convert failed ({str(e)[:120]}...), retrying with SELECT_TF_OPS")
            select_ops = True
            size = convert_to_tflite(infer, tflite_path, select_ops=True)
        print(f"  -> {tflite_path}  ({size/1024:.1f} KB)")

        if args.verify:
            orig = tf.keras.models.load_model(
                keras_path, custom_objects={"SequenceAugment": SequenceAugment}, compile=False
            )
            # Compare inference-stripped keras vs tflite (both should match; original has same weights)
            # Use `infer` for parity since SequenceAugment is identity at inference anyway
            if not verify_tflite(tflite_path, infer):
                print(f"  WARNING: verification failed for {kf}")

        tflite_paths.append(tflite_path)

    # "single" convenience copy: fold1 (best strat5 CV fold: 90.9%)
    single_src = os.path.join(args.out, "gru_fold1.tflite")
    single_dst = os.path.join(args.out, "gru_single.tflite")
    if os.path.exists(single_src):
        import shutil
        shutil.copy2(single_src, single_dst)
        print(f"\nSingle-model alias: {single_dst} (from fold1)")

    # Manifest for Android
    manifest = {
        "classes": CLASSES,
        "seq_len": 128,
        "n_features": 10,
        "ensemble": [os.path.basename(p) for p in tflite_paths],
        "single": "gru_single.tflite",
        "select_tf_ops": True,
        "note": "GRU TensorListReserve requires SELECT_TF_OPS on Android (org.tensorflow:tensorflow-lite-select-tf-ops).",
    }

    # Moment-matching stats from training joint-angle features
    try:
        import csv as _csv
        kp_dir = "data/training/keypoints_angles"
        man_csv = os.path.join(kp_dir, "manifest.csv")
        if os.path.exists(man_csv):
            rows = list(_csv.DictReader(open(man_csv)))
            chunks = []
            for row in rows:
                clip_id = os.path.splitext(os.path.basename(row["clip_path"]))[0]
                npy = os.path.join(kp_dir, f"{clip_id}.npy")
                if not os.path.exists(npy):
                    continue
                arr = np.load(npy)
                flat = np.nan_to_num(arr.reshape(arr.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
                valid = np.any(flat != 0, axis=1)
                chunks.append(flat[valid])
            if chunks:
                X = np.concatenate(chunks, axis=0)
                mean = X.mean(axis=0)
                std = X.std(axis=0)
                std = np.where(std < 1e-6, 1.0, std)
                manifest["feature_mean"] = [round(float(v), 6) for v in mean]
                manifest["feature_std"] = [round(float(v), 6) for v in std]
                manifest["n_train_frames"] = int(X.shape[0])
                manifest["moment_matching"] = True
                manifest["note_moment"] = (
                    "Per-channel mean/std of 10-dim sin/cos joint-angle features over valid training frames. "
                    "On-device moment-matching TTA: align live window stats to these before GRU inference."
                )
                print(f"Moment-matching stats from {X.shape[0]} train frames")
    except Exception as e:
        print(f"WARN: could not compute moment-matching stats: {e}")

    import json
    man_path = os.path.join(args.out, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {man_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
