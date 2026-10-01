"""
export_knn_assets.py

Exports training timing features + labels to a compact binary for on-device kNN5.

Binary format (little-endian):
  int32 n_samples
  int32 n_features
  float32 X[n_samples, n_features]   (row-major)
  int32   y[n_samples]               (0=drive, 1=lob, 2=topspin)

Usage (from project root):
  python scripts/export_knn_assets.py
"""

import json
import os
import struct

import numpy as np

CLASSES = ["drive", "lob", "topspin"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def main():
    src = "features/timing_train.npz"
    out = "android/app/src/main/assets/knn_train.bin"

    d = np.load(src, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y_str = d["y"].astype(str)

    y = np.array([CLASS_TO_IDX[s] for s in y_str], dtype=np.int32)
    assert X.shape[0] == y.shape[0]
    n, f = X.shape

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(struct.pack("<ii", n, f))
        fh.write(X.astype("<f4").tobytes(order="C"))
        fh.write(y.astype("<i4").tobytes(order="C"))

    size = os.path.getsize(out)
    print(f"Wrote {out}")
    print(f"  n={n} features={f}  size={size/1024:.1f} KB")
    print(f"  classes={ {c: int((y_str == c).sum()) for c in CLASSES} }")

    # Sidecar meta for Kotlin
    meta = {
        "n_samples": int(n),
        "n_features": int(f),
        "classes": CLASSES,
        "metric": "euclidean",
        "weights": "distance",
        "k": 5,
        "scaling": "none",
        "note": "kNN5 raw (no scaling) — best holdout mean (56.1%) in docs/RESULTS_SUMMARY.md",
    }
    meta_path = "android/app/src/main/assets/knn_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
