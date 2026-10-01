"""Android's bundled distance-weighted kNN5; no fitting or test-set adaptation."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def timing_features(window, valid_frames):
    """32 timing features from UNMATCHED sin/cos features, excluding trailing pad.

    A frame's velocity is its difference from the preceding frame. The first
    velocity is zero, matching the offline features used to build the bank.
    """
    flat = np.asarray(window, np.float32).reshape(-1, 10)
    if not 0 <= valid_frames <= len(flat):
        raise ValueError("Invalid timing-feature length")
    if valid_frames < 4:
        return np.zeros(32, np.float32)
    pairs = flat[:valid_frames].reshape(valid_frames, 5, 2)
    if not np.isfinite(pairs).all():
        raise ValueError("Timing features require finite pose frames")
    angles = np.arctan2(pairs[..., 0], pairs[..., 1])
    out, peaks, troughs = [], [], []
    denom = valid_frames-1
    for k in range(5):
        a = angles[:, k]
        peak, trough = int(a.argmax()), int(a.argmin())
        velocity = np.diff(a, prepend=a[0])
        max_velocity = int(np.abs(velocity).argmax())
        peaks.append(peak/denom); troughs.append(trough/denom)
        # Double accumulation and float32 rounding match the Kotlin calculations.
        mean = np.float32(np.sum(a, dtype=np.float64)/valid_frames)
        delta = a-mean
        std = np.sqrt(np.float32(np.sum(delta*delta, dtype=np.float64)/valid_frames))
        out.extend([peak/denom, trough/denom, max_velocity/denom, float(a.max()-a.min()), mean, std])
    out.extend([peaks[1]-peaks[0], troughs[1]-troughs[0]])
    return np.asarray(out, np.float32)


class Knn5:
    def __init__(self, directory):
        directory = Path(directory)
        self.meta = json.loads((directory/"knn_meta.json").read_text())
        data = (directory/"knn_train.bin").read_bytes()
        n, f = struct.unpack_from("<ii", data)
        self.k = self.meta["k"]
        self.classes = self.meta["classes"]
        if (n,f) != (self.meta["n_samples"],self.meta["n_features"]) or f != 32 or not 1 <= self.k <= n:
            raise ValueError("Invalid kNN bank metadata")
        if len(data) != 8+4*n*f+4*n:
            raise ValueError("Invalid kNN binary length")
        if self.meta["metric"] != "euclidean" or self.meta["weights"] != "distance" or self.meta["scaling"] != "none":
            raise ValueError("Unsupported kNN inference contract")
        self.X = np.frombuffer(data,dtype="<f4",count=n*f,offset=8).reshape(n,f)
        self.y = np.frombuffer(data,dtype="<i4",count=n,offset=8+4*n*f)
        if not np.isfinite(self.X).all() or np.any(self.y < 0) or np.any(self.y >= len(self.classes)):
            raise ValueError("Invalid kNN bank values")
        self.sha256 = hashlib.sha256(data).hexdigest()

    def predict(self, features):
        features = np.asarray(features,np.float32)
        if features.shape != (32,) or not np.isfinite(features).all():
            raise ValueError("Expected 32 finite timing features")
        delta = features-self.X
        # Sequential float accumulation matches Android's squared-distance loop.
        squared = np.zeros(len(self.X),np.float32)
        for j in range(32):
            squared += delta[:,j]*delta[:,j]
        distances = np.sqrt(squared)
        nearest = np.argsort(distances,kind="stable")[:self.k]
        probabilities = np.zeros(len(self.classes),np.float32)
        total = np.float32(0)
        for index in nearest:
            d = distances[index]
            weight = np.float32(1e12) if d < 1e-12 else np.float32(1)/d
            probabilities[self.y[index]] += weight
            total += weight
        return probabilities/total


def combine(gru, knn):
    return (np.asarray(gru,np.float32)+np.asarray(knn,np.float32))*np.float32(.5)
