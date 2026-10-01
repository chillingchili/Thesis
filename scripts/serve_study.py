"""Frozen protocol and shared features for the four follow-up investigations.

Training entry points only open the Coach B training partition. Original v2
artifacts and Android assets are read-only inputs throughout this study.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from serve_sequence import ROOT, fingerprint

DATA = ROOT / "data/serve_study_v3"
OUT = ROOT / "models/serve_study_v3"
CLASSES = ["drive", "lob", "topspin"]
SEED = 20261002
PROTOCOL = {
    "version": "serve_study_v3", "seed": SEED,
    "training_partition": "209 quality-eligible Coach B clips from three players",
    "primary_protocol": "leave_one_training_player_out",
    "secondary_protocol": "stratified_five_fold_development_CV",
    "conditions": ["angles_control", "angles_thesis_aug", "skeleton", "skeleton_motion"],
    "epochs": 100, "batch_size": 16, "gru_units": 32, "dropout": .15, "l2": .0001,
    "optimizer": "Adam learning_rate=0.001", "early_stopping": False,
    "training_budget": "one example per original training clip per epoch in every condition",
    "augmentation_sampling": "uniform original or either of two frozen composite variants; rejected variants fall back to original",
    "augmentation": {
        "variants_per_clip": 2, "contrast_range": [.9, 1.1],
        "brightness_fraction_range": [-.08, .08], "gaussian_sigma_fraction_range": [.03, .05],
        "crop_scale_range": [.94, .99], "crop_translation": "uniform within the cropped margin",
        "aspect_ratio": "preserved with isotropic affine transform",
        "temporal_sample_spacing_range": [1.02, 1.1],
        "temporal": "nearest decoded source frames, unique indices, retain endpoints, uniform output playback timestamps",
        "parameters_constant_per_clip": True, "noise_independent_per_frame": True,
        "flip": False, "shear": False,
        "pose": "fresh Lite VIDEO tracker per variant after all pixel transformations",
        "note": "raw videos are untouched; cached derived landmarks only; no feature-noise augmentation",
    },
    "features": {
        "angles": "existing full-serve 128 x 10 sin/cos representation",
        "skeleton": "33 aspect-corrected xy joints, centered at hips, median visible torso scale; 33 visibility masks (99 channels)",
        "skeleton_motion": "skeleton plus 66 adjacent-step displacements; invalid endpoints masked (165 channels)",
        "interpolation": "128 full-timeline samples; per-joint adjacent-valid interpolation only",
        "note": "richer inputs also increase GRU parameter count; this is an input-system comparison, not capacity-matched attribution",
    },
    "hybrid": "fixed 0.5 GRU + 0.5 raw Euclidean distance-weighted kNN5; fit kNN only on each training fold",
    "selection": "highest participant-CV mean subject accuracy among four conditions; ties follow condition order",
    "test_policy": "only after all candidates and selection are frozen; existing 175 clips are diagnostic, not an untouched final test",
    "coach_audit": "training clips only; blinded subtype review followed by skeleton-quality review; no automatic relabeling",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def freeze_protocol():
    p = dict(PROTOCOL, dataset_sha256=sha(ROOT / "data/serve_v2/dataset.json"),
             train_npz_sha256=sha(ROOT / "data/serve_v2/train.npz"),
             pose_contract_fingerprint=fingerprint())
    target = DATA / "protocol.json"
    if target.exists() and json.loads(target.read_text()) != p:
        raise ValueError("Frozen study protocol changed: create a new version")
    if not target.exists():
        write_json(target, p)
    return p


def training_data():
    freeze_protocol()
    d = np.load(ROOT / "data/serve_v2/train.npz", allow_pickle=False)
    keep = d["quality_accepted"]
    result = {k: d[k][keep] for k in d.files}
    assert len(result["y"]) == 209
    assert set(result["subjects"]) == {"CoachA", "Beginner1", "Beginner2"}
    return result


def training_rows():
    ids = set(training_data()["clip_ids"])
    frozen = json.loads((ROOT / "data/serve_v2/dataset.json").read_text())
    return [r for r in frozen["clips"] if r["split"] == "train" and r["clip_id"] in ids]


def metrics(y, p, subjects=None, available=None):
    y, p = np.asarray(y), np.asarray(p)
    available = np.ones(len(y), bool) if available is None else np.asarray(available, bool)
    pred = p.argmax(1)
    pred[~available] = -1
    ok = pred == y
    confident = available & (p.max(1) >= .6)
    result = dict(n=len(y), correct=int(ok.sum()), accuracy=float(ok.mean()),
                  macro_f1=float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                  confusion_with_unavailable=confusion_matrix(y, pred, labels=[0, 1, 2, -1]).tolist(),
                  available=int(available.sum()), confidence_coverage=float(confident.mean()),
                  confident_wrong=int(np.sum(confident & ~ok)),
                  confident_accuracy=float(ok[confident].mean()) if confident.any() else None)
    if subjects is not None:
        subjects = np.asarray(subjects)
        result["per_subject"] = {str(s): metrics(y[subjects == s], p[subjects == s], available=available[subjects == s])
                                 for s in sorted(set(subjects))}
        result["mean_subject_accuracy"] = float(np.mean([m["accuracy"] for m in result["per_subject"].values()]))
    return result


def skeleton_features(points, visibility, timestamps, aspect):
    """No fitting/statistics across clips; no interpolation through hidden joints."""
    p = np.asarray(points, np.float64).copy()
    v = (np.asarray(visibility) >= .5) & np.isfinite(p).all(2)
    t = np.asarray(timestamps, np.float64)
    if len(t) < 2 or not np.all(np.diff(t) > 0) or not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("Invalid timeline/aspect")
    p[:, :, 0] *= aspect
    anchor = v[:, [11, 12, 23, 24]].all(1)
    hip = (p[:, 23] + p[:, 24]) / 2
    shoulder = (p[:, 11] + p[:, 12]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    good = anchor & (torso > 1e-6)
    if not good.any():
        return np.zeros((128, 99), np.float32), np.zeros((128, 165), np.float32)
    scale = np.median(torso[good])
    p = (p - hip[:, None]) / scale
    v &= anchor[:, None]
    p[~v] = 0
    target = np.linspace(t[0], t[-1], 128)
    right = np.searchsorted(t, target).clip(0, len(t) - 1)
    left = np.maximum(right - 1, 0)
    left[np.isclose(target, t[right], rtol=0, atol=1e-9)] = right[np.isclose(target, t[right], rtol=0, atol=1e-9)]
    alpha = np.divide(target - t[left], t[right] - t[left], out=np.zeros(128), where=t[right] > t[left])
    valid = v[left] & v[right]
    xy = p[left] * (1 - alpha[:, None, None]) + p[right] * alpha[:, None, None]
    xy[~valid] = 0
    delta = np.zeros_like(xy)
    delta[1:] = xy[1:] - xy[:-1]
    delta[1:][~(valid[1:] & valid[:-1])] = 0
    base = np.concatenate([xy.reshape(128, 66), valid.astype(float)], axis=1).astype(np.float32)
    motion = np.concatenate([base, delta.reshape(128, 66)], axis=1).astype(np.float32)
    assert np.isfinite(motion).all()
    return base, motion
