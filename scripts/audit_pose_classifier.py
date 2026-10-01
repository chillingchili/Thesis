"""Reproduce pose/GRU audit without fitting models or selecting hyperparameters.

Run from the repository root:
    python scripts/audit_pose_classifier.py cached
    python scripts/audit_pose_classifier.py fresh --per-stratum 2

Metrics use manifest serve labels, not manual landmark ground truth. Fresh
sampling is deterministic, stratified by player and class, plus named failures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from extract_joint_angles import clip_angles
from normalize_keypoints import normalize_clip
from trim_after_contact import estimate_contact_frame

OUT = ROOT / "outputs/pose_audit"
ASSETS = ROOT / "android/app/src/main/assets"
CLASSES = ["drive", "lob", "topspin"]
SPECS = [
    ("train", "data/training", "keypoints", "keypoints_angles", "keypoints_trimmed"),
    ("test_beg3", "data/testing", "keypoints", "keypoints_angles", "keypoints_trimmed"),
    ("test_beg4", "data/testing", "keypoints_beg4", "keypoints_beg4_angles", "keypoints_beg4_trimmed"),
]


def write_json(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def inventory():
    rows, missing = [], []
    for split, base, raw_dir, angle_dir, trimmed_dir in SPECS:
        with (ROOT / base / angle_dir / "manifest.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                cid = Path(row["clip_path"].replace("\\", "/")).stem
                item = dict(split=split, clip_id=cid, label=row["serve_type"].strip().lower(),
                            subject="Beginner4" if split == "test_beg4" else cid.split("_")[0],
                            video_manifest=row["clip_path"],
                            raw=str(ROOT / base / raw_dir / (cid + ".npy")),
                            angles=str(ROOT / base / angle_dir / (cid + ".npy")),
                            trimmed=str(ROOT / base / trimmed_dir / (cid + ".npy")))
                (rows if all(Path(item[k]).exists() for k in ["raw", "angles", "trimmed"]) else missing).append(item)
    return rows, missing


def padded(angles):
    flat = angles.reshape(len(angles), 10)
    result = np.zeros((128, 10), np.float32)
    n = min(len(flat), 128)
    result[:n] = np.nan_to_num(flat[:n], nan=0, posinf=0, neginf=0)
    return result


def training_window(points):
    contact = estimate_contact_frame(points)
    return padded(clip_angles(normalize_clip(points[:min(contact + 10, len(points))])))


def quality(points):
    valid = np.isfinite(points).all(axis=(1, 2))
    angles = clip_angles(points)
    rad = np.arctan2(angles[..., 0], angles[..., 1])
    change = np.abs(np.angle(np.exp(1j * np.diff(rad, axis=0)))) * 180 / np.pi
    pairs = valid[1:] & valid[:-1]
    contact = estimate_contact_frame(points)
    return dict(frames=len(points), detected_frames=int(valid.sum()),
                detected_fraction=float(valid.mean()), contact=contact,
                contact_outside_window=bool(contact >= 128),
                valid_first128=int(valid[:128].sum()),
                wrist_jump_gt60=int(np.sum(change[:, 2][pairs] > 60)),
                adjacent_detected_pairs=int(pairs.sum()))


def metrics(y, probs):
    from sklearn.metrics import confusion_matrix, f1_score
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = pred == y
    accepted = confidence >= .6
    return dict(n=len(y), correct=int(correct.sum()), accuracy=float(correct.mean()),
                macro_f1=float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                confusion=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
                mean_confidence=float(confidence.mean()), accepted=int(accepted.sum()),
                coverage=float(accepted.mean()),
                accepted_accuracy=float(correct[accepted].mean()) if accepted.any() else None,
                accepted_wrong=int((accepted & ~correct).sum()),
                accuracy_abstentions_wrong=float((accepted & correct).mean()))


def aggregate(rows, probs):
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    groups = {"all": np.ones(len(rows), dtype=bool),
              "test_combined": np.array([r["split"] != "train" for r in rows])}
    for key in ["split", "subject"]:
        for value in sorted({r[key] for r in rows}):
            groups[value] = np.array([r[key] == value for r in rows])
    return {name: metrics(y[mask], probs[mask]) for name, mask in groups.items() if mask.any()}


def models():
    import tensorflow as tf
    from train_gru import SequenceAugment
    return [tf.keras.models.load_model(ROOT / f"models/gru_runs_angles_strat5/gru_fold{i}.keras",
                                      custom_objects={"SequenceAugment": SequenceAugment}, compile=False)
            for i in range(1, 6)]


def predict(model, x):
    # Explicit inference mode disables SequenceAugment and dropout.
    return np.concatenate([model(x[i:i+64], training=False).numpy() for i in range(0, len(x), 64)])


def cached():
    from desktop.pipeline import build_window
    from sklearn.model_selection import StratifiedKFold
    rows, missing = inventory()
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    batches = {name: [] for name in ["saved_training_features", "untrimmed_no_matching", "untrimmed_matching", "trimmed_matching"]}
    qualities = []
    hashes = {}
    trim_mismatches, angle_mismatches = [], []
    angle_maxdiff = 0.0
    for row in rows:
        raw, saved, trimmed = (np.load(row[k]) for k in ["raw", "angles", "trimmed"])
        q = quality(raw)
        q.update({k: row[k] for k in ["split", "subject", "clip_id", "label"]})
        q["saved_angle_frames"] = len(saved)
        qualities.append(q)
        expected_trimmed = raw[:min(q["contact"] + 10, len(raw))]
        if expected_trimmed.shape != trimmed.shape or not np.array_equal(expected_trimmed, trimmed, equal_nan=True):
            trim_mismatches.append(row["clip_id"])
        regenerated = clip_angles(normalize_clip(trimmed))
        if regenerated.shape != saved.shape or not np.allclose(regenerated, saved, atol=1e-6, equal_nan=True):
            angle_mismatches.append(row["clip_id"])
        if regenerated.shape == saved.shape:
            diff = np.abs(regenerated - saved)
            if np.isfinite(diff).any():
                angle_maxdiff = max(angle_maxdiff, float(np.nanmax(diff)))
        batches["saved_training_features"].append(padded(saved))
        batches["untrimmed_no_matching"].append(build_window(raw, manifest, False)[0][0])
        batches["untrimmed_matching"].append(build_window(raw, manifest, True)[0][0])
        batches["trimmed_matching"].append(build_window(trimmed, manifest, True)[0][0])
        digest = hashlib.sha256(Path(row["raw"]).read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(dict(split=row["split"], clip_id=row["clip_id"]))
    write_csv("cached_pose_quality.csv", qualities)
    write_json("inventory.json", dict(usable=len(rows), missing=missing, trim_mismatches=trim_mismatches,
               angle_mismatches=angle_mismatches, angle_maxdiff=angle_maxdiff,
               duplicate_pose_files=[v for v in hashes.values() if len(v) > 1]))
    batches = {k: np.stack(v) for k, v in batches.items()}
    modes = list(batches)
    x = np.concatenate(list(batches.values()))
    fold_probs = []
    for i, model in enumerate(models()):
        print(f"Cached prediction fold {i+1}/5 ({len(x)} windows)", flush=True)
        fold_probs.append(predict(model, x))
    fold_probs = np.array(fold_probs)
    means = fold_probs.mean(axis=0)
    probs = {mode: means[i*len(rows):(i+1)*len(rows)] for i, mode in enumerate(modes)}
    results = {mode: aggregate(rows, p) for mode, p in probs.items()}
    train_n = sum(r["split"] == "train" for r in rows)
    y_train = np.array([CLASSES.index(r["label"]) for r in rows[:train_n]])
    oof = np.zeros((train_n, 3), np.float32)
    fold_scores = []
    for i, (_, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(train_n), y_train)):
        oof[val] = fold_probs[i, val]
        fold_scores.append(metrics(y_train[val], oof[val]))
    results["reconstructed_oof"] = aggregate(rows[:train_n], oof)
    results["oof_folds"] = fold_scores
    # Verify the deployed TFLite conversion on actual windows, covering every split/player/class.
    import tensorflow as tf
    sample = select_sample(rows, 2)
    indices = [rows.index(r) for r in sample]
    x_parity = np.concatenate([batches[k][indices] for k in modes])
    keras_indices = np.concatenate([np.array(indices) + i*len(rows) for i in range(len(modes))])
    parity = []
    for i in range(5):
        path = ASSETS / f"gru_fold{i+1}.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0]["index"], interpreter.get_output_details()[0]["index"]
        converted = []
        for window in x_parity:
            interpreter.set_tensor(inp, window[None].astype(np.float32))
            interpreter.invoke()
            converted.append(interpreter.get_tensor(out)[0])
        converted = np.array(converted)
        keras = fold_probs[i, keras_indices]
        parity.append(dict(fold=i+1, n=len(converted), max_probability_diff=float(np.abs(converted-keras).max()),
                           label_disagreements=int(np.sum(converted.argmax(1) != keras.argmax(1))),
                           asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           exported_file_identical=path.read_bytes() == (ROOT / "models/tflite" / path.name).read_bytes()))
    results["tflite_parity"] = parity
    write_json("cached_metrics.json", results)
    records = []
    for j, row in enumerate(rows):
        for mode, p in probs.items():
            records.append(dict(split=row["split"], subject=row["subject"], clip_id=row["clip_id"],
                                truth=row["label"], mode=mode, predicted=CLASSES[int(p[j].argmax())],
                                confidence=float(p[j].max()), **dict(zip(CLASSES, map(float, p[j])))))
        if j < train_n:
            records.append(dict(split=row["split"], subject=row["subject"], clip_id=row["clip_id"],
                                truth=row["label"], mode="reconstructed_oof", predicted=CLASSES[int(oof[j].argmax())],
                                confidence=float(oof[j].max()), **dict(zip(CLASSES, map(float, oof[j])))))
    write_csv("cached_predictions.csv", records)
    for mode in probs:
        print(mode, {g: round(results[mode][g]["accuracy"], 4) for g in ["train", "test_beg3", "test_beg4", "test_combined"]}, flush=True)
    print("OOF", results["reconstructed_oof"]["train"], flush=True)


def select_sample(rows, per_stratum):
    rng = np.random.default_rng(42)
    result = []
    for subject in sorted({r["subject"] for r in rows}):
        for label in CLASSES:
            group = [r for r in rows if r["subject"] == subject and r["label"] == label]
            result.extend(group[i] for i in sorted(rng.choice(len(group), min(per_stratum, len(group)), replace=False)))
    for cid in ["Beginner2_Drive_001", "Beginner2_Drive_002", "Beginner2_Drive_007"]:
        item = next(r for r in rows if r["clip_id"] == cid)
        if item not in result:
            result.append(item)
    return result


def fresh_extract(path, variant):
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode {path}")
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), advertised_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if variant == "heavy":
        detector = mp.solutions.pose.Pose(model_complexity=2, min_detection_confidence=.5, min_tracking_confidence=.6)
    else:
        detector = mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
            # Legacy Solutions changes MediaPipe's process-global resource root.
            # A buffer avoids its Windows path interaction with the Tasks API.
            base_options=mp.tasks.BaseOptions(model_asset_buffer=(ASSETS / "pose_landmarker_lite.task").read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=.5, min_pose_presence_confidence=.5, min_tracking_confidence=.5))
    points, visibility = [], []
    start = time.perf_counter()
    try:
        with detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if variant == "heavy":
                    result = detector.process(rgb)
                    landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
                else:
                    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                       round(len(points)*1000/info["fps"]))
                    landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                points.append([[p.x, p.y] for p in landmarks] if landmarks else np.full((33, 2), np.nan))
                visibility.append([p.visibility for p in landmarks] if landmarks else np.zeros(33))
    finally:
        cap.release()
    info["seconds"] = time.perf_counter() - start
    if len(points) < 3:
        raise RuntimeError(f"Too few frames in {path}")
    return np.asarray(points, np.float32), np.asarray(visibility, np.float32), info


def fresh(per_stratum):
    from desktop.pipeline import build_window
    rows, _ = inventory()
    sample = select_sample(rows, per_stratum)
    videos = {}
    for base in [ROOT / "data/training/raw", ROOT / "data/testing/Beginner3", ROOT / "data/testing/Beginner4"]:
        for p in base.rglob("*.mp4"):
            if p.stem in videos:
                raise RuntimeError(f"Ambiguous video stem {p.stem}")
            videos[p.stem] = p
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    write_json("fresh_sample.json", sample)
    windows, meta, records = [], [], []
    repeated = []
    for j, row in enumerate(sample):
        cid = row["clip_id"]
        path = videos[cid]
        all_points = {"cached_heavy": np.load(row["raw"])}
        for variant in ["lite", "heavy"]:
            target = OUT / "fresh" / variant / (cid + ".npz")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                data = np.load(target)
                points, visibility = data["points"], data["visibility"]
                info = json.loads(str(data["info"]))
            else:
                print(f"Fresh {j+1}/{len(sample)} {variant} {cid}", flush=True)
                points, visibility, info = fresh_extract(path, variant)
                np.savez_compressed(target, points=points, visibility=visibility, info=json.dumps(info))
            all_points[variant] = points
            q = quality(points)
            arm_visible = np.min(visibility[:, [12, 14, 16, 20]], axis=1) >= .5
            q.update(split=row["split"], subject=row["subject"], clip_id=cid, variant=variant,
                     label=row["label"], arm_visibility_ge05_fraction=float(arm_visible.mean()), **info)
            records.append(q)
        for variant, points in all_points.items():
            for mode in ["training_preprocessing", "desktop_default", "desktop_no_matching"]:
                window = (training_window(points) if mode == "training_preprocessing" else
                          build_window(points, manifest, mode == "desktop_default")[0][0])
                windows.append(window)
                meta.append(dict(**{k: row[k] for k in ["split", "subject", "clip_id", "label"]},
                                 variant=variant, mode=mode))
        # Determinism check on the two clips specifically raised by the user.
        if cid in ["Beginner2_Drive_001", "Beginner2_Drive_002"]:
            for variant in ["lite", "heavy"]:
                again, _, _ = fresh_extract(path, variant)
                first = all_points[variant]
                equal_shape = again.shape == first.shape
                difference = np.abs(first - again) if equal_shape else np.array([np.nan])
                repeated.append(dict(clip_id=cid, variant=variant, same_shape=equal_shape,
                                     same_missing_mask=bool(equal_shape and np.array_equal(np.isnan(first), np.isnan(again))),
                                     max_coordinate_diff=float(np.nanmax(difference)) if np.isfinite(difference).any() else None))
        # Compare corresponding frames: disagreement is NOT landmark accuracy.
        for a, b in [("cached_heavy", "heavy"), ("heavy", "lite")]:
            pa, pb = all_points[a], all_points[b]
            n = min(len(pa), len(pb))
            aa, ab = clip_angles(pa[:n]), clip_angles(pb[:n])
            ra, rb = np.arctan2(aa[..., 0], aa[..., 1]), np.arctan2(ab[..., 0], ab[..., 1])
            diff = np.abs(np.angle(np.exp(1j*(ra-rb)))) * 180/np.pi
            comparison = dict(clip_id=cid, pair=f"{a}_vs_{b}", compared_frames=n,
                              same_length=len(pa) == len(pb),
                              median_angle_disagreement_deg=[float(np.nanmedian(diff[:, k])) for k in range(5)])
            repeated.append(comparison)
        write_csv("fresh_pose_quality.csv", records)
        write_json("fresh_comparisons.json", repeated)
    x = np.stack(windows)
    probabilities = []
    for i, model in enumerate(models()):
        print(f"Fresh classification fold {i+1}/5", flush=True)
        probabilities.append(predict(model, x))
    probs = np.mean(probabilities, axis=0)
    results = {}
    for variant in ["cached_heavy", "heavy", "lite"]:
        for mode in ["training_preprocessing", "desktop_default", "desktop_no_matching"]:
            ix = [i for i, r in enumerate(meta) if r["variant"] == variant and r["mode"] == mode]
            results[f"{variant}/{mode}"] = aggregate([meta[i] for i in ix], probs[ix])
    write_json("fresh_metrics.json", results)
    predictions = []
    for row, p in zip(meta, probs):
        predictions.append(dict(**row, predicted=CLASSES[int(p.argmax())], confidence=float(p.max()),
                                **dict(zip(CLASSES, map(float, p)))))
    write_csv("fresh_predictions.csv", predictions)
    print("Fresh audit complete", flush=True)


def order_check():
    """Isolate clip-order effects in the original shared Heavy tracker."""
    import mediapipe as mp
    from desktop.pipeline import build_window
    from extract_keypoints import extract_clip_keypoints
    videos = {p.stem: p for p in (ROOT / "data/training/raw").rglob("*.mp4")}
    first, second = (videos[f"Beginner2_Drive_{i:03}"] for i in [1, 2])
    with mp.solutions.pose.Pose(model_complexity=2, min_detection_confidence=.5,
                                min_tracking_confidence=.6) as pose:
        isolated = extract_clip_keypoints(str(second), pose)
        pose.reset()
        extract_clip_keypoints(str(first), pose)
        carried = extract_clip_keypoints(str(second), pose)
        pose.reset()
        reset = extract_clip_keypoints(str(second), pose)
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    data = {"isolated": isolated, "after_previous_clip": carried, "after_reset": reset}
    windows, labels = [], []
    for name, points in data.items():
        for mode in ["training_preprocessing", "desktop_default"]:
            windows.append(training_window(points) if mode == "training_preprocessing" else
                           build_window(points, manifest, True)[0][0])
            labels.append((name, mode))
    x = np.stack(windows)
    probs = np.mean([predict(m, x) for m in models()], axis=0)
    results = dict(clip_id=second.stem, preceding_clip=first.stem,
                  max_coordinate_diff_carried=float(np.nanmax(np.abs(isolated-carried))),
                  max_coordinate_diff_after_reset=float(np.nanmax(np.abs(isolated-reset))),
                  predictions=[dict(order=name, mode=mode, predicted=CLASSES[int(p.argmax())],
                                    confidence=float(p.max()), probabilities=p.tolist())
                               for (name, mode), p in zip(labels, probs)])
    write_json("order_check.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cached", "fresh", "order"])
    parser.add_argument("--per-stratum", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "cached":
        cached()
    elif args.stage == "fresh":
        fresh(args.per_stratum)
    else:
        order_check()
