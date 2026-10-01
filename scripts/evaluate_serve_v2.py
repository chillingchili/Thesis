"""Export frozen training-only GRUs, verify parity, then evaluate diagnostic tests.

This entry point cannot fit models or change their settings. No test-driven
checkpoint/ensemble selection or confidence tuning is performed.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from train_gru import SequenceAugment
from export_tflite import _rebuild_without_augment, convert_to_tflite
from retrain_serve_v2 import metric
from serve_sequence import ROOT, contract, fingerprint
sys.path.insert(0, str(ROOT))
from desktop.pipeline import build_window

OUT = ROOT / "models/serve_v2"
CLASSES = ["drive", "lob", "topspin"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return tf.keras.models.load_model(path, custom_objects={"SequenceAugment": SequenceAugment}, compile=False)


def predict(model, x):
    return np.concatenate([model(x[i:i+64], training=False).numpy() for i in range(0, len(x), 64)])


def evaluated(y, probs, quality):
    result = metric(y, probs)
    correct = probs.argmax(1) == y
    confident = (probs.max(1) >= .6) & quality
    result.update(quality_accepted=int(quality.sum()), quality_rejected=int((~quality).sum()),
                  accuracy_all_clips_quality_failures_wrong=float(np.mean(correct & quality)),
                  quality_accepted_accuracy=float(correct[quality].mean()) if quality.any() else None,
                  pipeline_coverage=float(confident.mean()),
                  pipeline_confident_accuracy=float(correct[confident].mean()) if confident.any() else None,
                  pipeline_confident_wrong=int(np.sum(confident & ~correct)),
                  pipeline_accuracy_abstentions_wrong=float(np.mean(confident & correct)))
    return result


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="models/serve_v2_fixed")
    args = parser.parse_args()
    OUT = ROOT / args.src
    frozen = json.loads((OUT / "frozen_models.json").read_text())
    for relative, expected in frozen["model_hashes"].items():
        if sha(OUT / relative) != expected:
            raise RuntimeError("Frozen model hash mismatch")
    cv = json.loads((OUT / "cv_results.json").read_text())
    train = np.load(ROOT / "data/serve_v2/train.npz")
    # Export parity uses training data only. Testing data is loaded after export.
    parity_x = train["X"]
    exported = OUT / "tflite"
    exported.mkdir(exist_ok=True)
    parity = []
    model_paths = [OUT / "stratified" / f"fold{i}" / "model.keras" for i in range(1,6)] + [OUT / "single.keras"]
    names = [f"gru_fold{i}.tflite" for i in range(1,6)] + ["gru_single.tflite"]
    for path, name in zip(model_paths, names):
        print(f"Exporting and verifying {name}", flush=True)
        original = load(path)
        infer = _rebuild_without_augment(original)
        target = exported / name
        convert_to_tflite(infer, str(target), select_ops=True)
        expected = predict(original, parity_x)
        interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=2)
        interpreter.allocate_tensors()
        converted = []
        for x in parity_x:
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x[None].astype(np.float32))
            interpreter.invoke()
            converted.append(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0])
        converted = np.array(converted)
        max_diff = float(np.abs(converted-expected).max())
        flips = int(np.sum(converted.argmax(1) != expected.argmax(1)))
        if max_diff >= 1e-4 or flips:
            raise RuntimeError(f"Conversion parity failed for {name}: diff={max_diff}, flips={flips}")
        parity.append(dict(file=name, windows=len(parity_x), max_probability_diff=max_diff,
                           label_disagreements=flips, sha256=sha(target)))
    manifest = dict(classes=CLASSES, seq_len=128, n_features=10, ensemble=names[:5], single=names[5],
                    select_tf_ops=True, moment_matching=False, preprocessing=contract(),
                    contract_fingerprint=fingerprint(), cv_target_met=frozen["cv_target_met"],
                    cv_accuracy=cv["stratified"]["metric"]["accuracy"],
                    participant_cv_accuracy=cv["participant"]["metric"]["accuracy"],
                    training_config_sha256=frozen["training_config_sha256"],
                    deployment_status="research_candidate_pending_independent_evaluation",
                    final_coach_c_evaluation="not_available", parity=parity)
    (exported / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Export complete; evaluating frozen model on existing diagnostic testing set", flush=True)
    test = np.load(ROOT / "data/serve_v2/diagnostic_test.npz")
    datasets = {"train": train, "diagnostic_test": test}
    x = np.concatenate([d["X"] for d in datasets.values()])
    new_probs = np.array([predict(load(p), x) for p in model_paths])
    probabilities = dict(v2_ensemble=new_probs[:5].mean(0), v2_single=new_probs[5])
    # Fair app baseline: the same fresh Lite poses, old desktop window/matching, old weights.
    legacy_manifest = json.loads((ROOT / "android/app/src/main/assets/manifest.json").read_text())
    legacy_x = []
    for data in datasets.values():
        for cid in data["clip_ids"]:
            points = np.load(ROOT / "data/serve_v2/poses" / f"{cid}.npz")["points"]
            legacy_x.append(build_window(points, legacy_manifest, True)[0][0])
    legacy_x = np.stack(legacy_x)
    probabilities["legacy_desktop_ensemble"] = np.mean([predict(load(ROOT / "models/gru_runs_angles_strat5" / f"gru_fold{i}.keras"), legacy_x) for i in range(1,6)], axis=0)
    results, rows = {}, []
    offset = 0
    for split, data in datasets.items():
        n = len(data["y"])
        results[split] = {}
        for mode, all_probs in probabilities.items():
            probs = all_probs[offset:offset+n]
            quality = data["quality_accepted"] if mode.startswith("v2") else np.ones(n, bool)
            group = dict(all=evaluated(data["y"], probs, quality))
            for subject in sorted(set(data["subjects"])):
                mask = data["subjects"] == subject
                group[subject] = evaluated(data["y"][mask], probs[mask], quality[mask])
            results[split][mode] = group
            for cid, subject, y, p, q in zip(data["clip_ids"], data["subjects"], data["y"], probs, quality):
                rows.append(dict(split=split, model=mode, clip_id=str(cid), subject=str(subject), truth=CLASSES[int(y)],
                                 predicted=CLASSES[int(p.argmax())] if q else "unavailable", confidence=float(p.max()) if q else None,
                                 quality_accepted=bool(q), **dict(zip(CLASSES, map(float,p)))))
        offset += n
    (OUT / "evaluation.json").write_text(json.dumps(dict(results=results, parity=parity,
                   test_used_for_training=False, model_selection="fixed before testing: v2_ensemble",
                   test_status="previously inspected diagnostic set; not Coach C evaluation"), indent=2), encoding="utf-8")
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for name, group in results["diagnostic_test"].items():
        print(name, group["all"], flush=True)


if __name__ == "__main__":
    main()
