"""Matched-budget GRU input/augmentation ablations. NEVER opens diagnostic data."""
import argparse
import json
import os
import time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from extract_timing_features import timing_features
from serve_study import DATA, OUT, PROTOCOL, SEED, ROOT, training_data, freeze_protocol, sha, write_json, metrics


def fold_membership(ids, subjects, tr, va, participant=False):
    if set(ids[tr]) & set(ids[va]):
        raise ValueError("Original clip leakage")
    if participant and set(subjects[tr]) & set(subjects[va]):
        raise ValueError("Participant leakage")
    return dict(train_ids=ids[tr].tolist(), validation_ids=ids[va].tolist(),
                train_subjects=sorted(set(subjects[tr])), validation_subjects=sorted(set(subjects[va])))


def build_model(dim, seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(PROTOCOL["l2"])
    inp = tf.keras.Input((128, dim))
    x = tf.keras.layers.Masking(mask_value=0.)(inp)
    x = tf.keras.layers.GRU(PROTOCOL["gru_units"], kernel_regularizer=reg, recurrent_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(PROTOCOL["dropout"])(x)
    out = tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=reg)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def arrays(condition, d):
    if condition.startswith("angles"):
        x = d["X"]
    else:
        s = np.load(DATA / "skeleton_train.npz")
        assert s["clip_ids"].tolist() == d["clip_ids"].tolist()
        x = s[condition]
    variants = None
    if condition == "angles_thesis_aug":
        complete = json.loads((DATA / "extraction_complete.json").read_text())
        assert complete["protocol_sha256"] == sha(DATA / "protocol.json")
        variants = np.repeat(x[:, None], 3, axis=1)
        for i, cid in enumerate(d["clip_ids"]):
            for v in (1, 2):
                aug = np.load(DATA / "augmented" / f"{cid}__a{v}.npz")
                assert json.loads(str(aug["provenance"]))["original_id"] == cid
                if json.loads(str(aug["quality"]))["quality_accepted"]:
                    variants[i, v] = aug["X"]
    return x, variants


def predict(model, x):
    return np.concatenate([model(x[i:i+32], training=False).numpy() for i in range(0, len(x), 32)])


def fit_fold(condition, protocol, fold, tr, va, x, variants, d):
    folder = OUT / condition / protocol / f"fold{fold}"
    record = folder / "result.json"
    membership = fold_membership(d["clip_ids"], d["subjects"], tr, va, protocol == "participant")
    seed = SEED + fold + (100 if protocol == "participant" else 0)
    config = dict(**membership, condition=condition, protocol_sha256=sha(DATA / "protocol.json"), seed=seed,
                  feature_array_sha256=__import__("hashlib").sha256(x.tobytes()).hexdigest(),
                  variants_sha256=__import__("hashlib").sha256(variants.tobytes()).hexdigest() if variants is not None else None,
                  trainer_sha256=sha(__file__))
    if record.exists():
        result = json.loads(record.read_text())
        if result["configuration"] != config or result["model_sha256"] != sha(folder / "model.keras"):
            raise ValueError("Saved fold/configuration mismatch")
        return np.load(folder / "predictions.npy"), result
    model = build_model(x.shape[2], seed)
    # Separate deterministic RNGs preserve the same sample order for every condition.
    order_rng, augmentation_rng = np.random.default_rng(seed), np.random.default_rng(seed + 10000)
    history, exposures = [], np.zeros(3, int)
    start = time.monotonic()
    for epoch in range(PROTOCOL["epochs"]):
        order = order_rng.permutation(tr)
        choices = augmentation_rng.integers(0, 3, size=len(order)) if variants is not None else np.zeros(len(order), int)
        exposures += np.bincount(choices, minlength=3)
        model.reset_metrics()
        for i in range(0, len(order), PROTOCOL["batch_size"]):
            ix = order[i:i+PROTOCOL["batch_size"]]
            batch = x[ix] if variants is None else variants[ix, choices[i:i+len(ix)]]
            log = model.train_on_batch(batch, d["y"][ix], return_dict=True)
        history.append({k: float(v) for k, v in log.items()})
        if (epoch + 1) % 20 == 0:
            print(f"{condition} {protocol} {fold} epoch={epoch+1} loss={history[-1]['loss']:.4f} elapsed={time.monotonic()-start:.0f}s", flush=True)
    p = predict(model, x[va])
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / "model.keras")
    np.save(folder / "predictions.npy", p)
    result = dict(configuration=config, model_sha256=sha(folder / "model.keras"), parameters=model.count_params(),
                  metric=metrics(d["y"][va], p, d["subjects"][va]),
                  train_metric=metrics(d["y"][tr], predict(model, x[tr])), history=history,
                  augmentation_choices=exposures.tolist(), elapsed_seconds=time.monotonic()-start,
                  outer_validation_passed_to_fit=False)
    write_json(record, result)
    print(f"DONE {condition} {protocol} {fold} accuracy={result['metric']['accuracy']:.4f}", flush=True)
    return p, result


def run_condition(condition, d):
    x, variants = arrays(condition, d)
    timing = np.stack([timing_features(a.reshape(128, 5, 2)) for a in d["X"]])
    reports = {}
    for name, split in [("participant", LeaveOneGroupOut()),
                        ("stratified", StratifiedKFold(5, shuffle=True, random_state=20261001))]:
        oof, knn = np.zeros((len(x), 3), np.float32), np.zeros((len(x), 3), np.float32)
        folds = []
        for fold, (tr, va) in enumerate(split.split(x, d["y"], d["subjects"]), 1):
            p, result = fit_fold(condition, name, fold, tr, va, x, variants, d)
            oof[va] = p
            k = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean").fit(timing[tr], d["y"][tr])
            knn[va] = k.predict_proba(timing[va])
            folds.append(result["metric"])
        hybrid = (oof + knn) * .5
        np.savez_compressed(OUT / condition / f"{name}_oof.npz", probabilities=oof, knn=knn, hybrid=hybrid,
                            y=d["y"], subjects=d["subjects"], clip_ids=d["clip_ids"])
        reports[name] = dict(gru=metrics(d["y"], oof, d["subjects"]), knn=metrics(d["y"], knn, d["subjects"]),
                             hybrid=metrics(d["y"], hybrid, d["subjects"]), folds=folds)
        write_json(OUT / condition / "cv_results.json", reports)
    write_json(OUT / condition / "complete.json", dict(protocol_sha256=sha(DATA / "protocol.json"),
               results_sha256=sha(OUT / condition / "cv_results.json")))


def freeze_selection():
    paths = [OUT / c / "complete.json" for c in PROTOCOL["conditions"]]
    if not all(p.exists() for p in paths):
        print("Some conditions still running; selection has NOT been made.", flush=True)
        return
    scores = {c: json.loads((OUT / c / "cv_results.json").read_text())["participant"]["gru"]["mean_subject_accuracy"]
              for c in PROTOCOL["conditions"]}
    selected = max(PROTOCOL["conditions"], key=lambda c: scores[c])
    frozen = dict(selected_condition=selected, participant_mean_subject_scores=scores,
                  criterion=PROTOCOL["selection"], diagnostic_test_used=False,
                  protocol_sha256=sha(DATA / "protocol.json"),
                  models={str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.glob("*/*/fold*/model.keras"))})
    target = OUT / "frozen_selection.json"
    if target.exists() and json.loads(target.read_text()) != frozen:
        raise ValueError("Frozen selection changed")
    write_json(target, frozen)
    print(f"All candidates frozen. Training-only selection: {selected} {scores}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=PROTOCOL["conditions"], default=PROTOCOL["conditions"])
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    freeze_protocol()
    d = training_data()
    for condition in args.conditions:
        run_condition(condition, d)
    freeze_selection()


if __name__ == "__main__":
    main()
