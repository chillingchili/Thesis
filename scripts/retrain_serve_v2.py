"""Train only the frozen Coach B partition. No test arrays are opened here.

Nested epoch selection: outer validation clips are never used for early stopping.
Writes explicit fold membership, histories, OOF predictions and model hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from train_gru import build_model
from serve_sequence import ROOT, contract, fingerprint

OUT = ROOT / "models/serve_v2"
SETTINGS = dict(seed=20261001, max_epochs=100, patience=12, batch_size=16,
                hidden_units=32, l2=1e-4, dropout=.15, recurrent_dropout=0., angle_noise=.05,
                feature_noise=.01, rate_min=.9, rate_max=1.1, max_shift=4,
                frame_drop=.03, style_offset=0., style_scale=0., inner_validation_fraction=.2,
                epoch_selection="nested_inner_validation_then_refit_outer_train",
                primary_model="five_fold_ensemble", minimum_cv_accuracy=.85,
                test_use="post_freeze_diagnostic_only_no_model_selection")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def new_model(seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    names = ["hidden_units", "l2", "dropout", "recurrent_dropout", "angle_noise", "feature_noise",
             "rate_min", "rate_max", "max_shift", "frame_drop", "style_offset", "style_scale"]
    return build_model(128, 10, 3, **{k: SETTINGS[k] for k in names})


def metric(y, p):
    pred = p.argmax(1)
    accepted = p.max(1) >= .6
    return dict(n=len(y), correct=int(np.sum(pred == y)), accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=[0,1,2], average="macro", zero_division=0)),
                confusion=confusion_matrix(y, pred, labels=[0,1,2]).tolist(),
                coverage=float(accepted.mean()), accepted_wrong=int(np.sum(accepted & (pred != y))),
                accepted_accuracy=float(np.mean(pred[accepted] == y[accepted])) if accepted.any() else None,
                accuracy_abstentions_wrong=float(np.mean(accepted & (pred == y))))


def train_fold(protocol, fold, tr, va, X, y, ids, groups):
    folder = f"{protocol}/fold{fold}"
    model_path = OUT / folder / "model.keras"
    report_path = OUT / folder / "result.json"
    if model_path.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if report["train_ids"] != ids[tr].tolist() or report["validation_ids"] != ids[va].tolist():
            raise RuntimeError("Saved fold membership does not match")
        model = tf.keras.models.load_model(model_path, compile=False)
        return model(X[va], training=False).numpy(), report
    seed = SETTINGS["seed"] + fold + (100 if protocol == "participant" else 0)
    if SETTINGS["epoch_selection"] == "fixed_100_after_training_only_pilot":
        model = new_model(seed+1000)
        print(f"{protocol} fold {fold}: fixed 100 epochs; outer validation never passed to fit", flush=True)
        history = model.fit(X[tr], y[tr], epochs=100, batch_size=SETTINGS["batch_size"], verbose=2)
        p = model(X[va], training=False).numpy()
        report = dict(train_ids=ids[tr].tolist(), validation_ids=ids[va].tolist(),
                      train_subjects=sorted(set(groups[tr])), validation_subjects=sorted(set(groups[va])),
                      selected_epochs=100, seed=seed, metric=metric(y[va], p),
                      train_metric=metric(y[tr], model(X[tr], training=False).numpy()),
                      refit_history={k:list(map(float,v)) for k,v in history.history.items()})
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        report["model_sha256"] = sha(model_path)
        save(f"{folder}/result.json",report)
        print(f"{protocol} fold {fold}: outer accuracy={report['metric']['accuracy']:.3f}",flush=True)
        return p, report
    inner = StratifiedShuffleSplit(1, test_size=SETTINGS["inner_validation_fraction"], random_state=seed)
    fit, stop = next(inner.split(X[tr], y[tr]))
    print(f"{protocol} fold {fold}: nested epoch selection, outer n={len(tr)}/{len(va)}", flush=True)
    model = new_model(seed)
    history = model.fit(X[tr[fit]], y[tr[fit]], validation_data=(X[tr[stop]], y[tr[stop]]),
                        epochs=SETTINGS["max_epochs"], batch_size=SETTINGS["batch_size"], verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=SETTINGS["patience"], restore_best_weights=True)])
    best_epochs = int(np.argmin(history.history["val_loss"])) + 1
    print(f"{protocol} fold {fold}: refit outer training only for {best_epochs} epochs", flush=True)
    model = new_model(seed+1000)
    refit = model.fit(X[tr], y[tr], epochs=best_epochs, batch_size=SETTINGS["batch_size"], verbose=2)
    p = model(X[va], training=False).numpy()
    train_p = model(X[tr], training=False).numpy()
    report = dict(train_ids=ids[tr].tolist(), validation_ids=ids[va].tolist(),
                  train_subjects=sorted(set(groups[tr])), validation_subjects=sorted(set(groups[va])),
                  inner_fit_ids=ids[tr[fit]].tolist(), inner_early_stop_ids=ids[tr[stop]].tolist(),
                  selected_epochs=best_epochs, seed=seed, metric=metric(y[va], p), train_metric=metric(y[tr], train_p),
                  inner_history={k: list(map(float,v)) for k,v in history.history.items()},
                  refit_history={k: list(map(float,v)) for k,v in refit.history.items()})
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    report["model_sha256"] = sha(model_path)
    save(f"{folder}/result.json", report)
    print(f"{protocol} fold {fold}: outer accuracy={report['metric']['accuracy']:.3f}", flush=True)
    return p, report


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="models/serve_v2")
    parser.add_argument("--schedule", choices=["nested", "fixed100"], default="nested")
    args = parser.parse_args()
    OUT = ROOT / args.out_dir
    if args.schedule == "fixed100":
        SETTINGS["epoch_selection"] = "fixed_100_after_training_only_pilot"
        SETTINGS["pilot_directory"] = "models/serve_v2"
        SETTINGS["cv_interpretation"] = "training development CV; schedule informed by earlier training-only pilot"
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    dataset_path = ROOT / "data/serve_v2/dataset.json"
    frozen = json.loads(dataset_path.read_text())
    if frozen["contract_fingerprint"] != fingerprint():
        raise RuntimeError("Feature contract differs from frozen extraction")
    train = np.load(ROOT / "data/serve_v2/train.npz")
    all_ids = train["clip_ids"]
    expected = [r["clip_id"] for r in frozen["clips"] if r["split"] == "train"]
    if all_ids.tolist() != expected or len(expected) != 210:
        raise RuntimeError("Training must contain exactly the frozen 210 Coach B IDs before quality filtering")
    eligible = train["quality_accepted"]
    X, y, ids, groups = (train[k][eligible] for k in ["X", "y", "clip_ids", "subjects"])
    if set(groups) != {"CoachA", "Beginner1", "Beginner2"}:
        raise RuntimeError("Unexpected subject in training")
    configuration = dict(settings=SETTINGS, contract=contract(), dataset_sha256=sha(dataset_path),
                         training_content_sha256=hashlib.sha256(b"".join(train[k].tobytes() for k in
                             ["X", "y", "clip_ids", "subjects", "quality_accepted"])).hexdigest(),
                         training_ids=ids.tolist(), excluded_training_ids=all_ids[~eligible].tolist(),
                         class_counts=np.bincount(y, minlength=3).tolist())
    config_path = OUT / "training_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise RuntimeError("Training configuration changed; use a new output version")
    save("training_config.json", configuration)
    completed = OUT / "frozen_models.json"
    if completed.exists():
        record = json.loads(completed.read_text())
        if record["training_config_sha256"] != sha(config_path):
            raise RuntimeError("Frozen training configuration mismatch")
        for relative, expected_hash in record["model_hashes"].items():
            if sha(OUT / relative) != expected_hash:
                raise RuntimeError("Frozen model file changed")
        print("This training run is already complete; all frozen hashes verified.", flush=True)
        return
    reports, selected_epochs = {}, []
    for protocol, splitter in [("stratified", StratifiedKFold(5, shuffle=True, random_state=SETTINGS["seed"])),
                               ("participant", LeaveOneGroupOut())]:
        oof = np.zeros((len(X), 3), np.float32)
        folds = []
        for fold, (tr, va) in enumerate(splitter.split(X, y, groups), 1):
            if set(ids[tr]) & set(ids[va]):
                raise RuntimeError("Fold overlap")
            if protocol == "participant" and set(groups[tr]) & set(groups[va]):
                raise RuntimeError("Player leakage")
            p, report = train_fold(protocol, fold, tr, va, X, y, ids, groups)
            oof[va] = p
            folds.append(report)
            if protocol == "stratified":
                selected_epochs.append(report["selected_epochs"])
        np.savez_compressed(OUT / f"{protocol}_oof.npz", probabilities=oof, y=y, clip_ids=ids, subjects=groups)
        reports[protocol] = dict(metric=metric(y, oof), folds=[r["metric"] for r in folds],
                                 per_subject={s: metric(y[groups==s], oof[groups==s]) for s in sorted(set(groups))})
        save("cv_results.json", reports)
    epochs = int(np.median(selected_epochs))
    model = new_model(SETTINGS["seed"] + 9999)
    print(f"Final single model: all {len(X)} eligible training clips, {epochs} epochs; no test input", flush=True)
    model.fit(X, y, epochs=epochs, batch_size=SETTINGS["batch_size"], verbose=2)
    model.save(OUT / "single.keras")
    paths = [OUT / "stratified" / f"fold{i}" / "model.keras" for i in range(1,6)] + [OUT / "single.keras"]
    save("frozen_models.json", dict(model_hashes={p.relative_to(OUT).as_posix(): sha(p) for p in paths},
                                    single_epochs=epochs, training_config_sha256=sha(config_path),
                                    test_data_used_for_training=False,
                                    cv_target_met=reports["stratified"]["metric"]["accuracy"] >= .85))
    print("Training finished. Weights frozen before any testing-set evaluation.", flush=True)


if __name__ == "__main__":
    main()
