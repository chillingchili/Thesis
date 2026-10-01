"""Read-only benchmark of bundled Android classifier assets and gate replay.

Uses cached fresh Lite VIDEO landmarks, not live Android camera callbacks.
Reports all predeclared windows; never chooses thresholds/weights from tests.
"""
import csv
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from serve_study import ROOT, CLASSES, sha, write_json, metrics
from extract_joint_angles import clip_angles
sys.path.insert(0, str(ROOT))
from desktop.hybrid import Knn5, timing_features, combine

ASSETS = ROOT / "android/app/src/main/assets"
DEST = ROOT / "outputs/serve_study_v3/hybrid"


def padded(sequence):
    sequence = np.asarray(sequence, np.float32).reshape(-1, 10)
    w = np.zeros((128, 10), np.float32)
    n = min(len(sequence), 128)
    w[:n] = sequence[-128:]
    return w, n


def moment_match(window, n, manifest):
    if not manifest.get("moment_matching") or n < 2:
        return window.copy()
    w = window.copy()
    mean = np.mean(w[:n], axis=0, dtype=np.float64).astype(np.float32)
    delta = w[:n] - mean
    std = np.sqrt(np.mean(delta * delta, axis=0, dtype=np.float64).astype(np.float32)).clip(1e-6)
    w[:n] = delta / std * np.asarray(manifest["feature_std"], np.float32) + np.asarray(manifest["feature_mean"], np.float32)
    return w


def gate_replay(features, times):
    """MainActivity cold-start gate, original video cadence, no artificial EOF flush.

    Final inference bypasses EMA in MainActivity. Multiple events are preserved
    for audit; the last completed event is the displayed final result.
    """
    f32 = np.float32
    active, low, count = False, 0, 0
    mean, dev = f32(0), f32(.02)
    refractory = 0.
    ring, idx, ring_count, previous = np.zeros(5, np.float32), 0, 0, None
    events, buffer = [], []
    start = None
    for frame_id, (feat, t) in enumerate(zip(features, times)):
        if not np.isfinite(feat).all():
            continue
        angles = np.arctan2(feat[0::2], feat[1::2])
        inst = f32(0)
        if previous is not None:
            delta = angles - previous
            delta[delta > np.pi] -= f32(2 * np.pi)
            delta[delta < -np.pi] += f32(2 * np.pi)
            inst = f32(np.abs(delta).sum(dtype=np.float32) / f32(5))
        previous = angles
        ring[idx] = inst; idx = (idx + 1) % 5; ring_count = min(ring_count + 1, 5)
        motion = f32(ring[:ring_count].sum(dtype=np.float32) / ring_count)
        if not active:
            threshold = min(max(f32(.07), mean + f32(3) * dev), f32(.5))
            if t * 1000 >= refractory and motion >= threshold:
                active, low, count, buffer, start = True, 0, 1, [feat], frame_id
            else:
                mean += f32(.1) * (motion - mean)
                dev += f32(.1) * (abs(motion - mean) - dev)
            continue
        buffer.append(feat); buffer = buffer[-128:]; count += 1
        end = min(max(f32(.05), mean + f32(1.5) * dev), f32(.5))
        if motion < end:
            low += 1
            mean += f32(.05) * (motion - mean)
            dev += f32(.05) * (abs(motion - mean) - dev)
        else:
            low = 0
        if low >= 15 or count >= 160:
            w, n = padded(buffer)
            events.append(dict(start_frame=start, end_frame=frame_id, active_frames=count,
                               window=w, valid_frames=n, available=n >= 32))
            active, low, count, buffer = False, 0, 0, []
            refractory = t * 1000 + 1200
    return events, active


def interpreters(manifest):
    result = []
    for name in manifest["ensemble"]:
        m = tf.lite.Interpreter(model_path=str(ASSETS / name), num_threads=2)
        m.allocate_tensors()
        result.append(m)
    return result


def infer(models, w):
    ps = []
    for m in models:
        m.set_tensor(m.get_input_details()[0]["index"], w[None].astype(np.float32))
        m.invoke()
        ps.append(m.get_tensor(m.get_output_details()[0]["index"])[0].copy())
    return np.mean(ps, axis=0)


def main():
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    bank = Knn5(ASSETS)
    training = np.load(ROOT / "features/timing_train.npz", allow_pickle=True)
    np.testing.assert_array_equal(bank.X, training["X"].astype(np.float32))
    np.testing.assert_array_equal(bank.y, [CLASSES.index(str(s)) for s in training["y"]])
    config = dict(assets={n: sha(ASSETS / n) for n in manifest["ensemble"] + ["knn_train.bin", "knn_meta.json", "manifest.json"]},
                  primary="first128_recorded_window", additional=["last128_recorded_window", "android_gate_replay"],
                  gru_moment_matching=True, knn_moment_matching=False, hybrid_gru_weight=.5,
                  replay="fresh tracker per recording, every decoded VIDEO pose callback, cold-start gate, final result without EMA, no EOF flush",
                  actual_device_connected=False, limitations="does not reproduce LIVE_STREAM frame drops, warm gate state across recordings or physical device runtime",
                  bank_samples=len(bank.y), source_sha256=sha(__file__), test_use="previously inspected diagnostic clips; no fitting/selection")
    target = DEST / "configuration.json"
    if target.exists() and json.loads(target.read_text()) != config:
        raise ValueError("Benchmark inputs changed")
    write_json(target, config)
    models = interpreters(manifest)
    data = np.load(ROOT / "data/serve_v2/diagnostic_test.npz")
    assert not set(data["clip_ids"]) & set(training["clip_ids"])
    results, records, gate_info, fixtures = {}, [], [], []
    windows = {name: [] for name in [config["primary"]] + config["additional"]}
    for cid in data["clip_ids"]:
        s = np.load(ROOT / "data/serve_v2/poses" / f"{cid}.npz")
        a = clip_angles(s["points"]).reshape(-1, 10)
        finite = np.isfinite(a).all(1)
        # Android ignores callbacks without a detected pose and doesn't inspect visibility.
        a = a[finite]; times = s["timestamps"][finite]
        windows["first128_recorded_window"].append((*padded(a[:128]), len(a) >= 32))
        windows["last128_recorded_window"].append((*padded(a), len(a) >= 32))
        events, unfinished = gate_replay(a, times)
        last = events[-1] if events else None
        windows["android_gate_replay"].append((last["window"], last["valid_frames"], last["available"]) if last else (np.zeros((128, 10), np.float32), 0, False))
        gate_info.append(dict(clip_id=str(cid), completed_events=len(events), unfinished_at_eof=unfinished,
                             events=[{k: v for k, v in e.items() if k != "window"} for e in events]))
    for window_name, cases in windows.items():
        gru, knn, available = [], [], []
        for i, (w, n, accepted) in enumerate(cases):
            gp = infer(models, moment_match(w, n, manifest)) if accepted else np.zeros(3, np.float32)
            kp = bank.predict(timing_features(w, n)) if accepted else np.zeros(3, np.float32)
            gru.append(gp); knn.append(kp); available.append(accepted)
            if i < 3 and window_name == "first128_recorded_window":
                fixtures.append(dict(clip_id=str(data["clip_ids"][i]), valid_frames=n, window=w.reshape(-1).tolist(),
                                     features=timing_features(w, n).tolist(), knn_probabilities=kp.tolist()))
        modes = dict(gru=np.array(gru), knn=np.array(knn), hybrid=combine(gru, knn))
        results[window_name] = {m: metrics(data["y"], p, data["subjects"], available) for m, p in modes.items()}
        for name, probs in modes.items():
            for cid, y, p, ok in zip(data["clip_ids"], data["y"], probs, available):
                records.append(dict(clip_id=str(cid), window=window_name, model=name, truth=CLASSES[int(y)],
                                    predicted=CLASSES[int(p.argmax())] if ok else "unavailable",
                                    available=bool(ok), confidence=float(p.max()), **dict(zip(CLASSES, map(float, p)))))
        print(window_name, {m: round(r["accuracy"], 4) for m, r in results[window_name].items()}, flush=True)
    write_json(DEST / "results.json", results)
    write_json(DEST / "gate_events.json", gate_info)
    write_json(DEST / "parity_fixtures.json", dict(cases=fixtures))
    with (DEST / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


if __name__ == "__main__":
    main()
