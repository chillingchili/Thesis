"""Recorded-video adapter. Imports heavy model runtimes only inside the worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "android/app/src/main/assets"
CANDIDATE_ASSETS = ROOT / "models/serve_v2_fixed/tflite"
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/desktop/yolo"))
# Ultralytics checks its parent with os.access before creating the directory.
# Create the chosen local parent first, avoiding its /tmp fallback on Windows.
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
MAX_SECONDS = 120
_MODELS = {}


class Cancelled(Exception):
    pass


def check(cancel):
    if cancel.is_set():
        raise Cancelled("Analysis cancelled")


def read_config(name):
    return json.loads((ASSETS / name).read_text(encoding="utf-8"))


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("This file could not be opened as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 < fps <= 240 or frames < 3:
            raise ValueError("Video needs at least three frames and a valid frame rate (up to 240 fps).")
        return dict(fps=fps, frames=frames, duration=frames / fps,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def build_window(keypoints, manifest, moment_matching=True):
    import numpy as np
    from extract_joint_angles import clip_angles
    flat = clip_angles(keypoints).reshape(len(keypoints), 10)
    n = min(len(flat), manifest["seq_len"])
    window = np.zeros((manifest["seq_len"], 10), dtype=np.float32)
    valid = np.isfinite(flat[:n]).all(axis=1)
    window[:n] = np.nan_to_num(flat[:n], nan=0.0, posinf=0.0, neginf=0.0)
    # Preserve missing-frame mask rather than moment-matching missing frames into a pose.
    if moment_matching and manifest.get("moment_matching") and valid.sum() >= 2:
        observed = window[:n][valid]
        matched = ((observed - observed.mean(axis=0)) / np.maximum(observed.std(axis=0), 1e-6)
                   * np.array(manifest["feature_std"]) + np.array(manifest["feature_mean"]))
        window[:n][valid] = matched
    return window[None], int(valid.sum())


def classify(keypoints, mode, moment_matching, visibility=None, timestamps=None):
    import numpy as np
    import tensorflow as tf
    if mode not in {"single", "ensemble", "hybrid", "v2_single", "v2_ensemble"}:
        raise ValueError("Unsupported serve classifier")
    candidate = mode.startswith("v2_")
    directory = CANDIDATE_ASSETS if candidate else ASSETS
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    base_mode = mode.removeprefix("v2_")
    metadata = {}
    if mode == "hybrid":
        # Android buffers only valid pose vectors. Keep that rule for both branches.
        from extract_joint_angles import clip_angles
        finite = np.isfinite(clip_angles(keypoints)).all(axis=(1,2))
        keypoints = keypoints[finite]
        if len(keypoints) < 32:
            raise ValueError("Hybrid classification requires at least 32 usable pose frames.")
    if candidate:
        from serve_sequence import build_sequence, fingerprint
        if manifest.get("contract_fingerprint") != fingerprint():
            raise ValueError("The retrained model's preprocessing or pose asset does not match this app.")
        if timestamps is None or visibility is None:
            raise ValueError("The retrained model requires pose visibility and frame times.")
        if timestamps[-1] - timestamps[0] > manifest["preprocessing"]["max_clip_seconds"]:
            raise ValueError("Select one complete serve of up to 12 seconds.")
        window, metadata = build_sequence(keypoints, visibility, timestamps)
        if not metadata["quality_accepted"]:
            raise ValueError("Too little reliable arm tracking. Keep the serving arm visible throughout the clip.")
        valid = metadata["valid_window_frames"]
        window = window[None]
        moment_matching = False
    else:
        window, valid = build_window(keypoints, manifest, moment_matching)
    if valid < 32:
        raise ValueError(f"Only {valid} usable pose frames in the GRU window; at least 32 are required.")
    names = manifest["ensemble"] if base_mode in {"ensemble", "hybrid"} else [manifest["single"]]
    probabilities = []
    for name in names:
        cache_key = str(directory / name)
        if cache_key not in _MODELS:
            interpreter = tf.lite.Interpreter(model_path=cache_key, num_threads=2)
            interpreter.allocate_tensors()
            _MODELS[cache_key] = interpreter
        interpreter = _MODELS[cache_key]
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], window.astype(np.float32))
        interpreter.invoke()
        probabilities.append(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0])
    probs = np.mean(probabilities, axis=0)
    components = {}
    if mode == "hybrid":
        from desktop.hybrid import Knn5, timing_features, combine
        bank_key = str(ASSETS / "knn_train.bin")
        if bank_key not in _MODELS:
            _MODELS[bank_key] = Knn5(ASSETS)
        bank = _MODELS[bank_key]
        if bank.classes != manifest["classes"]:
            raise ValueError("GRU and kNN class order differs")
        raw_window, raw_valid = build_window(keypoints, manifest, False)
        knn_probs = bank.predict(timing_features(raw_window[0], raw_valid))
        components = dict(gru_probabilities=dict(zip(bank.classes,map(float,probs))),
                          knn_probabilities=dict(zip(bank.classes,map(float,knn_probs))),
                          gru_weight=.5, knn_weight=.5, knn_bank_sha256=bank.sha256,
                          knn_training_samples=len(bank.X), timing_features_version="first_velocity_zero")
        probs = combine(probs,knn_probs)
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.isclose(probs.sum(), 1, atol=0.01):
        raise ValueError("The classifier returned invalid probabilities.")
    index = int(np.argmax(probs))
    return dict(status="ok", label=manifest["classes"][index], confidence=float(probs[index]),
                probabilities=dict(zip(manifest["classes"], map(float, probs))),
                model=base_mode, model_version="serve_v2" if candidate else "legacy",
                model_manifest_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
                feedback_eligible=manifest.get("cv_target_met", False) if candidate else True,
                valid_frames=valid, window_frames=128 if candidate else min(len(keypoints), 128),
                truncated_frames=0 if candidate else max(0, len(keypoints) - 128),
                moment_matching=moment_matching, preprocessing=metadata, components=components)


def extract_pose(path, start_frame, count, fps, out, progress, cancel):
    import numpy as np
    from serve_sequence import extract_video
    def on_frame(i, total):
        check(cancel)
        if i % 10 == 0:
            progress(0.05 + 0.45 * i / total, f"Tracking body motion · {i + 1}/{total} frames")
    keypoints, visibility, timestamps, _ = extract_video(path, start_frame, count, fps, on_frame)
    np.save(out / "keypoints.npy", keypoints)
    np.savez_compressed(out / "pose_details.npz", visibility=visibility, timestamps=timestamps)
    return keypoints


def paddle_analysis(path, keypoints, start_frame, contact, out, cancel):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from extract_paddle_angles import pick_det, expand_box, crop_angle
    weights = ROOT / "runs/detect/paddle_ft/weights/best.pt"
    if not weights.exists():
        raise FileNotFoundError("Paddle weights missing: runs/detect/paddle_ft/weights/best.pt")
    if "paddle" not in _MODELS:
        _MODELS["paddle"] = YOLO(str(weights))
    config = read_config("paddle_config.json")
    first = max(0, contact - 5)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + first)
    angles = []
    try:
        for i in range(first, min(len(keypoints), contact + 6)):
            check(cancel)
            ok, frame = cap.read()
            if not ok:
                break
            res = _MODELS["paddle"].predict(frame, imgsz=640, conf=0.25, verbose=False, device="cpu")[0]
            # Reuse the offline extractor's wrist association and contour method.
            reference_shape = (frame.shape[0], frame.shape[1])
            box, _ = pick_det(res, reference_shape, keypoints, i)
            if box is not None:
                x1, y1, x2, y2 = expand_box(box, frame.shape[:2])
                gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                angle, rect, _, status = crop_angle(gray)
                if status == "ok" and angle is not None and np.isfinite(angle):
                    angles.append(float(angle))
                    if rect is not None:
                        contour = cv2.boxPoints(rect).astype(np.int32) + [x1, y1]
                        cv2.polylines(frame, [contour.astype(np.int32)], True, (60, 230, 210), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 230, 210), 2)
            if i == contact:
                # Overlay pose on the contact inspection image.
                p = keypoints[i]
                for a, b in [(11, 12), (11, 23), (12, 24), (23, 24), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]:
                    if np.isfinite(p[[a, b]]).all():
                        q = (p[[a, b]] * [frame.shape[1], frame.shape[0]]).astype(int)
                        cv2.line(frame, tuple(q[0]), tuple(q[1]), (220, 190, 100), 2)
                cv2.imwrite(str(out / "contact.jpg"), frame)
    finally:
        cap.release()
    if not angles:
        raise ValueError("No usable paddle orientation within five frames of estimated contact.")
    angle = float(np.median(angles))
    low, high = config["band_low"], config["band_high"]
    state = "OPTIMAL" if low <= angle <= high else "OUTSIDE_BASELINE"
    if state != "OPTIMAL" and config.get("direction_validated", False):
        state = "TOO_CLOSED" if angle < low else "TOO_OPEN"
    return dict(status="ok", angle=angle, low=low, high=high, state=state,
                measurements=len(angles), direction_validated=config.get("direction_validated", False))


def ffmpeg_path():
    command = shutil.which("ffmpeg")
    if command:
        return command
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def encode_video(source, target, start=0, duration=None, cancel=None):
    executable = ffmpeg_path()
    if executable is None:
        raise RuntimeError("Install imageio-ffmpeg for browser video playback.")
    cmd = [executable, "-y", "-loglevel", "error", "-ss", str(start), "-i", str(source)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    with (target.parent / "encoding.log").open("ab") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while process.poll() is None:
                if cancel is not None:
                    check(cancel)
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError("Video conversion failed; see encoding.log in this run's output folder.")
        except BaseException:
            process.kill()
            process.wait()
            raise


def analyze(primary, secondary, out, options, progress, cancel):
    import numpy as np
    from derive_shift_threshold import clip_shift
    from trim_after_contact import estimate_contact_frame
    from desktop.feedback import evaluate
    started = time.monotonic()
    info = video_info(primary)
    start = options["start"]
    end = options["end"] if options["end"] is not None else info["duration"]
    if not 0 <= start < end <= info["duration"] + 0.05 or end - start > MAX_SECONDS:
        raise ValueError("Choose a valid clip range of up to 120 seconds within the video.")
    count = min(info["frames"] - round(start * info["fps"]), round((end - start) * info["fps"]))
    first = round(start * info["fps"])
    report = dict(video=info, range=dict(start=start, end=end), options=options, warnings=[],
                  serve=dict(status="unavailable"), shift=dict(status="unavailable"),
                  paddle=dict(status="unavailable"), landing=dict(status="disabled"))
    def stream(name, fn):
        check(cancel)
        try:
            return fn()
        except Cancelled:
            raise
        except Exception as exc:
            report[name] = dict(status="unavailable", reason=str(exc))
            return None
    progress(0.02, "Loading pose model")
    keypoints = stream("serve", lambda: extract_pose(primary, first, count, info["fps"], out, progress, cancel))
    if keypoints is not None:
        missing = float((~np.isfinite(keypoints).all(axis=(1, 2))).mean())
        report["pose_coverage"] = 1 - missing
        if missing > 0.05:
            report["warnings"].append(f"Pose missing in {missing:.0%} of frames; inspect framing and occlusion.")
        progress(0.52, "Classifying serve")
        def serve_prediction():
            details = {}
            if options["model"].startswith("v2_"):
                with np.load(out / "pose_details.npz") as pose_details:
                    details = {k: pose_details[k] for k in ["visibility", "timestamps"]}
            return classify(keypoints, options["model"], options["moment_matching"], **details)
        prediction = stream("serve", serve_prediction)
        if prediction:
            report["serve"] = prediction
            if prediction["truncated_frames"]:
                report["warnings"].append("The legacy GRU uses only the first 128 frames. Its window differs from the retrained complete-serve model.")
            if not prediction.get("feedback_eligible", True):
                report["warnings"].append("This retrained candidate has not met the thesis's 85% cross-validation target. Classification is shown for inspection; coaching is withheld.")
        config = read_config("shift_config.json")
        shift = clip_shift(keypoints)
        if shift is not None and np.isfinite(shift):
            report["shift"] = dict(status="ok", value=shift, threshold=config["threshold"], sufficient=bool(shift >= config["threshold"]))
        else:
            report["shift"]["reason"] = "Too few usable hip and body-height measurements."
        # A missing wrist trajectory must not invent a contact frame.
        wrist = keypoints[:, 16]
        if np.isfinite(wrist).all(axis=1).sum() >= 10:
            contact = estimate_contact_frame(keypoints)
            report["contact"] = dict(frame=first + contact, seconds=(first + contact) / info["fps"], method="peak right-wrist speed proxy")
            progress(0.58, "Measuring paddle angle around contact")
            paddle = stream("paddle", lambda: paddle_analysis(primary, keypoints, first, contact, out, cancel))
            if paddle:
                report["paddle"] = paddle
        else:
            report["paddle"]["reason"] = "Contact unavailable: insufficient wrist tracking."
    else:
        report["shift"]["reason"] = "Pose extraction unavailable."
        report["paddle"]["reason"] = "Pose/contact extraction unavailable."
    progress(0.65, "Preparing playback")
    try:
        encode_video(primary, out / "preview.mp4", start, end - start, cancel)
    except Cancelled:
        raise
    except Exception as exc:
        report["warnings"].append(str(exc))
    mode = options["landing"]
    if mode != "off":
        def landing():
            from balltrack_pipeline import process_video
            for weights in ["runs/detect/ball_yolo26s/weights/best.pt", "runs/pose/court_ft/weights/best.pt"]:
                if not (ROOT / weights).exists():
                    raise FileNotFoundError(f"Landing model missing: {weights}")
            source = out / "preview.mp4" if mode == "primary" else secondary
            if source is None:
                raise ValueError("A second-camera video is required for this landing mode.")
            if video_info(source)["duration"] > MAX_SECONDS:
                raise ValueError("Trim the landing clip to 120 seconds or less.")
            def landing_progress(value, desc=""):
                # Existing landing runner closes native video handles on normal return.
                # Let that stage finish before acknowledging cancellation.
                progress(0.68 + 0.25 * float(value), "Tracking ball and court · " + desc)
            contact_bound = report.get("contact", {}).get("frame")
            contact_bound = contact_bound - first if mode == "primary" and contact_bound is not None else None
            summary = process_video(str(source), str(out / "landing_raw.mp4"), stride=1,
                                    progress=landing_progress, min_bounce_frame=contact_bound)
            check(cancel)
            encode_video(out / "landing_raw.mp4", out / "landing.mp4", cancel=cancel)
            return dict(status="ok" if summary["landing"] else "unavailable", result=summary["landing"],
                        reason=None if summary["landing"] else "No reliable landing was detected.",
                        court_rate=summary["court_rate"], source=mode, min_bounce_frame=contact_bound)
        progress(0.68, "Loading ball and court models")
        result = stream("landing", landing)
        if result:
            report["landing"] = result
        if mode == "primary":
            report["warnings"].append("Same-view landing is experimental; the thesis specifies a second camera for reliable landing analysis.")
    check(cancel)
    rules = read_config("feedback_rules.json")
    feedback_confidence = report["serve"].get("confidence") if report["serve"].get("feedback_eligible", True) else None
    report["feedback"] = evaluate(rules, report["serve"].get("label"), feedback_confidence,
        report["shift"].get("sufficient"), report["paddle"].get("state"), mode != "off",
        (report["landing"].get("result") or {}).get("zone"))
    report["seconds"] = round(time.monotonic() - started, 2)
    report["assets"] = {name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
                        for name in ["manifest.json", "feedback_rules.json", "shift_config.json", "paddle_config.json"]}
    report["files"] = [p.name for p in out.iterdir() if p.name in {"preview.mp4", "landing.mp4", "contact.jpg", "keypoints.npy"}]
    progress(1, "Analysis complete")
    return report
