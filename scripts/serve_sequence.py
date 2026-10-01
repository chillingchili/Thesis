"""Versioned, deterministic GRU input contract shared by training and desktop.

Input is ONE complete, manually selected serve, not an arbitrary long recording.
Resample the complete timeline; never crop away the end or extrapolate missing poses.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from extract_joint_angles import clip_angles

ROOT = Path(__file__).resolve().parents[1]
POSE_MODEL = ROOT / "android/app/src/main/assets/pose_landmarker_lite.task"
REQUIRED = [11, 12, 14, 16, 20, 23, 24]
CONTRACT = dict(version="serve_sequence_v2", pose_model="pose_landmarker_lite.task",
                pose_mode="VIDEO", detection_confidence=.5, presence_confidence=.5,
                tracking_confidence=.5, reset_per_clip=True, seq_len=128, n_features=10,
                features="right_arm_elbow_shoulder_wrist_torso_twist_sin_cos",
                coordinates="normalized_image_xy", window="complete_selected_serve",
                resampling="linear_sin_cos_then_unit_normalize", visibility_min=.5,
                min_source_frames=32, min_valid_fraction=.8, moment_matching=False,
                max_clip_seconds=12, temporal_axis="elapsed_time")


def contract():
    return dict(CONTRACT, pose_sha256=hashlib.sha256(POSE_MODEL.read_bytes()).hexdigest())


def fingerprint():
    return hashlib.sha256(json.dumps(contract(), sort_keys=True).encode()).hexdigest()


def resample_angles(features, timestamps=None, seq_len=128):
    """Preserve both endpoints; interpolate only across two adjacent valid frames.

    Missing frames have ten zeros. Renormalizing each sin/cos pair avoids angle
    wraparound and retains physical angle encoding. Android uses the same rule.
    """
    features = np.asarray(features, np.float32).reshape(-1, 10)
    if len(features) < 2:
        raise ValueError("A sequence needs at least two frames")
    times = np.arange(len(features), dtype=np.float64) if timestamps is None else np.asarray(timestamps, np.float64)
    if times.shape != (len(features),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing")
    valid = np.isfinite(features).all(1) & np.any(features != 0, axis=1)
    targets = np.linspace(times[0], times[-1], seq_len)
    right = np.searchsorted(times, targets, side="left").clip(0, len(times)-1)
    left = np.maximum(right-1, 0)
    exact = np.isclose(targets, times[right], rtol=0, atol=1e-9)
    left[exact] = right[exact]
    denom = times[right]-times[left]
    alpha = np.divide(targets-times[left], denom, out=np.zeros_like(targets), where=denom > 0)
    mixed = features[left]*(1-alpha[:, None]) + features[right]*alpha[:, None]
    pairs = mixed.reshape(seq_len, 5, 2)
    norms = np.linalg.norm(pairs, axis=2, keepdims=True)
    usable = valid[left] & valid[right] & np.all(norms[:, :, 0] > 1e-6, axis=1)
    output = np.zeros((seq_len, 5, 2), np.float32)
    output[usable] = pairs[usable] / norms[usable]
    return output.reshape(seq_len, 10)


def build_sequence(points, visibility, timestamps=None):
    points = np.asarray(points, np.float32)
    if points.ndim != 3 or points.shape[1:] != (33, 2):
        raise ValueError("Expected (frames, 33, 2) landmarks")
    visibility = np.asarray(visibility)
    if visibility.shape != points.shape[:2]:
        raise ValueError("Visibility is required for every landmark/frame")
    valid = np.isfinite(points[:, REQUIRED]).all(axis=(1, 2)) & np.all(visibility[:, REQUIRED] >= CONTRACT["visibility_min"], axis=1)
    angles = clip_angles(points).reshape(-1, 10)
    valid &= np.isfinite(angles).all(axis=1)
    angles[~valid] = 0
    window = resample_angles(angles, timestamps, CONTRACT["seq_len"])
    usable = np.any(window != 0, axis=1)
    accepted = (int(valid.sum()) >= CONTRACT["min_source_frames"]
                and float(valid.mean()) >= CONTRACT["min_valid_fraction"]
                and float(usable.mean()) >= CONTRACT["min_valid_fraction"])
    metadata = dict(contract=CONTRACT["version"], source_frames=len(points), valid_source_frames=int(valid.sum()),
                    valid_source_fraction=float(valid.mean()), valid_window_frames=int(usable.sum()),
                    valid_window_fraction=float(usable.mean()), quality_accepted=bool(accepted),
                    window_frames=len(window), source_start_frame=0, source_end_frame=len(points)-1,
                    truncated_frames=0, moment_matching=False)
    return window, metadata


def extract_video(path, start_frame=0, count=None, fps=None, on_frame=None):
    """Fresh Tasks VIDEO tracker for every recording, matching desktop extraction."""
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {path}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS)
    count = count if count is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-start_frame
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video frame rate is invalid")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=POSE_MODEL.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5, min_tracking_confidence=.5)
    points, visibility, timestamps = [], [], []
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose:
            for i in range(count):
                if on_frame:
                    on_frame(i, count)
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(i*1000/fps))
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                points.append([[p.x, p.y] for p in landmarks] if landmarks else np.full((33, 2), np.nan))
                visibility.append([p.visibility for p in landmarks] if landmarks else np.zeros(33))
                timestamps.append(i/fps)
    finally:
        cap.release()
    if len(points) < 3:
        raise ValueError("Too few decoded video frames")
    return np.asarray(points, np.float32), np.asarray(visibility, np.float32), np.asarray(timestamps), fps
