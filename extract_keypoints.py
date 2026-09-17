"""
extract_keypoints.py

Extracts 33-point 2D BlazePose keypoint sequences from serve video clips.

Expected input:
  A manifest CSV with columns: clip_path, serve_type
    - clip_path : path to the trimmed video clip
    - serve_type: "drive" | "lob" | "topspin"
  (A "label" good/bad column can be added later once Coach B's annotation
  is done, but this script and the GRU it feeds don't need it.)

Output:
  One .npy file per clip in --out_dir, shape (num_frames, 33, 2), holding
  BlazePose's normalized (0-1) image-plane x,y coordinates. A copy of the
  manifest is placed alongside so train_gru.py doesn't need the raw paths.

Usage:
  python extract_keypoints.py --manifest data/manifest.csv --out_dir data/keypoints

Install (if needed):
  pip install mediapipe opencv-python numpy --break-system-packages
"""

import argparse
import csv
import os
import shutil

import cv2
import mediapipe as mp
import numpy as np

NUM_LANDMARKS = 33


def extract_clip_keypoints(video_path: str, pose) -> np.ndarray:
    """Run BlazePose over every frame of a clip. Returns (frames, 33, 2).
    Frames where detection fails are filled with NaN so train_gru.py (or a
    manual review pass) can decide how to handle them, rather than silently
    dropping frames."""
    cap = cv2.VideoCapture(video_path)
    frames_kpts = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks is None:
            frames_kpts.append(np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32))
            continue

        # Only x, y (image-plane) are kept. BlazePose's z is a non-metric
        # pseudo-depth and is deliberately discarded.
        kpts = np.array(
            [[lm.x, lm.y] for lm in result.pose_landmarks.landmark],
            dtype=np.float32,
        )
        frames_kpts.append(kpts)

    cap.release()
    if not frames_kpts:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(frames_kpts, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
    ) as pose:
        with open(args.manifest, newline="") as f:
            rows = list(csv.DictReader(f))

        for i, row in enumerate(rows):
            clip_path = row["clip_path"]
            clip_id = os.path.splitext(os.path.basename(clip_path))[0]
            out_path = os.path.join(args.out_dir, f"{clip_id}.npy")

            if os.path.exists(out_path):
                print(f"[{i + 1}/{len(rows)}] skip (already extracted): {clip_id}")
                continue

            print(f"[{i + 1}/{len(rows)}] extracting: {clip_id}")
            kpts = extract_clip_keypoints(clip_path, pose)

            nan_frac = np.isnan(kpts).any(axis=(1, 2)).mean()
            if nan_frac > 0.05:
                print(
                    f"  WARNING: {nan_frac:.1%} of frames had no detected pose "
                    f"in {clip_id} (over your 5% frame-drop threshold) — "
                    f"flag for manual review."
                )

            np.save(out_path, kpts)

    shutil.copy(args.manifest, os.path.join(args.out_dir, "manifest.csv"))
    print("Done.")


if __name__ == "__main__":
    main()
