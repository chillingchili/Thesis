"""
trim_after_contact.py

Estimates the contact frame in each clip (peak right-wrist speed, a proxy
for the moment of paddle-ball impact) and reports how many frames of
post-contact follow-through/celebration each clip carries. With --apply,
writes trimmed keypoint arrays to --out_dir, keeping N frames of buffer
past the estimated contact point instead of the full clip.

This is a dry-run-by-default tool: run it once without --apply to see the
report and sanity-check the estimated contact frames before trimming
anything.

Usage:
  # report only
  python trim_after_contact.py --keypoints_dir data/keypoints

  # actually write trimmed copies, keeping 10 frames past contact
  python trim_after_contact.py --keypoints_dir data/keypoints \
      --out_dir data/keypoints_trimmed --buffer 10 --apply
"""

import argparse
import csv
import os
import shutil

import numpy as np

RIGHT_WRIST = 16


def estimate_contact_frame(kpts: np.ndarray) -> int:
    """Returns the index of peak right-wrist frame-to-frame speed.
    NaN frames (failed detections) are treated as zero speed so they're
    never picked as the contact point. Speed is median-filtered (window 5)
    so a single tracking jump can't hijack the argmax, and if the peak
    still lands in the first 20% of the clip (pose-init spike / early
    glitch), falls back to 70% of the clip so --apply can't wipe the stroke."""
    wrist = kpts[:, RIGHT_WRIST, :]
    speed = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    speed = np.nan_to_num(speed, nan=0.0)

    # median filter: kill single-frame spikes before picking the peak
    if len(speed) >= 5:
        pad = np.pad(speed, (2, 2), mode="edge")
        speed = np.array([np.median(pad[i : i + 5]) for i in range(len(speed))])

    contact = int(np.argmax(speed)) + 1  # +1: diff index -> frame index

    # sanity guard: early peak means the heuristic failed on this clip
    if contact < 0.2 * kpts.shape[0]:
        contact = int(0.7 * kpts.shape[0])
    return contact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_dir", required=True)
    parser.add_argument("--out_dir", default=None, help="Required with --apply")
    parser.add_argument("--buffer", type=int, default=10, help="Frames to keep past contact")
    parser.add_argument("--apply", action="store_true", help="Write trimmed copies")
    args = parser.parse_args()

    if args.apply and not args.out_dir:
        parser.error("--apply requires --out_dir")

    manifest_path = os.path.join(args.keypoints_dir, "manifest.csv")
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.apply:
        os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'clip_id':30s} {'frames':>7s} {'contact':>8s} {'kept':>6s} {'trimmed %':>10s}")

    for row in rows:
        clip_id = os.path.splitext(os.path.basename(row["clip_path"]))[0]
        npy_path = os.path.join(args.keypoints_dir, f"{clip_id}.npy")
        if not os.path.exists(npy_path):
            continue

        kpts = np.load(npy_path)
        n_frames = kpts.shape[0]
        if n_frames < 3:
            continue

        contact = estimate_contact_frame(kpts)
        keep_until = min(contact + args.buffer, n_frames)
        trimmed_pct = (n_frames - keep_until) / n_frames * 100

        print(f"{clip_id:30s} {n_frames:7d} {contact:8d} {keep_until:6d} {trimmed_pct:9.1f}%")

        if args.apply:
            trimmed = kpts[:keep_until]
            np.save(os.path.join(args.out_dir, f"{clip_id}.npy"), trimmed)

    if args.apply:
        shutil.copy(manifest_path, os.path.join(args.out_dir, "manifest.csv"))
        print(f"\nTrimmed copies written to {args.out_dir}")
    else:
        print("\nDry run only — pass --apply --out_dir <dir> to actually write trimmed copies.")


if __name__ == "__main__":
    main()
