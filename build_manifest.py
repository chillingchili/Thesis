"""
build_manifest.py

Auto-generates manifest.csv (clip_path, serve_type) by scanning a directory
of video clips, instead of hand-typing it. Works two ways:

  1. Folder-based: clips organized like
         data/raw/drive/*.mp4
         data/raw/lob/*.mp4
         data/raw/topspin/*.mp4
     -> serve_type is read from the immediate parent folder name.

  2. Filename-based: clips named like your recording convention,
         CoachA_Drive_01.mp4, Beginner1_Lob_23.mp4, ...
     -> serve_type is read from whichever of drive/lob/topspin appears
        in the filename.

Folder name takes priority if both are present and disagree; anything
that matches neither is skipped and printed as a warning so you can fix
the filename/location by hand.

Usage:
  python build_manifest.py --root data/raw --out data/manifest.csv
"""

import argparse
import csv
import os
import re

CLASSES = ["drive", "lob", "topspin"]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

FILENAME_PATTERN = re.compile(
    r"(?i)(?<![a-z])(" + "|".join(CLASSES) + r")(?![a-z])"
)


def serve_type_from_folder(path: str) -> str | None:
    parent = os.path.basename(os.path.dirname(path)).lower()
    return parent if parent in CLASSES else None


def serve_type_from_filename(path: str) -> str | None:
    match = FILENAME_PATTERN.search(os.path.basename(path))
    return match.group(1).lower() if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder to scan recursively")
    parser.add_argument("--out", required=True, help="Where to write manifest.csv")
    args = parser.parse_args()

    rows = []
    skipped = []

    for dirpath, _, filenames in os.walk(args.root):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS:
                continue

            clip_path = os.path.join(dirpath, fname)
            serve_type = serve_type_from_folder(clip_path) or serve_type_from_filename(
                clip_path
            )

            if serve_type is None:
                skipped.append(clip_path)
                continue

            rows.append({"clip_path": clip_path, "serve_type": serve_type})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "serve_type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clips to {args.out}")
    counts = {c: sum(1 for r in rows if r["serve_type"] == c) for c in CLASSES}
    print(f"  breakdown: {counts}")

    if skipped:
        print(f"\n{len(skipped)} clip(s) skipped — couldn't determine serve type:")
        for path in skipped:
            print(f"  {path}")


if __name__ == "__main__":
    main()
