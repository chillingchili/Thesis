"""Extract one contact frame per clip for manual paddle labeling.

Coach clips: all. Other roots: stratified sample per serve type (seeded).
Writes <out>/frames/<clip_id>.jpg (JPEG q95) and <out>/frames.js with
const FRAMES = [{id, src, group, serve}] for the label tool (file:// safe).

Usage:
  python scripts/extract_label_frames.py \
      --videos data/training/raw/coach data/training/raw/beg1 data/training/raw/beg2 \
      --keypoints data/training/keypoints --out label_tool --per-type 45
"""
import argparse
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trim_after_contact import estimate_contact_frame  # noqa: E402

SERVE_RE = re.compile(r"(drive|lob|topspin)", re.I)


def group_of(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for key, name in (("coach", "CoachA"), ("beg1", "Beg1"), ("beg2", "Beg2")):
        if any(key in p for p in parts):
            return name
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--out", default="label_tool")
    ap.add_argument("--per-type", type=int, default=45,
                    help="sample size per serve type for non-coach roots")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    entries = []
    for root in args.videos:
        vids = sorted(Path(root).rglob("*.mp4"))
        group = group_of(Path(root))
        if group != "CoachA" and args.per_type > 0:
            by_type = {}
            for v in vids:
                m = SERVE_RE.search(v.stem)
                key = m.group(1).lower() if m else "none"
                by_type.setdefault(key, []).append(v)
            picked = []
            for key in sorted(by_type):
                pool = by_type[key]
                rng.shuffle(pool)
                picked += pool[: args.per_type]
            vids = sorted(picked)
        for v in vids:
            npy = Path(args.keypoints) / f"{v.stem}.npy"
            if not npy.exists():
                continue
            kpts = np.load(npy)
            if kpts.shape[0] < 30:
                continue
            contact = estimate_contact_frame(kpts)
            cap = cv2.VideoCapture(str(v))
            cap.set(cv2.CAP_PROP_POS_FRAMES, contact)
            ok, img = cap.read()
            if not ok:
                cap.release()
                cap = cv2.VideoCapture(str(v))
                img, ok = None, False
                for _ in range(contact + 1):
                    ok, img = cap.read()
                    if not ok:
                        break
            cap.release()
            if not ok:
                continue
            m = SERVE_RE.search(v.stem)
            entries.append({
                "id": v.stem,
                "src": f"frames/{v.stem}.jpg",
                "group": group,
                "serve": m.group(1).lower() if m else "",
                "contact": int(contact),
            })
            cv2.imwrite(str(frames / f"{v.stem}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"{root}: cumulative {len(entries)} frames", flush=True)

    entries.sort(key=lambda e: (e["group"], e["serve"], e["id"]))
    js = "const FRAMES = " + repr(entries).replace("'", '"') + ";\n"
    (out / "frames.js").write_text(js, encoding="utf-8")
    counts = {}
    for e in entries:
        counts[e["group"]] = counts.get(e["group"], 0) + 1
    print(f"DONE {len(entries)} frames {counts} -> {out}/frames.js", flush=True)


if __name__ == "__main__":
    main()
