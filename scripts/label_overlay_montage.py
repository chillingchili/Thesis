"""QA montage: draw manual labels from labels.json onto their frames.

Random sample (seeded) of labeled frames with red boxes, for visual
verification of label quality before fine-tuning.

Usage:
  python scripts/label_overlay_montage.py --labels labels.json \
      --frames label_tool/frames --manifest label_tool/frames.js \
      --out outputs/label_qa_montage.jpg --n 12 --seed 3
"""
import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    js = Path(args.manifest).read_text(encoding="utf-8")
    ids = re.findall(r'"id":\s*"([^"]+)"', js)
    labeled = [i for i in ids
               if i in labels and not labels[i].get("skip") and "x" in labels[i]]
    random.Random(args.seed).shuffle(labeled)
    picks = labeled[: args.n]

    tiles = []
    for i in picks:
        img = cv2.imread(str(Path(args.frames) / f"{i}.jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        b = labels[i]
        x1 = max(0, int(b["x"] * W))
        y1 = max(0, int(b["y"] * H))
        x2 = min(W - 1, int((b["x"] + b["w"]) * W))
        y2 = min(H - 1, int((b["y"] + b["h"]) * H))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, i, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2)
        tiles.append(cv2.resize(img, (640, 360)))

    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"wrote {len(picks)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
