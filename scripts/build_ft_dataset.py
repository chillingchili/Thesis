"""Build the paddle fine-tune dataset from manual labels (labels.json).

Reads label_tool/labels.json ({id: {x,y,w,h} normalized | {skip:true}}),
validates against label_tool/frames.js, optionally mixes in N images from the
original public dataset (class-0 paddle labels, train split only) so the
fine-tune keeps general paddle robustness, then writes a YOLO dataset:

  datasets/paddle_ft/images/{train,val}/<id>.jpg
  datasets/paddle_ft/labels/{train,val}/<id>.txt
  datasets/paddle_ft/data.yaml

Usage:
  python scripts/build_ft_dataset.py --labels label_tool/labels.json \
      --frames label_tool/frames --manifest label_tool/frames.js \
      --mix-original 800 --out datasets/paddle_ft
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix-original", type=int, default=800)
    ap.add_argument("--original", default="datasets/paddle")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    js = Path(args.manifest).read_text(encoding="utf-8")
    ids = re.findall(r'"id":\s*"([^"]+)"', js)
    print(f"manifest frames: {len(ids)}, label entries: {len(labels)}")

    labeled, skipped, missing, unlabeled = [], [], [], []
    for i in ids:
        if i not in labels:
            unlabeled.append(i)
            continue
        b = labels[i]
        if b.get("skip"):
            skipped.append(i)
            continue
        if "x" not in b or b["w"] <= 0.001 or b["h"] <= 0.001:
            unlabeled.append(i)
            continue
        b["x"] = max(0.0, min(b["x"], 1.0 - b["w"]))
        b["y"] = max(0.0, min(b["y"], 1.0 - b["h"]))
        labeled.append(i)

    print(f"labeled={len(labeled)} skipped={len(skipped)} "
          f"unlabeled={len(unlabeled)}")
    if len(labeled) < 50:
        raise SystemExit(f"only {len(labeled)} labeled frames - not enough to fine-tune")

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rng.shuffle(labeled)
    n_val = max(1, int(len(labeled) * args.val_frac))
    splits = {"val": labeled[:n_val], "train": labeled[n_val:]}

    frames = Path(args.frames)
    for split, members in splits.items():
        for i in members:
            src = frames / f"{i}.jpg"
            shutil.copy2(src, out / "images" / split / f"{i}.jpg")
            b = labels[i]
            (out / "labels" / split / f"{i}.txt").write_text(
                f"0 {b['x'] + b['w'] / 2:.6f} {b['y'] + b['h'] / 2:.6f} "
                f"{b['w']:.6f} {b['h']:.6f}\n", encoding="utf-8")

    n_mix = 0
    orig = Path(args.original)
    if args.mix_original > 0 and orig.exists():
        pool = sorted(p for p in (orig / "images" / "train").glob("*")
                      if (orig / "labels" / "train" / (p.stem + ".txt")).exists())
        rng.shuffle(pool)
        for p in pool[: args.mix_original]:
            shutil.copy2(p, out / "images" / "train" / p.name)
            shutil.copy2(orig / "labels" / "train" / (p.stem + ".txt"),
                         out / "labels" / "train" / (p.stem + ".txt"))
            n_mix += 1

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: paddle\n", encoding="utf-8")

    print(f"train={len(splits['train']) + n_mix} (ft {len(splits['train'])} + "
          f"mix {n_mix})  val={len(splits['val'])}")
    print(f"skipped kept out: {len(skipped)}, still unlabeled: {len(unlabeled)}")
    if unlabeled:
        print(f"unlabeled: {', '.join(unlabeled[:15])}"
              + (" ..." if len(unlabeled) > 15 else ""))
    print(f"-> {out}/data.yaml")


if __name__ == "__main__":
    main()
