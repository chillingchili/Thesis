"""Build a paddle-only YOLO dataset from a full Pickleball.v2 ndjson export.

Reads an Ultralytics-platform ndjson manifest (datasets/*.ndjson), keeps only
images containing at least one paddle box, rewrites every label file to a
single class-0 "paddle" entry, downloads the selected images from the signed
CDN URLs (export fresh - URLs expire), and writes:

  datasets/paddle/images/{train,val,test}/<file>
  datasets/paddle/labels/{train,val,test}/<file>.txt
  datasets/paddle/data.yaml

Images with no paddle box are dropped (no negatives kept). Selection is a
seeded random subsample per split so runs are reproducible.

Usage:
  python scripts/prepare_paddle_dataset.py datasets/pickleballv2-....ndjson
  python scripts/prepare_paddle_dataset.py datasets/<file>.ndjson --max-train 6000
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_yolo_datasets import fetch  # noqa: E402


def paddle_class_id(class_names: dict) -> int:
    for k, v in class_names.items():
        if "paddle" in str(v).lower():
            return int(k)
    sys.exit(f"no paddle class in header class_names: {class_names}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ndjson")
    ap.add_argument("--out", default="paddle", help="output folder name under datasets/")
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-val", type=int, default=800)
    ap.add_argument("--max-test", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    root = Path("datasets") / args.out
    header = None
    rows = []  # (split, file, url, boxes)

    with open(args.ndjson, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "dataset":
                header = d
                continue
            if d.get("type") != "image":
                continue
            boxes = (d.get("annotations") or {}).get("boxes") or []
            rows.append((d["split"], d["file"], d["url"], boxes))

    if not header:
        sys.exit("ndjson has no dataset header line")
    pid = paddle_class_id(header["class_names"])
    print(f"paddle class id = {pid} of {header['class_names']}", flush=True)

    by_split = {}  # split -> list of (file, url, label_text)
    n_paddle = 0
    for split, fname, url, boxes in rows:
        pb = [b for b in boxes if int(b[0]) == pid]
        if not pb:
            continue
        n_paddle += 1
        label = "\n".join(
            f"0 {float(b[1]):.6f} {float(b[2]):.6f} {float(b[3]):.6f} {float(b[4]):.6f}"
            for b in pb
        )
        by_split.setdefault(split, []).append((fname, url, label))

    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    rng = random.Random(args.seed)
    jobs = []  # (url, img_dest, lbl_dest, label_text)
    for split, items in sorted(by_split.items()):
        raw = len(items)
        if split in caps and raw > caps[split]:
            items = rng.sample(items, caps[split])
        print(f"{split}: selecting {len(items)} / {raw} paddle images", flush=True)
        for fname, url, label in items:
            img_dest = root / "images" / split / fname
            lbl_dest = root / "labels" / split / (Path(fname).stem + ".txt")
            jobs.append((url, img_dest, lbl_dest, label))

    for _, _, lbl_dest, txt in jobs:
        lbl_dest.parent.mkdir(parents=True, exist_ok=True)
        lbl_dest.write_text(txt, encoding="utf-8")

    print(
        f"{args.out}: {len(jobs)} selected / {n_paddle} paddle images "
        f"(manifest has {len(rows)} rows), labels written",
        flush=True,
    )

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, url, dest): dest for url, dest, _, _ in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  {futs[fut].name}: {r}", flush=True)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} (ok={ok} skip={skip} fail={fail})", flush=True)

    data_yaml = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "paddle"},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print(f"DONE {args.out}: ok={ok} skip={skip} fail={fail} -> {root}/data.yaml", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
