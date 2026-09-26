"""Convert Ultralytics-platform ndjson manifests into local YOLO training folders.

Reads datasets/<name>.ndjson, downloads images from signed CDN URLs, and writes:
  datasets/<out>/images/{train,val,test}/<file>
  datasets/<out>/labels/{train,val,test}/<file>.txt
  datasets/<out>/data.yaml

detect -> class cx cy w h ; pose -> class x1 y1 v1 x2 y2 v2 ... (no bbox)
Idempotent: existing files are skipped (resume-safe).
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

UA = {"User-Agent": "Mozilla/5.0"}
RETRIES = 3


def fetch(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            data = urllib.request.urlopen(req, timeout=30).read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # noqa: BLE001
            if attempt == RETRIES - 1:
                return f"FAIL {e}"
            time.sleep(1.5 * (attempt + 1))
    return "FAIL"


def label_for(ann: dict, task: str) -> str:
    if task == "detect":
        lines = []
        for b in ann.get("boxes", []):
            lines.append(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}")
        return "\n".join(lines)
    # pose: [cls, cx, cy, w, h, x,y,v ...]
    pose = ann.get("pose", [])
    out = []
    for arr in pose:
        cls = int(arr[0])
        kpts = arr[5:]
        vals = " ".join(f"{v:.6f}" for v in kpts)
        out.append(f"{cls} {vals}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ndjson")
    ap.add_argument("--out", required=True, help="output folder name under datasets/")
    ap.add_argument("--task", choices=["detect", "pose"], required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    root = Path("datasets") / args.out
    header = None
    jobs = []  # (url, image_dest, label_dest, label_text)

    with open(args.ndjson, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "dataset":
                header = d
                continue
            if d.get("type") != "image":
                continue
            split = d["split"]
            img_dest = root / "images" / split / d["file"]
            lbl_dest = root / "labels" / split / (Path(d["file"]).stem + ".txt")
            txt = label_for(d.get("annotations") or {}, args.task)
            jobs.append((d["url"], img_dest, lbl_dest, txt))

    # labels: write first (instant, also for already-downloaded images)
    n_lbl = 0
    for _, img_dest, lbl_dest, txt in jobs:
        if txt:
            lbl_dest.parent.mkdir(parents=True, exist_ok=True)
            lbl_dest.write_text(txt, encoding="utf-8")
            n_lbl += 1

    print(f"{args.out}: {len(jobs)} images, {n_lbl} non-empty labels", flush=True)

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

    # data.yaml
    names = header["class_names"] if header else {"0": "object"}
    names = {int(k): v for k, v in names.items()}
    data_yaml = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    if args.task == "pose":
        data_yaml["kpt_shape"] = [14, 3]
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print(f"DONE {args.out}: ok={ok} skip={skip} fail={fail} -> {root}/data.yaml", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
