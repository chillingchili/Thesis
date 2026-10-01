"""Freeze Coach B's annotated training set and rebuild versioned Lite features.

Existing Beginner3/4 labels are diagnostic tests, never a claimed Coach C final set.
Extraction may run in separate OS processes; every clip owns a fresh tracker.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from coach_annotations import read_annotations, WORKBOOK
from serve_sequence import ROOT, contract, fingerprint, extract_video, build_sequence

OUT = ROOT / "data/serve_v2"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze():
    train_files = {p.name: p for p in (ROOT / "data/training/raw").rglob("*.mp4")}
    rows = []
    for a in read_annotations():
        path = train_files[a["Filename"]]
        rows.append(dict(clip_id=path.stem, video=path.relative_to(ROOT).as_posix(), label=a["Serve Type"].lower(),
                         subject=a["Participant"], split="train", label_source="Coach B workbook",
                         form=a["Form Judgment"], video_sha256=sha(path)))
    for filename, subject in [("manifest.csv", "Beginner3"), ("manifest_beg4.csv", "Beginner4")]:
        with (ROOT / "data/testing" / filename).open(newline="") as f:
            for row in csv.DictReader(f):
                path = ROOT / row["clip_path"].replace("\\", "/")
                rows.append(dict(clip_id=path.stem, video=path.relative_to(ROOT).as_posix(), label=row["serve_type"].lower(),
                                 subject=subject, split="diagnostic_test", label_source="existing manifest, not Coach C",
                                 form=None, video_sha256=sha(path)))
    if len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    if len({r["video_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate videos across frozen manifest")
    payload = dict(contract=contract(), contract_fingerprint=fingerprint(), annotation_sha256=sha(WORKBOOK),
                   final_evaluation_status="Coach C annotated 30-clip set not supplied", clips=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "dataset.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Frozen dataset changed; create a new version instead of overwriting")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def process(row, contract_hash):
    target = OUT / "poses" / (row["clip_id"] + ".npz")
    provenance = dict(contract_fingerprint=contract_hash, video_sha256=row["video_sha256"])
    if target.exists():
        saved = np.load(target)
        if json.loads(str(saved["provenance"])) != provenance:
            raise ValueError(f"Stale cached extraction: {target}")
        points, visibility, timestamps, fps = (saved[k] for k in ["points", "visibility", "timestamps", "fps"])
    else:
        points, visibility, timestamps, fps = extract_video(ROOT / row["video"])
        if len(points)/float(fps) > contract()["max_clip_seconds"]:
            raise ValueError(f"Clip is too long to be a single serve: {row['clip_id']}")
        np.savez_compressed(target, points=points, visibility=visibility, timestamps=timestamps, fps=fps,
                            provenance=json.dumps(provenance, sort_keys=True))
    window, metadata = build_sequence(points, visibility, timestamps)
    np.save(OUT / "windows" / (row["clip_id"] + ".npy"), window)
    return dict(**row, **metadata, fps=float(fps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    frozen = freeze()
    for name in ["poses", "windows"]:
        (OUT / name).mkdir(exist_ok=True)
    results = []
    split_counts = {s: sum(r["split"] == s for r in frozen["clips"]) for s in ["train", "diagnostic_test"]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, row, frozen["contract_fingerprint"]) for row in frozen["clips"]]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"{len(results)}/{len(futures)} {row['split']} {row['clip_id']} usable={row['quality_accepted']} valid={row['valid_source_fraction']:.3f}", flush=True)
            if sum(r["split"] == row["split"] for r in results) == split_counts[row["split"]]:
                by_id = {r["clip_id"]: r for r in results}
                selected = [by_id[r["clip_id"]] for r in frozen["clips"] if r["split"] == row["split"]]
                write_split(row["split"], selected)
    by_id = {r["clip_id"]: r for r in results}
    ordered = [by_id[r["clip_id"]] for r in frozen["clips"]]
    (OUT / "quality.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    for split in ["train", "diagnostic_test"]:
        selected = [r for r in ordered if r["split"] == split]
        write_split(split, selected)
    print("Frozen extraction complete", flush=True)


def write_split(split, selected):
    # Keep failures in the evaluation inventory; training filters them explicitly.
    np.savez_compressed(OUT / f"{split}.npz", X=np.stack([np.load(OUT / "windows" / (r["clip_id"] + ".npy")) for r in selected]),
                        y=np.array([("drive", "lob", "topspin").index(r["label"]) for r in selected]),
                        clip_ids=np.array([r["clip_id"] for r in selected]), subjects=np.array([r["subject"] for r in selected]),
                        quality_accepted=np.array([r["quality_accepted"] for r in selected]))
    (OUT / f"{split}_quality.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
