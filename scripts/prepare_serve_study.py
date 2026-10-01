"""Apply thesis pixel/video augmentation BEFORE fresh pose extraction, train only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2
import numpy as np

from serve_sequence import POSE_MODEL, build_sequence
from serve_study import DATA, ROOT, PROTOCOL, SEED, freeze_protocol, training_rows, sha, write_json, skeleton_features


def variant_parameters(cid, variant):
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{cid}:{variant}".encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    a = PROTOCOL["augmentation"]
    return dict(seed=seed, contrast=float(rng.uniform(*a["contrast_range"])),
                brightness=float(rng.uniform(*a["brightness_fraction_range"])),
                sigma=float(rng.uniform(*a["gaussian_sigma_fraction_range"])),
                crop_scale=float(rng.uniform(*a["crop_scale_range"])),
                offset_x=float(rng.random()), offset_y=float(rng.random()),
                spacing=float(rng.uniform(*a["temporal_sample_spacing_range"])))


def source_indices(count, spacing):
    indices = np.rint(np.arange(0, count - 1, spacing)).astype(int)
    return np.unique(np.r_[0, indices.clip(0, count - 1), count - 1])


def augment_frame(frame, params, rng):
    h, w = frame.shape[:2]
    scale = params["crop_scale"]
    left = params["offset_x"] * (1 - scale) * w
    top = params["offset_y"] * (1 - scale) * h
    affine = np.array([[1 / scale, 0, -left / scale], [0, 1 / scale, -top / scale]], np.float32)
    crop = cv2.warpAffine(frame, affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pixels = (crop.astype(np.float32) - 127.5) * params["contrast"] + 127.5 + params["brightness"] * 255
    # OpenCV's float32 Gaussian generator avoids a full-resolution float64
    # allocation per frame. Seed every frame for reproducibility across workers.
    noise = np.empty_like(pixels)
    cv2.setRNGSeed(int(rng.integers(1, 2147483647)))
    cv2.randn(noise, (0., 0., 0.), (params["sigma"] * 255,) * 3)
    pixels += noise
    return np.clip(pixels, 0, 255).astype(np.uint8)


def process_variant(row, variant):
    import mediapipe as mp
    cv2.setNumThreads(1)
    assert row["split"] == "train"
    path = ROOT / row["video"]
    if sha(path) != row["video_sha256"]:
        raise ValueError(f"Source changed: {path}")
    params = variant_parameters(row["clip_id"], variant)
    provenance = dict(original_id=row["clip_id"], video_sha256=row["video_sha256"], extractor_sha256=sha(__file__),
                      protocol_sha256=sha(DATA / "protocol.json"), parameters=params)
    target = DATA / "augmented" / f"{row['clip_id']}__a{variant}.npz"
    if target.exists():
        saved = np.load(target)
        previous = json.loads(str(saved["provenance"]))
        if previous == provenance:
            return json.loads(str(saved["quality"]))
        if "extractor_sha256" not in previous:
            saved.close()
            pilot = target.with_suffix(".pilot.npz")
            if pilot.exists():
                raise ValueError("Pilot already preserved")
            target.rename(pilot)
        else:
            raise ValueError("Stale augmentation cache")
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or count < 2 or fps <= 0:
        cap.release()
        raise ValueError(f"Cannot read {path}")
    selected = source_indices(count, params["spacing"])
    chosen = set(selected.tolist())
    points, visibility, times, actual = [], [], [], []
    rng = np.random.default_rng(params["seed"] + 1)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=POSE_MODEL.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5, min_tracking_confidence=.5)
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose:
            for i in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Truncated video at {i}/{count}: {path}")
                if i not in chosen:
                    continue
                frame = augment_frame(frame, params, rng)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = len(points) / fps
                result = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(ts * 1000))
                lm = result.pose_landmarks[0] if result.pose_landmarks else None
                points.append([[p.x, p.y] for p in lm] if lm else np.full((33, 2), np.nan))
                visibility.append([p.visibility for p in lm] if lm else np.zeros(33))
                times.append(ts)
                actual.append(i)
    finally:
        cap.release()
    points, visibility, times = np.array(points, np.float32), np.array(visibility, np.float32), np.array(times)
    window, quality = build_sequence(points, visibility, times)
    quality.update(original_id=row["clip_id"], variant=variant)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(temp, X=window, points=points, visibility=visibility, timestamps=times,
                        source_indices=np.array(actual), fps=fps, quality=json.dumps(quality),
                        provenance=json.dumps(provenance, sort_keys=True))
    temp.replace(target)
    return quality


def prepare_skeletons(rows):
    target = DATA / "skeleton_train.npz"
    if target.exists():
        return
    base, motion = [], []
    for row in rows:
        saved = np.load(ROOT / "data/serve_v2/poses" / f"{row['clip_id']}.npz")
        cap = cv2.VideoCapture(str(ROOT / row["video"]))
        aspect = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        a, b = skeleton_features(saved["points"], saved["visibility"], saved["timestamps"], aspect)
        base.append(a); motion.append(b)
    np.savez_compressed(target, skeleton=np.array(base), skeleton_motion=np.array(motion),
                        clip_ids=np.array([r["clip_id"] for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="smoke extraction only; never produces completion marker")
    args = parser.parse_args()
    freeze_protocol()
    rows = training_rows()
    prepare_skeletons(rows)
    jobs = [(r, v) for r in rows for v in (1, 2)]
    if args.limit:
        jobs = jobs[:args.limit]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_variant, r, v) for r, v in jobs]
        for f in as_completed(futures):
            q = f.result(); results.append(q)
            print(f"{len(results)}/{len(jobs)} {q['original_id']} a{q['variant']} quality={q['quality_accepted']}", flush=True)
    if not args.limit:
        write_json(DATA / "augmentation_quality.json", sorted(results, key=lambda r: (r["original_id"], r["variant"])))
        write_json(DATA / "extraction_complete.json", dict(variants=len(results), accepted=sum(r["quality_accepted"] for r in results),
                   protocol_sha256=sha(DATA / "protocol.json"), skeleton_sha256=sha(DATA / "skeleton_train.npz")))


if __name__ == "__main__":
    main()
