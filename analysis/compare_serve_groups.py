"""Compare beginner vs coach keypoint stats per serve type (drive/lob/topspin)."""
import os
import re
import numpy as np
from collections import defaultdict

KP_DIR = "data/keypoints"
PATTERN = re.compile(r"^(Beginner\d|CoachA)_(Drive|Lob|Topspin)_(\d+)\.npy$")

# BlazePose indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

clips = []
for fname in sorted(os.listdir(KP_DIR)):
    m = PATTERN.match(fname)
    if not m:
        continue
    person, serve, _ = m.groups()
    data = np.load(os.path.join(KP_DIR, fname))
    if data.ndim != 3:
        continue

    valid = ~np.isnan(data).any(axis=(1, 2))
    k = data[valid]
    if len(k) < 2:
        continue

    group = "Coach" if person.startswith("Coach") else "Beginners"

    # mean frame-to-frame joint velocity (motion energy)
    diff = np.diff(k, axis=0)
    motion = float(np.mean(np.linalg.norm(diff, axis=-1)))

    # range of motion: temporal std of each joint, averaged
    rom = float(np.mean(np.std(k, axis=0)))

    # wrist path length (cumulative displacement)
    wrists = k[:, [L_WRIST, R_WRIST], :]  # (T, 2, 2)
    wrist_path = float(np.mean(np.sum(np.linalg.norm(np.diff(wrists, axis=0), axis=-1), axis=-1)))

    # body size in image plane: mean spatial std across joints per frame
    # (controls for camera distance / zoom)
    scale = float(np.mean(np.std(k, axis=1)))

    # mean pose: spatially center each frame on mid-hip, then average over time
    mid_hip = k[:, [23, 24], :].mean(axis=1, keepdims=True)  # (T, 1, 2)
    centered = k - mid_hip
    mean_pose_flat = centered.mean(axis=0).reshape(-1)

    clips.append(dict(
        group=group, person=person, serve=serve.lower(),
        frames=int(data.shape[0]), valid_frac=float(valid.mean()),
        motion=motion, rom=rom, wrist_path=wrist_path, scale=scale,
        mean_pose=mean_pose_flat,
    ))

print(f"Loaded {len(clips)} clips\n")

def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")

metrics = ["frames", "motion", "rom", "wrist_path", "scale"]

print("=" * 78)
print("BEGINNERS vs COACH - per serve type (mean +/- std, Cohen's d)")
print("=" * 78)

for serve in ["drive", "lob", "topspin"]:
    beg = [c for c in clips if c["group"] == "Beginners" and c["serve"] == serve]
    coach = [c for c in clips if c["group"] == "Coach" and c["serve"] == serve]
    print(f"\n### {serve.upper()}  (beginners n={len(beg)}, coach n={len(coach)})")
    print(f"{'metric':<12} {'beginners':>22} {'coach':>22} {'diff%':>8} {'d':>7}")
    for m in metrics:
        a = np.array([c[m] for c in beg], dtype=float)
        b = np.array([c[m] for c in coach], dtype=float)
        d = cohen_d(a, b)
        pct = (a.mean() - b.mean()) / b.mean() * 100 if b.mean() != 0 else float("nan")
        flag = " <<<" if abs(d) >= 0.8 else (" <<" if abs(d) >= 0.5 else "")
        print(f"{m:<12} {a.mean():>10.4f} +/- {a.std():<9.4f} {b.mean():>10.4f} +/- {b.std():<9.4f} {pct:>7.1f}% {d:>7.2f}{flag}")

# Pose similarity: mean Euclidean distance between average centered poses
print("\n" + "=" * 78)
print("AVERAGE POSE SHAPE distance (mean-centered mean pose, all joints)")
print("=" * 78)
for serve in ["drive", "lob", "topspin"]:
    beg = np.mean([c["mean_pose"] for c in clips if c["group"] == "Beginners" and c["serve"] == serve], axis=0)
    coach = np.mean([c["mean_pose"] for c in clips if c["group"] == "Coach" and c["serve"] == serve], axis=0)
    dist = float(np.mean(np.linalg.norm((beg - coach).reshape(-1, 2), axis=1)))
    # baseline: beginner-vs-beginner variance for scale
    beg_poses = np.array([c["mean_pose"] for c in clips if c["group"] == "Beginners" and c["serve"] == serve])
    intra = float(np.mean([np.mean(np.linalg.norm((p - beg).reshape(-1, 2), axis=1)) for p in beg_poses]))
    print(f"{serve:<9} coach-beg dist={dist:.5f}   beg-beg intra-dist={intra:.5f}   ratio={dist/intra:.2f}x")

# Scale-normalized: divide motion/rom/wrist_path by per-clip body scale
# (controls for camera distance / person size in frame)
print("\n" + "=" * 78)
print("SCALE-NORMALIZED (metric / body scale) - controls for framing/zoom")
print("=" * 78)
for serve in ["drive", "lob", "topspin"]:
    beg = [c for c in clips if c["group"] == "Beginners" and c["serve"] == serve]
    coach = [c for c in clips if c["group"] == "Coach" and c["serve"] == serve]
    print(f"\n### {serve.upper()}")
    print(f"{'metric':<16} {'beginners':>22} {'coach':>22} {'diff%':>8} {'d':>7}")
    for m in ["motion", "rom", "wrist_path"]:
        a = np.array([c[m] / c["scale"] for c in beg])
        b = np.array([c[m] / c["scale"] for c in coach])
        d = cohen_d(a, b)
        pct = (a.mean() - b.mean()) / b.mean() * 100
        flag = " <<<" if abs(d) >= 0.8 else (" <<" if abs(d) >= 0.5 else "")
        print(f"{m+'/scale':<16} {a.mean():>10.4f} +/- {a.std():<9.4f} {b.mean():>10.4f} +/- {b.std():<9.4f} {pct:>7.1f}% {d:>7.2f}{flag}")

# Distribution of serve types (does composition differ?)
print("\n" + "=" * 78)
print("SERVE-TYPE composition (counts)")
print("=" * 78)
for g in ["Beginners", "Coach"]:
    sub = [c for c in clips if c["group"] == g]
    counts = {s: sum(1 for c in sub if c["serve"] == s) for s in ["drive", "lob", "topspin"]}
    total = len(sub)
    print(f"{g:<10} n={total:<4} " + "  ".join(f"{s}={counts[s]} ({counts[s]/total:.0%})" for s in counts))

# Validity / NaN comparison
print("\n" + "=" * 78)
print("DETECTION quality (valid frame fraction)")
print("=" * 78)
for g in ["Beginners", "Coach"]:
    fr = np.array([c["valid_frac"] for c in clips if c["group"] == g])
    print(f"{g:<10} mean={fr.mean():.4f}  min={fr.min():.4f}  max={fr.max():.4f}")
