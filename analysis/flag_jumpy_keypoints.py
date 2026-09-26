"""Find clips with jumbled/randomly-jumping keypoints via group-relative outliers."""
import os
import re
import numpy as np

KP_DIR = "data/keypoints"
PATTERN = re.compile(r"^(Beginner\d|CoachA)_(Drive|Lob|Topspin)_(\d+)\.npy$")

clips = []
for fname in sorted(os.listdir(KP_DIR)):
    m = PATTERN.match(fname)
    if not m:
        continue
    data = np.load(os.path.join(KP_DIR, fname))
    if data.ndim != 3:
        continue
    valid = ~np.isnan(data).any(axis=(1, 2))
    k = data[valid]
    if len(k) < 3:
        continue

    group = "Coach" if m.group(1).startswith("Coach") else "Beginners"
    scale = float(np.mean(np.std(k, axis=1)))
    diff = np.linalg.norm(np.diff(k, axis=0), axis=-1)  # (T-1, 33)

    # fraction of individual joint-steps that are impossible jumps (>0.5*scale)
    joint_tele_rate = float((diff > 0.5 * scale).mean())

    # single-frame snap-back spikes (joint out then reverses hard) as fraction
    if len(k) >= 4:
        v1, v2 = diff[:-1], diff[1:]
        d1 = np.diff(k, axis=0)[:-1]
        d2 = np.diff(k, axis=0)[1:]
        cos = np.sum(d1 * d2, axis=-1) / np.maximum(v1 * v2, 1e-12)
        big = (v1 > 0.2 * scale) & (v2 > 0.2 * scale) & (cos < -0.7)
        spike_rate = float(big.mean())
    else:
        spike_rate = 0.0

    # whole-body teleports as rate
    center = k.mean(axis=1)
    csteps = np.linalg.norm(np.diff(center, axis=0), axis=1)
    body_tele_rate = float((csteps > 0.2 * scale).mean())

    # jitter: accel/vel
    vel = max(float(np.mean(np.linalg.norm(np.diff(k, axis=0), axis=-1))), 1e-9)
    accel = float(np.mean(np.linalg.norm(np.diff(k, n=2, axis=0), axis=-1)))
    jitter = accel / vel

    max_jump_rel = float(diff.max()) / scale

    clips.append(dict(
        fname=fname, group=group, serve=m.group(2), n=len(k),
        joint_tele=joint_tele_rate, spike=spike_rate,
        body_tele=body_tele_rate, jitter=jitter, max_jump=max_jump_rel,
    ))

def robust_z(vals):
    v = np.asarray(vals, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad < 1e-9:  # degenerate: most values identical - fall back to std
        std = v.std()
        if std < 1e-9:
            return np.zeros_like(v)
        return (v - med) / std
    return 0.6745 * (v - med) / mad

# z-scores WITHIN each group (coach naturally moves more/faster)
for g in ["Beginners", "Coach"]:
    idx = [i for i, c in enumerate(clips) if c["group"] == g]
    for key in ["joint_tele", "spike", "body_tele", "jitter", "max_jump"]:
        zs = robust_z([clips[i][key] for i in idx])
        for i, z in zip(idx, zs):
            clips[i][f"z_{key}"] = float(z)

# composite: worst robust-z among chaos indicators (exclude max_jump alone - single spike can be ok)
flagged = []
for c in clips:
    z_indicators = [c["z_joint_tele"], c["z_spike"], c["z_body_tele"], c["z_jitter"], c["z_max_jump"]]
    c["worst_z"] = max(z_indicators)
    c["n_bad_z"] = sum(z >= 3.5 for z in z_indicators)
    # flag: worst_z >= 4 OR at least 3 indicators with z >= 3.5
    if c["worst_z"] >= 4.0 or c["n_bad_z"] >= 3:
        flagged.append(c)

flagged.sort(key=lambda c: -c["worst_z"])

print(f"Scanned {len(clips)} clips; flagged {len(flagged)} as jumbled/outlier\n")
if flagged:
    print(f"{'file':<40} {'grp':<10} {'worstZ':>7} {'#z>=3.5':>7} {'jntTel%':>8} {'spike%':>7} {'bdyTel%':>8} {'jitter':>6} {'maxJ/sc':>8}")
    print("-" * 120)
    for c in flagged:
        print(f"{c['fname']:<40} {c['group']:<10} {c['worst_z']:>7.1f} {c['n_bad_z']:>7} "
              f"{c['joint_tele']:>7.1%} {c['spike']:>6.1%} {c['body_tele']:>7.1%} "
              f"{c['jitter']:>6.2f} {c['max_jump']:>8.2f}")

print("\nFlagged by group:")
for g in ["Beginners", "Coach"]:
    n_sub = sum(1 for c in clips if c["group"] == g)
    n_fl = sum(1 for c in flagged if c["group"] == g)
    print(f"  {g}: {n_fl}/{n_sub} ({n_fl/n_sub:.0%})")

print("\nFlagged by serve:")
for s in ["Drive", "Lob", "Topspin"]:
    n_sub = sum(1 for c in clips if c["serve"] == s)
    n_fl = sum(1 for c in flagged if c["serve"] == s)
    print(f"  {s}: {n_fl}/{n_sub} ({n_fl/n_sub:.0%})")

# Also show population medians for context
print("\nPopulation medians:")
for key in ["joint_tele", "spike", "body_tele", "jitter", "max_jump"]:
    for g in ["Beginners", "Coach"]:
        v = [c[key] for c in clips if c["group"] == g]
        print(f"  {key:<12} {g:<10} median={np.median(v):.3f}")
