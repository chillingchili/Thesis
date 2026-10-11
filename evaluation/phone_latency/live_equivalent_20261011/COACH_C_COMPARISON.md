# Live vs PC model output on the Coach C reserved clips

**Date:** 2026-10-11
**Device:** Xiaomi 23021RAAEG (Note 12), SM6225, Android 14, adb `1fb637d`
**Clips:** C001, C002, C007 — the clips reserved for Coach C (`evaluation/coach_c/`)
**App build:** commit-era `installDebug` of 2026-10-11 02:26, replay harness enabled

---

## 1. Question

The live gate returns ~35 frames per serve, while the PC pipeline feeds the GRU a
128-frame window. Does that shrink the quality of the coaching output, and how does
the live output compare with the frozen PC model output that Coach C is being asked
to validate against?

**Short answer: yes, live is materially weaker.** On C001 the live model never once
crossed the 0.6 confidence threshold in 25 identical replays (max 0.546, median
0.466), and on C002/C007 the live gate could not produce any verdict at all. The PC
pipeline produced a verdict on all three.

---

## 2. What model is actually on the phone

Verified by hash, not assumed. The five GRU weights in the APK are **byte-identical**
to the weights frozen for the Coach C session:

| File | SHA-256 (APK) | Matches `session2_gru/freeze.json` |
|---|---|---|
| `gru_fold1.tflite` | `2481e11793770c8bf4ab7e0d93029cca8f81a8485df95af736b2ac63f7bb8373` | yes |
| `gru_fold2.tflite` | `9f7a5cefe571ce1d81d099fcc4d2165821159b8c0c4cb6c694510792210f20be` | yes |
| `gru_fold3.tflite` | `70555916b6dfeeef2389c3f3604879e8e1182c151e335c7940003de5d63a216a` | yes |
| `gru_fold4.tflite` | `f1d83af6a5dd167f77f63b8ff87b85424955098b9299419faa2fa51c7ba867b5` | yes |
| `gru_fold5.tflite` | `9b511816d9899b4f7000b524d12afcdc7c15385f9b5083472434c9b81f9d0c96` | yes |

So the PC/live comparison below is **the same classifier**, differing only in how the
input window is built. It is not a comparison of two different models.

Note this is **not** `models/serve_v2_fixed` (that model's fold hashes differ, e.g.
`gru_fold1` = `c87c95fb…`, and its manifest has `moment_matching: false`). The phone
runs the legacy ensemble with `moment_matching: true`, matching
`session2_gru/reports/*.json`.

---

## 3. Ground truth

From `evaluation/coach_c/source/researcher_key.csv` (`recording_category`):

| Clip | Source | Ground-truth serve |
|---|---|---|
| C001 | CoachA | **lob** |
| C002 | Beginner4 / Topsin | **topsin** |
| C007 | Beginner3 / Drive | **drive** |

---

## 4. Three-tier comparison

### Tier 1 — PC, Coach C frozen output

`evaluation/coach_c/session2_gru/reports/{C001,C002,C007}.json`
Legacy 5-fold ensemble + moment matching, **first 128 frames** of the clip at ~60 fps.

| Clip | Truth | Drive | Lob | Topsin | Predicted | Conf | Feedback | Frames |
|---|---|---|---|---|---|---|---|---|
| C001 | lob | 0.0066 | 0.4782 | **0.5152** | topsin ✗ | 0.515 | LOW_CONFIDENCE | 128 (91 truncated) |
| C002 | topsin | 0.3107 | 0.0254 | **0.6638** | topsin ✓ | 0.664 | CORRECTIVE | 128 (21 truncated) |
| C007 | drive | 0.1727 | 0.2109 | **0.6164** | topsin ✗ | 0.616 | CORRECTIVE | 128 (32 truncated) |

**1 / 3 correct.** The report itself warns: *"The legacy GRU uses only the first 128
frames. Its window differs from the retrained complete-serve model."* For C001 the
contact frame is 164 — the GRU's 128-frame window **ends before contact**, which is
visible in the near-chance drive/lob split (0.0066 vs 0.4782).

### Tier 2 — Android, offline recorded benchmark (no gate)

`evaluation/phone_latency/note12_optimized_full_20261010/raw.json`
Same weights, **last 128 usable frames** of the full clip, pose at 1920×1080.

| Clip | Truth | Predicted | Conf | Feedback | Usable frames |
|---|---|---|---|---|---|
| C001 | lob | drive ✗ | 0.6251 | POSITIVE | 219 |
| C002 | topsin | drive ✗ | 0.6373 | CORRECTIVE | 149 |
| C007 | drive | topsin ✗ | 0.7019 | CORRECTIVE | 160 |

**0 / 3 correct.**

### Tier 3 — Android, live gate (real-time replay)

`verdict_{C001,C002,C007}.txt` — 2026-10-11, looped real-time replay through the
production `MainActivity` gate (not a benchmark). Pose at 640×480.

| Clip | Truth | Serves | Classified | Predicted | Conf min/med/max | Frames med | ≥0.6 | Feedback |
|---|---|---|---|---|---|---|---|---|
| C001 | lob | 25 | 25 | topsin 22, drive 3 ✗ | 0.402 / 0.466 / 0.546 | 35 | **0 / 25** | LOW_CONFIDENCE |
| C002 | topsin | 43 | **0** | — | — | 21 | 0 | INSUFFICIENT_DATA |
| C007 | drive | 7 | **0** | — | — | 12 | 0 | INSUFFICIENT_DATA |

**0 / 1 correct on the only clip it could classify; 2 of 3 clips produced no verdict.**

Latency held regardless: `end_to_shown` max **401 ms** (C001), 274 ms (C002),
273 ms (C007) — all far inside the 2000 ms target.

---

## 5. Is the live output weaker than the PC output?

Yes, on three independent axes.

**1. Frame budget — 3.7× fewer frames collapse the confidence ceiling.**
PC feeds 128 frames; live feeds a median of 35. On C001 live never once reached the
0.6 feedback threshold in 25 identical replays (best 0.546). The PC pipeline cleared
0.6 on C002 (0.664) and C007 (0.616). Live is not merely noisier — it is structurally
unable to reach the confidence the feedback rules require, so it degrades to
`LOW_CONFIDENCE` "could not identify this serve" text instead of coaching.

**2. Coverage — live could not classify 2 of 3 clips at all.**
C002 and C007 never reached the app's own `finalMinFrames = 32` floor, so the app
correctly refused to guess and returned `INSUFFICIENT_DATA`. The PC pipeline returned
a verdict on all three.

**3. Stability — live is not reproducible.**
C001, replayed identically 25 times, flipped label 22/3 (topsin/drive) with
confidence spanning 0.144. The PC output is deterministic.

---

## 6. Caveats — read before quoting these numbers

**(a) The Android offline tier disagrees with PC on all three clips, and that is not a
port bug.** Same weights, different window convention: the PC legacy report uses the
**first** 128 frames (`truncated_frames` 91/21/32), the Android `AngleBuffer(128, 10)`
keeps the **last** 128. For C001 that is frames 0–127 (ends before contact at 164) vs
frames 91–218 (contains contact). Both are defensible; they are simply not the same
input. Any PC-vs-Android parity claim must pin the window convention first.

**(b) The C002 and C007 live runs are harness-degraded, not representative of a real
camera.** `ServeClipReplay` decodes via `MediaMetadataRetriever.getFrameAtIndex`,
which recreates the `MediaCodec` on every frame (~1 frame/s; 70–82 frames consumed and
1 rendered per decoder instance). Partial decode within the run: C002 75/149 frames,
C007 80/160. The replay then paces those frames loosely, so the gate only accumulated
12–25 (C002) and 7–19 (C007) frames. **Treat C002/C007 live only as evidence that the
gate correctly refuses below 32 frames — not as a quality measurement.**

**(c) C001 is the valid live measurement.** It decoded 110/219 frames, replayed
properly, and the gate accumulated 33–37 frames — matching the independently derived
live floor of 34.2 frames in `live_equivalent.json`. Its 35-frame confidence ceiling
is the number to cite.

**(d) The PC baseline is itself weak, so "weaker than PC" is a low bar.** PC got 1/3
right, and on C001 its window excluded the contact frame entirely. The honest
conclusion is not that the PC model is good and live degraded it, but that **the
legacy 128-frame GRU is marginal on these clips at 128 frames and unusable at 35.**

**(e) Threshold interaction.** `session2_gru/freeze.json` sets `threshold: 0.6`. Live
C001's maximum over 25 attempts was 0.546. The feedback layer's low-confidence
refusal is therefore doing the right thing — but it means a live user of this build
would receive no coaching on any of these three clips.

---

## 7. Implications for the thesis claim

* The **2-second live latency target is met and robust** — `end_to_shown` ≤ 401 ms
  across 75 live serves, ~5× margin. This claim stands.
* The **accuracy claim does not transfer from the PC pipeline to live.** The PC
  evaluation that Coach C is validating was computed on a 128-frame window; live
  delivers ~35. Reporting PC accuracy alongside a live latency figure would conflate
  two different operating points.
* If live accuracy matters, the lever is the **window, not the model**: either extend
  the gate's active window toward the 128-frame training window, or retrain on a
  short-window distribution. Retraining on 35-frame windows is the only fix that
  preserves the current latency budget; widening the gate trades directly against the
  2 s target.

---

## 8. Provenance

| Artifact | Path |
|---|---|
| PC frozen reports | `evaluation/coach_c/session2_gru/reports/{C001,C002,C007}.json` |
| PC freeze (hashes, threshold 0.6) | `evaluation/coach_c/session2_gru/freeze.json` |
| Ground truth | `evaluation/coach_c/source/researcher_key.csv` |
| Export provenance | `evaluation/coach_c/provenance.json` |
| Android offline benchmark | `evaluation/phone_latency/note12_optimized_full_20261010/raw.json` |
| Live verdicts (raw logcat) | `evaluation/phone_latency/live_equivalent_20261011/verdict_{C001,C002,C007}.txt` |
| Live UI dumps | `evaluation/phone_latency/live_equivalent_20261011/verdict_{C001,C002,C007}.xml` |
| Latency derivation | `evaluation/phone_latency/live_equivalent_20261011/live_equivalent.json` |
| APK weight hashes | `android/app/src/main/assets/gru_fold*.tflite` |

One measurement-only code change was made for this run: `MainActivity.kt` now appends
`pred=`/`conf=` to the existing `serve_ended` log line so the per-serve verdict is
recoverable from logcat. No behavioural change.
