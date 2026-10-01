# Four follow-up investigations — 2026-10-01

Completed model experiments and a local Android-classifier benchmark. The coach
review packet is prepared; human judgments remain pending. No source label,
original model, or Android model asset was replaced.

Thesis video augmentation improved held-out-player GRU accuracy from
**36.4% to 47.8%**
(+11.5 percentage points)
against the matched control. Diagnostic accuracy changed from
**46.3% to 47.4%**
(+1.1 points).

Skeleton and skeleton-motion inputs reached
**55.4% and 56.0%**
on the diagnostic clips, while their held-out-player development scores were
lower than the angle control. Model ranking therefore varies across these
players and evaluation procedures. The fixed research hybrids scored lower
than their corresponding GRUs on the diagnostic set. These one-seed results
show improvements in specific comparisons, without establishing a consistently
reliable model or a single cause for the remaining errors.

## 1. Thesis video augmentation

The pipeline now applies brightness/contrast, independent per-pixel Gaussian
noise (3–5% of the 0–255 intensity range), slight isotropic crop/translation, and
2–10% temporal subsampling **before** fresh BlazePose Lite VIDEO extraction.
No flips, shears, or nonuniform stretching are used. Parameters are fixed within
each clip; Gaussian noise varies by frame. Original raw videos are untouched.

Two deterministic composite variants were extracted for each of the 209
eligible original training clips: **418 variants**, of which
**418 passed** the existing tracking gate. Rejected variants
fall back to their original clip during sampling. This produces additional
training views, not additional independent participants. The original 210-clip
inventory retains the one pre-existing tracking rejection.

Visual spot checks were rendered for each training player. Across all variants,
the median of per-clip mean angular changes was
3.12 degrees on jointly valid frames;
0 variants were identical to their original
feature window. These changes include tracker sensitivity and temporal
resampling, and are not new independent serve mechanics.
[Example original/augmented frames and skeletons](../outputs/serve_study_v3/previews/Beginner2_Drive_001.jpg).

Each condition sees exactly one example per original training clip per epoch,
for 100 epochs, batch size 16, GRU 32, dropout 0.15, L2 0.0001 and Adam 0.001.
The augmented condition samples original/variant 1/variant 2 uniformly; other
conditions see the original. Batch order and model seed match where dimensions
permit. This controls optimizer-update count. No early stopping or validation
feedback is used during these fits. No feature-level augmentation is mixed into
the new ablation, isolating the new video-augmentation treatment.

Original IDs are split first. All variants inherit their original's membership.
Validation sees unaugmented originals only. The primary development comparison
holds out one entire training player; the secondary five-fold split uses the
same original memberships as v2. Every fold saves IDs, hashes and histories.

## 2. Richer input comparison

| Input condition | Stratified development CV | Held-out-player CV | Diagnostic test |
|---|---:|---:|---:|
| Five angles, no augmentation | 87.6% | 36.4% | 46.3% |
| Five angles, thesis video augmentation | 85.6% | 47.8% | 47.4% |
| 33-joint coordinates + visibility | 91.9% | 35.4% | 55.4% |
| 33-joint coordinates + motion + visibility | 91.9% | 32.1% | 56.0% |

The skeleton representation uses 33 image-space joints, with x corrected for
image aspect ratio, per-frame hip centering, median visible torso-length scaling,
and a visibility mask per joint (99 channels). The motion condition appends 66
adjacent-step displacements (165 channels). It preserves all 128 full-serve
timeline samples. Missing joints are zeroed; neither interpolation nor motion
differences bridge missing observations. The model does not receive player IDs.
These displacements describe movement per normalized serve step, not measured
physical velocity. Richer inputs also increase parameter count, so this compares
input systems rather than proving an information-only causal effect. Aspect-ratio
correction also differs from the legacy normalized-image angle representation.
The five-angle GRUs have 4,323 parameters; the skeleton
and skeleton-motion GRUs have 12,867 and
19,203, respectively.

| Input condition | Beginner 1 held out | Beginner 2 held out | Coach A held out |
|---|---:|---:|---:|
| Five angles, no augmentation | 23.2% | 55.1% | 31.0% |
| Five angles, thesis video augmentation | 37.7% | 73.9% | 32.4% |
| 33-joint coordinates + visibility | 34.8% | 37.7% | 33.8% |
| 33-joint coordinates + motion + visibility | 34.8% | 30.4% | 31.0% |

Training-only selection chose **Five angles, thesis video augmentation**, using the
predeclared mean accuracy across held-out players. The choice and all 32 model
hashes were frozen before opening the diagnostic test arrays. Each diagnostic
GRU result averages that condition's five stratified-fold models, all trained
on training clips only. The existing 175 Beginner 3/4 clips were previously
inspected and are not an untouched final evaluation. One diagnostic quality
failure remains in the denominator as unavailable/incorrect.

![Accuracy comparisons](../outputs/serve_study_v3/comparison.png)

## 3. GRU / kNN / hybrid

The deployed bank contains 438 older training examples. Its contents were
verified against the saved training-feature source and checked for test-ID
overlap. The local benchmark invokes the actual five Android TFLite files and
bundled kNN bank. GRU gets moment-matched features; kNN gets unmatched timing
features; fusion remains the app's fixed 50/50 average.

| Bundled-model input window | Available hybrid results | GRU | kNN5 | Hybrid |
|---|---:|---:|---:|---:|
| first128_recorded_window | 175/175 | 51.4% | 45.7% | 49.7% |
| last128_recorded_window | 175/175 | 40.6% | 47.4% | 50.3% |
| android_gate_replay | 22/175 | 4.0% | 4.6% | 4.0% |

The primary window is the first 128 valid frames of each recorded clip, matching
the desktop comparison. The last-128 window probes the phone's rolling buffer.
Gate replay ports the current cold-start MainActivity gate, at source-video
cadence with every available pose, and selects the last completed event. It does
not force an event to finish at EOF. Short/no completed events count as
unavailable. Gate replay is **not physical Android accuracy**: LIVE_STREAM frame
drops, callback timing, continuous tracker state and live warm-up are absent.
Its result demonstrates sensitivity to segmentation assumptions, not a measured
failure rate for the connected phone. No device was connected.

The existing Kotlin timing/kNN fixture tests, Android build and lint checks pass.
The desktop first-window benchmark is independently reproduced with TFLite.
The earlier 56.1% kNN figure used a different cached-feature evaluation and
per-player averaging; it is not a measured score for the phone's fixed hybrid.

For a fair new-training comparison, kNN was refitted inside each same training
fold using the current full-serve 32 timing features, with no scaling or weight
tuning. It never uses the held-out player's clips. These are new research hybrids,
not the existing 438-example Android bank:

| Input condition | Held-out-player GRU | Fold-fitted kNN | Fixed hybrid |
|---|---:|---:|---:|
| Five angles, no augmentation | 36.4% | 53.6% | 53.1% |
| Five angles, thesis video augmentation | 47.8% | 53.6% | 57.9% |
| 33-joint coordinates + visibility | 35.4% | 53.6% | 53.1% |
| 33-joint coordinates + motion + visibility | 32.1% | 53.6% | 50.2% |

| Input condition | Diagnostic GRU | 209-example kNN | Fixed hybrid |
|---|---:|---:|---:|
| Five angles, no augmentation | 46.3% | 42.9% | 44.6% |
| Five angles, thesis video augmentation | 47.4% | 42.9% | 43.4% |
| 33-joint coordinates + visibility | 55.4% | 42.9% | 49.1% |
| 33-joint coordinates + motion + visibility | 56.0% | 42.9% | 46.3% |

Hybrid weights and confidence threshold (0.60 for reporting) were not selected
using diagnostic labels. Confidence coverage, wrong confident predictions,
macro-F1, per-player results and confusion matrices are saved with every result.

## 4. Coach and pose audit

Audited all 210 training annotations against the frozen
inventory: **0 label-file disagreements** and
**1 tracking-quality rejection**. Agreement between two
files does not establish that the visible executed serve supports the label.

Prepared **30 blinded review cases**, balancing player/class
cells with two confident participant-held-out errors and one seeded comparison
per cell, then adding the quality-rejected clip and named regression examples.
Only training clips are included. The coach first judges the renamed raw videos,
then reviews six raw/skeleton frame pairs per clip. The response sheet allows
ambiguous/unjudgeable labels and specific wrist/hand, occlusion and identity
errors. The researcher key, original labels, model predictions and previous
coach comments are outside the shared ZIP.

**Human review remains pending; no label was corrected automatically.** Blank
responses produce 30 pending judgments, not a fabricated agreement score.
The scoring script validates IDs, reviewer/date and response values and writes
agreement/disagreement findings separately from the original dataset. Because
the packet deliberately samples difficult cases, its agreement rate cannot be
reported as dataset-wide or final-evaluation accuracy.

[Coach packet](../outputs/serve_study_v3/coach_review/coach_review_packet.zip) ·
[Response sheet](../outputs/serve_study_v3/coach_review/packet/responses.csv) ·
[Researcher inventory](../outputs/serve_study_v3/coach_review/training_audit.csv)

## Interpretation limits

- These are development experiments with one seed and only three training
  participants. They do not establish a hard accuracy ceiling or prove that data
  quantity, GRU architecture, or a specific feature is the sole bottleneck.
- Player and recording conditions are partly confounded: the thesis records
  Coach A at a different venue from the beginners. A held-out-player gap cannot
  be attributed exclusively to personal serving style.
- The matched control is a new fit with a controlled sampler, not a rerun of the
  older 438-example GRU. Compare augmentation against this matched control;
  compare deployed assets separately.
- Uniform full-clip resampling can remove much of a global playback-rate change.
  Temporal subsampling here primarily changes which frames reach the tracker
  and interpolation. Its benefit is not isolated from the other augmentations.
- No paddle/ball classifier branch was trained: the richer-input experiment
  measures skeletal information already available with reliable provenance.
- Coach C's independent evaluation and a physical-phone replay/live check remain
  unavailable. No candidate was promoted into the desktop default or Android.

## Reproduce and inspect

```powershell
python scripts/test_serve_study.py
python scripts/test_serve_study_integrity.py
python scripts/prepare_serve_study.py --workers 6
python scripts/train_serve_study.py
python scripts/evaluate_serve_study.py
python scripts/benchmark_serve_hybrid.py
python scripts/prepare_coach_review.py
python scripts/score_coach_review.py outputs/serve_study_v3/coach_review/packet/responses.csv
python scripts/report_serve_study.py
```

[Frozen protocol](../data/serve_study_v3/protocol.json) ·
[Frozen selection and model hashes](../models/serve_study_v3/frozen_selection.json) ·
[Diagnostic evaluation](../models/serve_study_v3/diagnostic_evaluation.json) ·
[Android asset benchmark](../outputs/serve_study_v3/hybrid/results.json) ·
[Per-clip diagnostic predictions](../models/serve_study_v3/diagnostic_predictions.csv)

Implementation references: [MediaPipe VIDEO pose API](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
and [scikit-learn split/preprocessing isolation](https://scikit-learn.org/stable/modules/cross_validation.html).
