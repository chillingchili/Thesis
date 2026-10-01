# GRU + kNN5 recorded-clip audit — 2026-10-01

The desktop previously used GRU alone. It now defaults to the Android-style
hybrid: five GRU folds averaged, combined equally with distance-weighted kNN5.
The earlier pose audit and v2 retraining scores were GRU-only measurements.
The retrained v2 models remain separate; this hybrid uses the original Android
GRU assets and existing kNN bank, not a mixture with the v2 preprocessing.

## Evaluation and data boundary

No fitting or tuning was performed for this comparison. The bundled 438-sample
kNN bank exactly matches `features/timing_train.npz`, with IDs checked against
the training manifest. All 175 diagnostic test IDs are absent from the bank.
The 50/50 weight and k=5 were inherited from Android, not selected on these tests.
Bank SHA-256: `d36f397acc205608d96f0568a015a681f5580514b6d2d219c8154b1cb4d9d413`.

Inputs are the existing fresh BlazePose Lite pose caches: 210 Coach B annotated
training clips and 175 separate Beginner 3/4 diagnostic test clips. Training
scores below are in-sample checks on this annotated subset, not cross-validation
or a fresh evaluation of all 438 training clips. These 175 clips are not the
independent Coach C 30-clip thesis evaluation, whose annotations do not exist yet.

Use the first 128 valid angle frames, zero-pad shorter windows, and match GRU
channel moments to training statistics. kNN receives 32 timing features from
the original unmatched angle vectors, excluding padding. Its first velocity is
zero, consistent with the offline training extractor.

The batch evaluator uses the original Keras GRU checkpoints; real desktop TFLite
inference was additionally checked on both Beginner 2 Drive examples, with the
same final probabilities. Python/Kotlin feature and kNN parity passes for three
real clip fixtures using the actual bundled bank.

## Results

| Model | Annotated training subset (210) | Separate test (175) | Test macro F1 |
|---|---:|---:|---:|
| GRU ensemble | 64.8% (136) | 51.4% (90) | 0.479 |
| kNN5 | 69.5% (146) | 45.7% (80) | 0.413 |
| GRU + kNN5 | 75.2% (158) | 49.7% (87) | 0.496 |

The hybrid improves the in-sample score, but does not improve overall diagnostic
test accuracy. These results do not support claiming that adding kNN5 solves the
inconsistency. They measure serve classification through the pose pipeline, not
BlazePose landmark accuracy against hand-labeled keypoints.

At the existing 0.60 confidence gate, GRU accepts 130/175 clips (74.3% coverage),
of which 63 are wrong. Hybrid accepts 89/175 (50.9% coverage), of which 41 are
wrong. Accepted accuracy is 51.5% for GRU and 53.9% for hybrid. Counting withheld
predictions as unsuccessful, hybrid succeeds on 48/175 (27.4%). The reduction in
wrong accepted results comes with substantially more withheld feedback.

| Clip | GRU prediction | Hybrid prediction | Hybrid feedback gate |
|---|---|---|---|
| Beginner2_Drive_001 | Lob, 69.1% | Lob, 43.5% | Withhold: below 60% |
| Beginner2_Drive_002 | Lob, 86.3% | Lob, 43.2% | Withhold: below 60% |

Neither example is correctly classified by this hybrid. It reduces confidence
in the wrong label instead of making the coaching agree with the coach.

## Android corrections and scope

Android previously combined GRU and kNN during interim inference but bypassed
kNN for the final result. Final classification and feedback now both use the
combined probabilities; final inference still bypasses interim EMA smoothing.
If the kNN model is unavailable, Android retains its GRU fallback.

Android also computed the first velocity as the first angle minus zero. This
could falsely place peak velocity at the first frame. It now uses zero first
velocity, matching the training extractor. Applying the old velocity formula
to the same recorded windows yields 45.7% hybrid accuracy, versus 49.7% with the
correction. This is a controlled recorded-window comparison, not a measurement
of an installed Android app.

Android captures its latest 128 motion-gated frames; desktop uses the first
128 valid frames of the selected recording. Identical assets, timing features
and fusion do not establish end-to-end live camera parity. Android's changes
require installation of the rebuilt APK; no phone installation was performed.

## Verification and artifacts

- Desktop: 17 tests pass, including hybrid fusion, unmatched kNN inputs, training
  feature parity, padding and feedback behavior. JavaScript syntax check passes.
- Android: 31 unit tests pass, including two new hybrid tests. Debug APK builds;
  lint completes with 0 errors and 97 warnings. No on-device tests were run.
- `outputs/hybrid_audit/metrics.json`: complete confusion matrices and metrics.
- `outputs/hybrid_audit/predictions.csv`: per-clip component and hybrid scores.
- `outputs/hybrid_audit/tflite_smoke.json`: actual desktop inference for both examples.
- Reproduce batch comparison: `python scripts/evaluate_desktop_hybrid.py`.
- Tests: `python -m unittest desktop.test_hybrid desktop.test_desktop` and Gradle
  `:app:testDebugUnitTest :app:assembleDebug :app:lintDebug`.
