# Capture, augmented geometry and calibrated fusion — v4

Completed automated work requested in items 1, 3 and 5. Coach review and new-player collection remain deferred. All studies use existing recordings. No coach judgments or new independent evaluation are claimed.

Training-only selection chose **Lean arm/torso, control**, based on mean macro-F1 across held-out players for the nested tuned hybrid. The selected research model is available in Android recorded replay; the existing live GRU/kNN models remain the default.

## 1. Match capture and training

Android now includes **Replay Serve → Choose serve video → Save comparison**. It processes one trimmed complete serve (maximum 12 seconds, Android 9+) with a fresh MediaPipe Lite VIDEO tracker. It decodes indexed video frames and pairs them with sorted presentation timestamps from the video track. Unsupported duplicate timestamps or disagreement between track sample count and decoded frame count are rejected rather than guessed. Missing detections retain their timeline slots. Required-joint visibility is checked at 0.5, with at least 32 usable source frames and 80% usable source/output frames. The complete timeline is resampled to 128 steps; angle pairs are renormalized after interpolation.

The saved JSON includes source-video SHA-256, frame timestamps, points, visibility, angle window, quality and predictions. The old first-128 and last-128 paths and the separate complete-serve research path are shown on the same recording. Research GRU inputs receive no moment matching; its kNN uses timing features from the resampled angles. The new policy is associated only with its new model/bank, never applied to the legacy assets.

**No physical Android is connected.** The app compiles and the numerical feature contract is verified with real training-clip and missing-pose fixtures. Android/Python pose detection, decoder rotation, native inference performance and real LIVE_STREAM callback behavior still require the phone trace. Recorded VIDEO replay deliberately bypasses the live motion gate; it does not establish that automatic live serve boundaries are fixed.

Offline evaluation of the unchanged bundled classifier illustrates window sensitivity:

| Existing assets / window | GRU | Fixed hybrid | Clips losing poses | Discarded poses |
|---|---:|---:|---:|---:|
| first128 | 51.4% | 49.7% | 142 | 3789 |
| last128 | 40.6% | 50.3% | 142 | 3789 |
| full_timeline | 48.0% | 47.4% | 0 | 0 |

The old models were trained under their legacy input rules. Resampling alone is not assumed to improve their accuracy. The new full-serve pipeline is paired with models trained under that pipeline. Source-pose counts exclude failed detections in the legacy windows, while complete-timeline quality counts missing poses.

## 3. Augmentation with richer features

Reused all 418 frozen training-only video augmentation variants, including their newly extracted landmarks. No raw recording is changed. Every variant follows its original clip into a training fold; validation uses originals only. Each model receives one sampled original/variant per original per epoch, for 100 epochs with the same GRU32 training settings as v3. The angle baseline reuses its three verified v3 participant-held-out models.

The lean representation contains the existing ten angle channels plus aspect-corrected, hip-centered coordinates, visibility and adjacent displacements for shoulders, right elbow/wrist/index, and hips (45 channels). Full skeleton and full skeleton/motion use 99 and 165 channels. Lean control versus lean augmented isolates augmentation at fixed input size; cross-representation comparisons also change model capacity.

| Condition | Held-out-player GRU | Fixed hybrid | Nested tuned hybrid | Mean-player tuned macro-F1 |
|---|---:|---:|---:|---:|
| Angles + augmentation | 47.8% | 57.9% | 53.1% | 0.481 |
| Lean arm/torso, control | 35.9% | 53.6% | 56.9% | 0.522 |
| Lean arm/torso + augmentation | 38.3% | 54.5% | 53.6% | 0.489 |
| Full skeleton + augmentation | 33.5% | 52.6% | 55.5% | 0.509 |
| Full skeleton/motion + augmentation | 32.5% | 50.7% | 53.6% | 0.489 |

## 5. Fusion weights and confidence

For each outer held-out player, two inner folds hold out one remaining player and train on the other. Their group-held-out predictions fit the fusion policy. The outer player contributes no original/augmented training clip, label, calibration example or weight-selection example. The fixed grid tests GRU weights 0, 0.25, 0.5, 0.75 and 1. It selects mean-player macro-F1, with probability loss and proximity to 0.5 as declared tie breakers. A single temperature on the log fused probabilities minimizes inner player-balanced log loss. Temperature changes probability sharpness while preserving the fused class ranking.

Acceptance thresholds are tested only on those inner predictions: at least 80% observed accuracy, at least 20 accepted cases and at least five per inner player. If none qualifies, confidence acceptance is disabled. These small-sample thresholds are development criteria, not a guarantee of correctness.

| Condition | Fixed hybrid log loss | Nested calibrated log loss | Fixed ECE | Nested ECE | Nested acceptance coverage | Accepted accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Angles + augmentation | 1.430 | 1.093 | 0.162 | 0.114 | 19.1% | 35.0% |
| Lean arm/torso, control | 2.233 | 1.100 | 0.333 | 0.264 | 18.7% | 25.6% |
| Lean arm/torso + augmentation | 2.108 | 1.010 | 0.325 | 0.089 | 18.7% | 25.6% |
| Full skeleton + augmentation | 2.118 | 1.034 | 0.347 | 0.159 | 18.7% | 25.6% |
| Full skeleton/motion + augmentation | 1.793 | 1.010 | 0.294 | 0.089 | 18.7% | 25.6% |

The final research policy is fit on all participant-held-out training predictions after nested evaluation. Inner-fold acceptance thresholds did not transfer successfully: outer accepted-case accuracy was only 25.6–35.0%, despite meeting 80% in the inner fitting data. These thresholds are unsuitable as validated coaching-confidence filters. All final policies disable confidence acceptance. It accompanies a single model refit on all 209 eligible clips. The shift from one/two-player base fits to three-player final fits can affect calibration. Policies trained here cannot be transplanted onto the old five-model ensemble.

Selected policy: GRU weight **0.25**, kNN weight **0.75**, temperature **9.244**, acceptance threshold **None** (None means disabled).

## Diagnostic evaluation after selection

All candidates, policies and checkpoint hashes were frozen before this run opened diagnostic labels. These 175 clips were inspected in earlier work; they remain diagnostic, not an untouched final test. This table evaluates single full-training fits, whereas v3 diagnostic results used five-fold ensembles. One tracking-quality rejection remains incorrect/unavailable in the denominator.

| Condition / mode | Accuracy | Macro-F1 | Drive recall | Lob recall | Topspin recall |
|---|---:|---:|---:|---:|---:|
| Angles + augmentation / gru | 49.7% | 0.477 | 70.5% | 22.4% | 55.4% |
| Angles + augmentation / fixed_hybrid | 46.3% | 0.426 | 85.2% | 20.7% | 30.4% |
| Angles + augmentation / tuned_calibrated | 42.9% | 0.374 | 86.9% | 15.5% | 23.2% |
| Lean arm/torso, control / gru | 45.7% | 0.418 | 78.7% | 41.4% | 14.3% |
| Lean arm/torso, control / fixed_hybrid | 40.6% | 0.328 | 88.5% | 22.4% | 7.1% |
| Lean arm/torso, control / tuned_calibrated | 44.0% | 0.377 | 91.8% | 17.2% | 19.6% |
| Lean arm/torso + augmentation / gru | 45.7% | 0.451 | 65.6% | 31.0% | 39.3% |
| Lean arm/torso + augmentation / fixed_hybrid | 41.7% | 0.372 | 80.3% | 17.2% | 25.0% |
| Lean arm/torso + augmentation / tuned_calibrated | 42.9% | 0.375 | 86.9% | 17.2% | 21.4% |
| Full skeleton + augmentation / gru | 56.6% | 0.494 | 93.4% | 67.2% | 5.4% |
| Full skeleton + augmentation / fixed_hybrid | 46.9% | 0.391 | 95.1% | 36.2% | 5.4% |
| Full skeleton + augmentation / tuned_calibrated | 42.9% | 0.348 | 95.1% | 17.2% | 12.5% |
| Full skeleton/motion + augmentation / gru | 52.0% | 0.434 | 90.2% | 62.1% | 0.0% |
| Full skeleton/motion + augmentation / fixed_hybrid | 41.7% | 0.319 | 93.4% | 27.6% | 0.0% |
| Full skeleton/motion + augmentation / tuned_calibrated | 41.1% | 0.325 | 93.4% | 17.2% | 8.9% |

## Interpretation and remaining work

- Three training players remain the entire independent training population. Nested inner models train on one player; their predictions can be substantially worse than models trained on two or three. Reported metrics are small-sample development evidence.
- Each condition has properly separated outer evaluation of its fusion policy. Selecting the best condition on these outer scores can inflate the winner’s apparent performance. A separate new-player final test remains unavailable.
- Pose geometry cannot directly measure ball spin. Per-class recall and macro-F1 accompany accuracy so a collapsed serve class is visible. No source label was changed without coach review.
- A better calibration score need not imply better recognition. Log loss/Brier combine discrimination and probability quality; reliability bins and ECE are also saved. Temperature alone preserves class predictions.
- Real phone replay and live motion-gate validation remain pending. No connected-device accuracy or speed is claimed.

## Outputs and reproduction

Verification: **20 Python checks and 35 Kotlin unit tests pass**. Android debug build, lint and the phone-test APK build pass. The selected TFLite model matches Keras across all 175 diagnostic inputs within the declared numerical tolerance. Research model/bank hashes inside the APK were checked. Physical-phone instrumentation tests remain pending.

- [Frozen protocol](../models/serve_refinement_v4/protocol.json)
- [Frozen selection and policies](../models/serve_refinement_v4/frozen_selection.json)
- [Diagnostic results](../outputs/serve_refinement_v4/diagnostic_results.json)
- [Per-clip predictions](../outputs/serve_refinement_v4/diagnostic_predictions.csv)
- [Research mobile bundle](../models/serve_refinement_v4/android_bundle/manifest.json)
- [Complete automated-study package](../outputs/serve_refinement_v4/complete_outputs.zip)
- [Android debug APK](../android/app/build/outputs/apk/debug/app-debug.apk)

```powershell
python scripts/train_serve_refinement.py
python scripts/evaluate_serve_refinement.py
python scripts/test_serve_refinement.py
python scripts/test_serve_refinement_integrity.py
python scripts/report_serve_refinement.py
# After Replay Serve → Save comparison on Android:
python scripts/compare_serve_replay.py path/to/serve-comparison.json --video path/to/the-same.mp4
```

A connected-phone instrumentation test, `com.thesis.pickleballserve.ServeResearchModelTest`, checks the actual Android TFLite runtime, kNN bank and calibrated probabilities against three training-only numerical fixtures. It has not been run without a device. The replay trace comparison separately checks decoded-video/pose behavior.

References: [independent calibration data](https://scikit-learn.org/stable/modules/calibration.html), [MediaPipe Android modes](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/android), [indexed video frames](https://developer.android.com/reference/android/media/MediaMetadataRetriever), [video timestamps](https://developer.android.com/reference/android/media/MediaExtractor).
