# Pickleball Serve — Android Demo

Rule-based feedback is now shown after each completed serve. It uses Coach B
source-linked rules, the final GRU's 0.60 confidence gate and independent
weight-shift/paddle states. Feedback is a research preview pending Coach C
validation. See [rule-based implementation notes](../rule-based/README.md) for
rule coverage, annotated threshold provenance, reproducible checks and limitations.

Thesis proof-of-concept: live camera → BlazePose (MediaPipe) → 5 joint angles → GRU (TFLite) → drive / lob / topspin.

## What it does
- Real-time pose estimation (33 BlazePose landmarks)
- Extracts same 5 joint angles as `scripts/extract_joint_angles.py` (elbow, shoulder, wrist, torso, twist → sin/cos = 10 features/frame)
- Sliding window of 128 frames (matches training `seq_len`)
- TFLite GRU inference — **ensemble** (5 folds, default) or **single** (fold1)
- Minimal overlay: predicted class, confidence, per-class probability bars, pose status, buffer fill

## Requirements
- Android Studio (installed ✓)
- Device/emulator with camera (API 26+)
- Assets (see `app/src/main/assets/README.md`):
  - 6 × `.tflite` + `manifest.json` (already copied)
  - `pose_landmarker_lite.task` (**must download**)

## Build & Run
1. Open `android/` in Android Studio.
2. Download pose model into assets (see assets/README.md).
3. Let Gradle sync (first sync downloads MediaPipe, CameraX, TFLite).
4. Run on device/emulator (back camera).

## Toggle
- **Toggle Mode** button: switches single ↔ ensemble at runtime.
- **Reset Buffer**: clears the 128-frame window.

## Gradle deps (already in app/build.gradle.kts)
- CameraX 1.3.4
- MediaPipe `tasks-vision:0.10.14`
- TensorFlow Lite 2.16.1 + **select-tf-ops** (required — GRU uses TensorListReserve)

## Pipeline parity with training
| Stage | Training (Python) | On-device (Kotlin) |
|---|---|---|
| Pose | BlazePose via extract_keypoints.py | MediaPipe Pose Landmarker |
| Angles | extract_joint_angles.py | JointAngles.kt (same math) |
| Window | pad_or_truncate to 128 | AngleBuffer.kt (trailing zero pad) |
| Model | .keras | .tflite (SELECT_TF_OPS) |
| Ensemble | mean of 5 fold softmax | GruClassifier.kt |

## Known limitations (thesis demo scope)
- Live accuracy ≈ holdout ceiling (~50–60%), not the 86.7% CV number (subject shift).
- Single-person pose only (NumPoses=1).
- Right-handed angle assumption (matches training).
- Portrait orientation locked.

## Ball and landing analysis

### Complete recorded-serve comparison

Use **Replay Serve** to choose an existing MP4 trimmed to one complete serve,
up to 12 seconds (Android 9 or newer). A fresh Lite VIDEO tracker processes the
complete recording using its presentation timestamps. Required-joint visibility
and missing detections follow the same quality and 128-step resampling contract
as the research training pipeline. Long serves retain both endpoints.

The screen compares the old first-128 and rolling last-128 classifiers with the
separate research model under `assets/serve_research/`. That bundle includes its
own training bank, learned fusion weight and probability temperature. It is used
only for recorded replay; the current live modes continue to use the old assets.
If validation found no qualifying confidence threshold, research confidence
acceptance is disabled rather than presenting a fitted threshold as reliable.

**Save comparison** exports the video hash, frame timestamps, landmarks,
visibility, quality, feature window and model probabilities. Check it on the
desktop with:

```powershell
python scripts/compare_serve_replay.py path/to/serve-comparison.json --video path/to/the-same.mp4
```

No physical-phone accuracy or speed has been measured for this path. Kotlin
feature/policy parity uses real training-clip fixtures, including missing poses.
Recorded replay bypasses the live motion gate and does not validate automatic
live serve boundaries. See [the v4 study](../analysis/SERVE_REFINEMENT_V4.md)
for complete model, fusion, calibration and window-comparison results.

### Serve hybrid inference

HYBRID mode averages the five-fold GRU ensemble and the bundled distance-weighted
kNN5 probabilities, 50% each. Both interim and final predictions now use this
combination; final feedback uses the same combined label/confidence displayed by
the UI. Previously, the final path bypassed kNN. Final predictions bypass the
interim EMA smoothing. If kNN is unavailable, the GRU remains the fallback.

Timing features use zero velocity for the first frame, matching the offline
training extractor. The existing 438-sample training bank is unchanged. Python
and Kotlin feature/kNN parity is checked against real clip fixtures by
`HybridPredictionTest`. See `../analysis/HYBRID_CLASSIFIER_AUDIT.md` for recorded
clip accuracy and the distinction from Android's motion-gated live window.

### Landing pipeline

The landing screen targets 30 fps playback with concurrent ball/court analysis on the Redmi Note 12
(Snapdragon 685 / Adreno 610). Timestamp-paced decoding and lightweight tracking have their own worker.
A separate, bounded inference worker uses two-thread XNNPACK, avoiding GPU contention and the several
minutes of initial GPU compilation observed on this device. Initialization, inference, and cleanup
remain on the same inference thread. The optional GPU path still requires matching
`tensorflow-lite-gpu` and `tensorflow-lite-gpu-api` dependencies.

The same trained YOLO checkpoints are exported with INT8 weights and FP32 activations. A 320px
full-frame detector finds/reacquires objects; a 160px crop detector confirms moving balls at the
640px tracking image's pixel scale. Yellow/green motion proposes crops but is never itself treated
as a model detection. Other colors retain the full-frame fallback. Court updates are spaced by
3–5 seconds; robust background optical flow tracks camera motion between them. OpenCV's native
homography solver uses only confident keypoints. Up to three seconds of bounded frame history
aligns delayed detections with the current frame; lost/expired tracks are hidden.

Playback samples sources above 30 fps while preserving presentation timestamps (e.g. 60 → 30 fps).
Analysis pauses when the activity is hidden and the screen stays awake while it is visible. The last
in-flight model result is incorporated before the final landing result. These are model estimates,
not validated line-call measurements; occlusion and small/blurred balls can still cause missed tracks.

Runtime assets: `ball_320_w8a32.tflite`, `ball_roi160_w8a32.tflite`, and `court_320_w8a32.tflite`.
Original 640px assets/checkpoints are retained. Reproduce exports using
`scripts/export_mobile_candidates.py --size 320 --quantize w8a32` and
`scripts/export_mobile_candidates.py --size 160 --quantize w8a32 --ball-only` in the configured WSL environment.
The rejected fully quantized 320px candidates are outside the APK in `outputs/mobile_candidates/`.

Replay bitmaps belong to the decoder and are reused. Preprocessing refreshes its tensor for every
frame, and the preview receives its own snapshot. Rotation is applied once. Decoder output is drained
through its own end-of-stream marker, and leaving the activity queues model cleanup after active work.

Validation tasks: `:app:testDebugUnitTest`, `:app:connectedDebugAndroidTest`, and `:app:lintDebug`.
Phone tests cover mutated bitmap inputs, rotation, decoder ownership/EOS, thread affinity, model
linkage, delayed/lost tracks, and real playback at 720p/30 and 1080p/60. Performance gates include
playback duration, displayed frames, tracking P95 <25 ms, Android render P95 <33.4 ms, court coverage,
and ball positions checked against visible serve reference points. Run them with the phone unlocked.
Connected Gradle tests can uninstall the app at teardown; reinstall the debug APK or run from
Android Studio afterward. Prefer installing both APKs with `adb install -r` then running
`adb shell am instrument -w -r -e class com.thesis.pickleballserve.landing.LandingPipelineTest,com.thesis.pickleballserve.landing.RealtimeTrackerTest com.thesis.pickleballserve.test/androidx.test.runner.AndroidJUnitRunner`.
Test video fixtures live only in `src/androidTest/assets`. See `NOTE12_BENCHMARK.md` for measured results.
