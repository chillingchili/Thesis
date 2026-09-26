# Pickleball Serve — Android Demo

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
