# Serve Lab desktop tester

Double-click **launchers/Serve Lab.bat** in the project root. A local Python server opens
the interface at http://127.0.0.1:7861. Keep its console open while analyzing;
Ctrl+C closes the server. This is a local browser application, not a packaged
Windows executable. It does not send your videos to a cloud service.

1. Drag a video into the upload area (or click to browse).
2. Use one complete, right-handed, side-view serve. Optionally set start/end
   seconds to select the motion from a longer video. The selected range must
   be at most 120 seconds; each upload is limited to 250 MB.
3. The default is **GRU + kNN5 hybrid**, using the Android five-fold GRU ensemble
   and bundled kNN5 bank with equal probability weights. Single GRU and ensemble
   remain available. Training-statistic matching is enabled for GRU by default;
   kNN always receives the original, unmatched angle features.
4. For ball landing, open **Analysis settings**. Choose a second-camera video
   of the same serve (analyzed in full, so trim it beforehand), or choose the
   experimental same-view option when court lines and bounce are visible.
   Landing is off by default because the thesis requires a secondary camera.
5. Click **Analyze serve**. View subtype probabilities, weight-shift verdict,
   paddle angle/reference band, optional landing zone, and fixed coaching cues.
6. Inspect the contact image and ball/court replay. Download the JSON report
   for the input settings, measurements, rule IDs, missing signals, asset hashes
   and processing time.

Files stay in `outputs/desktop/<run-id>/`: source uploads, compatible MP4 preview,
keypoints, contact image where available, optional landing replay, and report.
They remain until you remove them. Original source videos are never modified.
Only one analysis runs at a time. Cancellation finishes the current model
operation; the existing ball/court stage finishes before cancellation takes effect.

## Environment

The current workspace's Python 3.10 environment already has the required
libraries and model assets. For another machine, install the dependencies into
a virtual environment and copy the model/config files listed below:

```powershell
python -m pip install -r desktop/requirements.txt
python -m desktop.app --open
```

Optional port override: `python -m desktop.app --port 7862 --open`.
The app binds only to loopback. It serves only named result artifacts rather
than arbitrary filesystem paths. FFmpeg (or imageio-ffmpeg) supplies compatible
H.264 playback for MOV/AVI/MKV and annotated output videos. Playback previews
are silent; source uploads preserve original audio.

Required assets:

- Android assets: `pose_landmarker_lite.task`, `manifest.json`, `gru_single.tflite`,
  `gru_fold1.tflite` through `gru_fold5.tflite`, `shift_config.json`,
  `paddle_config.json`, `feedback_rules.json`, `knn_meta.json`, `knn_train.bin`.
- Paddle: `runs/detect/paddle_ft/weights/best.pt`.
- Landing: `runs/detect/ball_yolo26s/weights/best.pt` and
  `runs/pose/court_ft/weights/best.pt`.

The interface checks asset presence; model initialization errors are reported
per stream during analysis. It never silently downloads substitute weights.

## Relationship to Android and the thesis

The tester reuses the Android TFLite GRUs, pose task, threshold configs and
hand-authored rule JSON. The Python feedback evaluator follows the Android
confidence gate, rule order, subtype restrictions, missing-signal notifications,
and scoped positive confirmation. Directional paddle cues stay disabled while
`direction_validated` is false. Coach C validation is still pending.

This is the **recorded-clip** path. Select one complete serve, from setup through
follow-through. The original classifiers remain available as baselines. Their
first-128-frame window and optional moment matching differ from the old training
preprocessing, as documented in `analysis/POSE_CLASSIFIER_AUDIT.md`.

Hybrid mode uses the first 128 valid pose frames (at least 32 required), then zero
pads shorter sequences. Android live inference uses its latest 128 motion-gated
frames. The model assets, 32 timing features and 50/50 fusion agree; these different
capture windows mean a desktop clip need not reproduce a live phone prediction.
The existing kNN bank contains 438 training samples; no testing samples are added.
Its first velocity is zero, matching the training extractor. The report includes
both component probability distributions and the bank hash. The combined label
and confidence control feedback, including the existing 0.60 confidence gate.
This configuration supports comparison with Android; it is not proven more
accurate. See `analysis/HYBRID_CLASSIFIER_AUDIT.md` for the measured results.

After exporting `models/serve_v2_fixed/tflite`, Analysis settings also offers the
retrained research candidate. It uses exactly the same `scripts/serve_sequence.py`
extractor and input builder as the new training pipeline: fresh BlazePose Lite
tracker per clip, landmark visibility checks, and the complete selected timeline
resampled into 128 angle vectors. It does not use moment matching. Select one
serve of at most 12 seconds. The model contract and pose asset must match the
exported manifest; mismatches fail explicitly.

The candidate displays predictions for inspection but withholds coaching if its
training-only cross-validation score misses the thesis's 85% target. Its models
are separate from the Android assets and original baseline. Android's live
motion gate has not been validated for the recorded-clip candidate. No filename
or annotation label influences prediction.

Weight shift uses all usable frames in the selected interval. Paddle uses only
the estimated contact ±5 frames, through the existing Python contour extractor.
Desktop YOLO uses the original `.pt` weights, while Android uses quantized
TFLite detectors, so numerical parity is not asserted. The body/paddle streams
remain usable if the optional landing stream fails.

Landing reuses `scripts/balltrack_pipeline.py`, including its existing bounce
heuristic, homography and zone classifier. These are model estimates, not
validated line calls. The inherited zone classifier does not know the intended
diagonal receiving service box. Same-view landing is explicitly experimental;
neither landing mode claims to diagnose the biomechanical cause of a fault.
For same-view analysis, candidate bounces must occur at or after the estimated
contact frame when available. A separate-camera clip has its own time origin,
so it uses the original bounce search; no synchronization offset is inferred.

The JSON report preserves research metadata. This tool does not train on the
evaluation partition or mark feedback as coach-validated. Desktop runtime is
not evidence for the thesis's Android two-second latency target.

## Verification

```powershell
python -m unittest desktop.test_desktop desktop.test_hybrid -v
```

Tests cover file/range validation, queue/cancel behavior, local-origin and file
boundaries, missing pose handling, zero masks/padding/truncation, confidence
gating, combined rules, subtype restrictions and optional landing states.

Implementation references: [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
and [MediaPipe pose task options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions).
