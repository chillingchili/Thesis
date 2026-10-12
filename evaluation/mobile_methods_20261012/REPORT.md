# Five mobile optimization experiments — 12 October 2026

## Decision

Enable the tested CPU pose-priority scheduling by default. Keep the original five-fold GRU and 0.60 confidence gate. Leave GPU pose and revised end detection as explicit research options; do not promote either replacement classifier. These experiments improved some component timings and classification availability, but did not establish dependable coaching within the full two-second thesis boundary.

## 1. Pose priority and 2. End detection — actual app replay

The matched control uses the same corrected replay and paddle bookkeeping as each candidate. Each configuration replayed C001, C002 and C007 three times: nine source replays per configuration, not nine independent clips. CPU variant order was rotated by clip. GPU alone was tested afterward; the combined GPU condition was an exploratory follow-up.

| Condition | Classified replays / 9 | Substantive feedback events | Paddle available events | Median / maximum gate-end to UI |
|---|---:|---:|---:|---|
| both | 3/9 | 0 | 1 | 117.0 / 588.0 ms |
| control | 3/9 | 0 | 0 | 284.0 / 403.0 ms |
| end_v2 | 3/9 | 0 | 0 | 273.0 / 409.0 ms |
| gpu_both | 5/9 | 0 | 1 | 192.0 / 386.0 ms |
| gpu_pose | 8/9 | 2 | 1 | 390.0 / 413.0 ms |
| pose_priority | 3/9 | 0 | 2 | 118.0 / 216.0 ms |

Pose priority permits at most one detector job in flight and spaces accepted submissions by at least 350 ms. Cheap wrist bookkeeping remains ordered on the pose thread, so dropping/throttling images does not reorder contact observations. Stale-generation jobs are ignored, and pending jobs clear between serves. The threshold and original detector/model weights are unchanged. The 118 ms median versus 284 ms control is a timing benefit in these repeated trials; classification availability remained 3/9, and no substantive feedback appeared. It is not a demonstrated accuracy improvement.

Revised end detection allows settling below 20% of the observed motion peak only after at least 300 ms from that peak, still requiring the original low-motion hold. It did not increase classification availability. Combining it with GPU and pose priority reduced classified replays to 5/9 versus 8/9 for GPU alone; one of nine combined replays produced no end event. Missing events are retained in the denominator. This end policy remains disabled by default.

## 3. Mobile-rate training — participant-separated pilot

Only the 209 quality-accepted training clips were used. Three outer folds held out Beginner1, Beginner2 and CoachA in turn. No Coach C responses, frozen feedback or diagnostic outcomes were used for fitting or selecting epochs. Mobile variants sample cached full-resolution VIDEO-mode poses at 6, 15 and 30 fps, retaining real observations and zero padding; these simulate sampling frequency, not the actual live pose extractor, motion gate, resolution or frame-drop process. All variants from a source clip stay in that clip’s training partition.

Models use a fixed 30-epoch budget. The full-window control repeats its original input once per rate variant to match the mobile model’s optimizer-step count and initialization seed. Primary comparisons below use the middle tested rate, 15 fps; all three rates are saved. These are exploratory participant-held-out development scores, not a new independent final test and not directly comparable to legacy 86.7% stratified CV or the old 175-clip diagnostic score.

| Model | Correct / 209 | Accuracy at simulated 15 fps | Accepted at 0.60 and ≥32 real frames | Correct among accepted |
|---|---:|---:|---:|---:|
| full_window_control32_matched | 50/209 | 23.9% | 192 | 47 |
| mobile32 | 79/209 | 37.8% | 184 | 72 |
| mobile_ensemble2 | 99/209 | 47.4% | 162 | 77 |
| distilled16 | 86/209 | 41.1% | 170 | 70 |
| hard_label16 | 108/209 | 51.7% | 144 | 84 |

Matching input sampling improved the 32-unit single model from 23.9% to 37.8% in this pilot. However, confident errors remain common: the mobile32 candidate accepted 184 cases, of which only 72 matched the stored labels. These uncalibrated probabilities do not justify deployment or lowering the confidence gate.

## 4. Pose acceleration — component and app results

CPU and GPU each processed the same nine sampled frames with one warm-up pass and three measured passes (27 measurements per backend), excluding decoding and model initialization. Both found poses in every measured case. Median pose time was 80.69 ms on CPU and 55.24 ms on GPU, about 31.5% lower. Maximum observed times were 137.78 and 60.48 ms. CPU ran before GPU; this is a small sequential component comparison, not a thermal-controlled causal estimate.

Landmarks were not identical. Median normalized XY disagreement was 0.00365 and maximum 0.10721 across joints; among classifier joints the maximum was 0.07628. Disagreement is not ground-truth pose error, but it prevents assuming output parity.

In actual-app replay, GPU alone raised classified source replays from 3/9 to 8/9 and produced two substantive outputs. Those outputs were C001 classified as Topspin with positive feedback and C007 classified as Topspin with a corrective cue; both classifications disagree with their recording categories (Lob and Drive). Recording categories are not independent Coach C subtype ground truth, and no claim of cue validity follows. GPU therefore remains experimental despite its speed and availability gains.

## 5. Distillation — quality and device latency

Each outer-fold student learned from two 32-unit mobile-trained teachers fitted only to the other participants. Targets were 50% stored hard label and 50% average teacher probabilities (temperature 1). The student has 16 GRU units. This distills a newly trained mobile teacher ensemble, not the legacy frozen five-fold ensemble; the latter was not used as an unbiased held-out-participant teacher because it has prior training-participant exposure.

The distilled student reached 41.1% at simulated 15 fps, below its two-teacher ensemble (47.4%) and a same-size 16-unit hard-label control (51.7%). Full-training exports were staged in a separate phone research directory, never copied over app assets.

| Isolated candidate | File size | Median inference | Maximum inference |
|---|---:|---:|---:|
| mobile32 | 32012 bytes | 16.61 ms | 18.98 ms |
| distilled16 | 20236 bytes | 16.19 ms | 23.02 ms |

Each model had one excluded warm-up and 30 measured inferences on the same fixed tensor, with two CPU threads. Phone probabilities matched Keras within 1e-4. The distilled model saved about 37% in file size but only 0.42 ms in median inference, with a higher observed maximum. That does not justify its lower accuracy. Neither candidate was promoted.

## What the two-second results do and do not establish

All 53 observed end-event UI updates across 54 source replays occurred within 588 ms of the app’s end detector firing. One combined-GPU replay had no end event. Fast missing/low-confidence notices are not successful coaching.

The corrected harness logs a source-clip endpoint separately and supplies repeated idle images before/after playback. Those repeated images are marked synthetic and excluded from classification, hip and paddle evidence; they only keep the end detector receiving observations. Exact timestamp matching prevents associating a pose with another frame or its synthetic flag. Decoding uses contiguous batches and is excluded from the live replay interval.

GPU alone had two UI updates 3.566 and 3.591 seconds after the recorded source ended. Several CPU/combined outputs occurred before the file ended. A file endpoint and the estimated motion peak are not independently marked physical serve completion; earlier output may mean either legitimate early completion or incomplete capture. The UI log also timestamps label updates, not externally observed rendered pixels. Therefore these trials neither certify the full physical-capture-to-display two-second requirement nor prove reliable coach replacement.

## Installed state and validation

The final installed app keeps CPU pose, the original five-fold GRU, pre-roll and the existing end detector; pose priority is enabled. GPU and revised end detection require explicit research intent flags. The new trained models remain isolated files for research. Frozen Coach C reports and submitted answers were not changed.

Android JVM tests, debug builds and lint passed; CPU/GPU and candidate-model device tests passed. Exact APK/model hashes, protocol files, raw logs, input fixtures, held-out predictions and scoring scripts are retained. See `phone_summary.json`, `training_summary.json`, `model_latency_summary.json`, `pose-cpu-gpu.json`, `phone/`, `training_v2/` and `phone_tested_models/`.

The initial timestamp-unit pilot was aborted and excluded (`training/ABORTED.md`). The initial unmatched-exposure training control is preserved separately and superseded by the matched control in final results. No failed or missing phone event was silently scored as a successful classification.

## Recommended next step

Collect timestamped live-phone pose sequences with independently marked serve endpoints, then train/evaluate the established capture representation using participant-separated data. GPU deserves that follow-up because it increases acquisition capacity, but requires coordinate/output validation. The present experiments show that replacing the GRU with a tiny model is not the main solution.
