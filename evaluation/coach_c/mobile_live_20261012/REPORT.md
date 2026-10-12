# Coach C: 30 clips through the live Android pipeline

Date: 12 October 2026. Device: Xiaomi 23021RAAEG. Original five-fold GRU; CPU pose; pose priority enabled. All 30 source files and original model weights verified by SHA-256. One replay per clip; no tuning during this run.

## Findings

| Measure | Result |
|---|---:|
| Completed source replays | 30/30 |
| Clips with completed displayed feedback (including notices) | 29/30 |
| Substantive coaching outputs | 2/30 |
| Frozen Session 2 substantive outputs | 21/30 |
| Any GRU classification, including low confidence | 6/30 |
| Classifications meeting 0.60 confidence | 4/30 |
| Recording-category agreement among confident classifications | 1/4 |
| Available paddle measurements | 2/30 |
| Exact full feedback-text matches to frozen output | 2/30 |
| Median / maximum detected-end to UI assignment | 26 / 397 ms |
| Results before source-file end | 3 |
| Results 0-2 seconds after source-file end | 26 |
| Results more than 2 seconds after source-file end | 0 |

| Final phone status | Clips |
|---|---:|
| LOW_CONFIDENCE | 2 |
| INSUFFICIENT_DATA | 23 |
| NO_OUTPUT | 1 |
| PARTIAL | 2 |
| CORRECTIVE | 2 |

## Interpretation

The app produced substantive coaching on 2 of 30 clips, versus 21 frozen outputs previously prepared for Coach C. This measures coverage, not coaching correctness. The frozen desktop and live phone paths differ in several stages, so this comparison does not isolate a single cause.

Insufficient-data results numbered 23; their usable real-pose counts were 12 to 30. The app requires 32 real poses before classification. Missing classification can therefore happen before the original GRU has an opportunity to evaluate a serve. A completed source replay is not proof of a completed motion-gated capture.

Recording-category agreement is not independent Coach C classification accuracy. A CORRECTIVE/POSITIVE status is not proof that its advice is valid. Coach C evaluated the frozen messages, not these new live-app outputs; his ratings cannot be transferred to this run.

Low detected-end latency does not establish a successful two-second coaching result. No-output cases remain failures of availability; source-file end is not an independently marked physical serve endpoint. This replay excludes decoding and camera acquisition, and measures UI text assignment rather than externally observed rendered pixels. Single-pass results do not establish repeatability.

## Suggested next work

1. **Compare exact phone and desktop inputs.** Capture pose timestamps and tensors, then run identical tensors through both classifiers. Isolate pose extraction, capture boundaries and normalization before retraining.
2. **Test explicit start/finish capture.** Compare a recording button or bounded window against automatic gating, retaining setup and follow-through. This may fix missed boundaries, but does not itself fix low pose throughput.
3. **Recover pose and paddle evidence.** Evaluate GPU or scheduling changes on quality as well as speed. Do not lower evidence/confidence requirements just to display more advice. MediaPipe live-stream mode can ignore incoming frames while busy; nominal camera FPS is not usable pose FPS. [Official documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python).
4. **Validate on fresh recordings after tuning.** These 30 clips are now mobile diagnostics. Keep the finished Coach C study separate; train on participant-separated phone data and independently judge any changed coaching outputs.

## Per-clip results

| Clip | Recording category | Phone prediction | Real poses | Phone status | Frozen status | Detected-end to UI ms |
|---|---|---|---:|---|---|---:|
| C001 | lob | topspin | 44 | LOW_CONFIDENCE | LOW_CONFIDENCE | 131 |
| C002 | topspin | Unavailable | 28 | INSUFFICIENT_DATA | CORRECTIVE | 23 |
| C003 | lob | Unavailable | None | NO_OUTPUT | LOW_CONFIDENCE | None |
| C004 | topspin | Unavailable | 27 | INSUFFICIENT_DATA | LOW_CONFIDENCE | 152 |
| C005 | lob | topspin | 46 | PARTIAL | PARTIAL | 128 |
| C006 | lob | Unavailable | 28 | INSUFFICIENT_DATA | CORRECTIVE | 172 |
| C007 | drive | Unavailable | 22 | INSUFFICIENT_DATA | CORRECTIVE | 19 |
| C008 | drive | topspin | 43 | PARTIAL | CORRECTIVE | 201 |
| C009 | topspin | Unavailable | 27 | INSUFFICIENT_DATA | CORRECTIVE | 32 |
| C010 | drive | Unavailable | 16 | INSUFFICIENT_DATA | CORRECTIVE | 206 |
| C011 | drive | Unavailable | 27 | INSUFFICIENT_DATA | CORRECTIVE | 17 |
| C012 | topspin | Unavailable | 30 | INSUFFICIENT_DATA | CORRECTIVE | 278 |
| C013 | topspin | Unavailable | 27 | INSUFFICIENT_DATA | LOW_CONFIDENCE | 23 |
| C014 | drive | Unavailable | 20 | INSUFFICIENT_DATA | LOW_CONFIDENCE | 20 |
| C015 | drive | topspin | 38 | CORRECTIVE | CORRECTIVE | 397 |
| C016 | lob | Unavailable | 26 | INSUFFICIENT_DATA | CORRECTIVE | 19 |
| C017 | topspin | Unavailable | 28 | INSUFFICIENT_DATA | CORRECTIVE | 26 |
| C018 | lob | Unavailable | 21 | INSUFFICIENT_DATA | CORRECTIVE | 275 |
| C019 | topspin | Unavailable | 27 | INSUFFICIENT_DATA | POSITIVE | 16 |
| C020 | topspin | Unavailable | 28 | INSUFFICIENT_DATA | CORRECTIVE | 23 |
| C021 | lob | Unavailable | 27 | INSUFFICIENT_DATA | CORRECTIVE | 21 |
| C022 | topspin | Unavailable | 12 | INSUFFICIENT_DATA | POSITIVE | 13 |
| C023 | drive | topspin | 34 | LOW_CONFIDENCE | CORRECTIVE | 127 |
| C024 | lob | Unavailable | 27 | INSUFFICIENT_DATA | CORRECTIVE | 20 |
| C025 | lob | Unavailable | 28 | INSUFFICIENT_DATA | LOW_CONFIDENCE | 19 |
| C026 | drive | Unavailable | 12 | INSUFFICIENT_DATA | LOW_CONFIDENCE | 16 |
| C027 | topspin | topspin | 34 | CORRECTIVE | CORRECTIVE | 112 |
| C028 | lob | Unavailable | 25 | INSUFFICIENT_DATA | CORRECTIVE | 10 |
| C029 | drive | Unavailable | 28 | INSUFFICIENT_DATA | LOW_CONFIDENCE | 275 |
| C030 | drive | Unavailable | 18 | INSUFFICIENT_DATA | CORRECTIVE | 188 |

See [protocol](PROTOCOL.md), [per-clip CSV](clip_results.csv), [all event records](results.json) and [APK/source identities](protocol.json). Raw logs are in logs/. Summary uses the final completed displayed result per clip; all event records are preserved.

## Exact displayed phone feedback

### C001 - LOW_CONFIDENCE

The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C002 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C003 - NO_OUTPUT

(No completed feedback was displayed during the replay observation window.)

### C004 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.

### C005 - PARTIAL

No supported corrective cue for the available measurements. Overall serve form has not been assessed.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C006 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C007 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C008 - PARTIAL

No supported corrective cue for the available measurements. Overall serve form has not been assessed.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C009 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C010 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C011 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C012 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C013 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C014 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C015 - CORRECTIVE

Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C016 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C017 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C018 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C019 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C020 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C021 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C022 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C023 - LOW_CONFIDENCE

The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C024 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C025 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C026 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Weight shift unavailable.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C027 - CORRECTIVE

Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C028 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C029 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.

### C030 - INSUFFICIENT_DATA

Serve classification unavailable. Record another serve for coaching feedback.
