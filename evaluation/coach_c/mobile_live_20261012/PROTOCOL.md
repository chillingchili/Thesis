# Evaluation protocol and interpretation

This evaluates the current original-GRU Android app on all 30 source clips used by Coach C. Source hashes match researcher_key.csv on the computer and phone. All five GRU asset hashes match the pre-experiment baseline. Only replay-specific displayed-text logging was added; no inference, threshold or feedback rule changes were made for this evaluation.

One replay per clip, C001 through C030, restarting the app each time. CPU pose, pose priority enabled, original end detector, 0.60 confidence threshold and 32-real-pose minimum. Every source clip stays in the denominator, including absent results. This is not a repeated-trials stability estimate. CPU/GPU performance from the earlier three-clip pilot is not substituted for this full-set result.

Frames are decoded before playback, then delivered at camera pace through the live app pose, gate, GRU, paddle and feedback path. Synthetic idle images run for 1.5 seconds before and 3 seconds after the clip. They support gate settling but are excluded from classifier and mechanics evidence. This is recorded-video replay, not validation of actual camera exposure, focus, orientation or thermal conditions during a session. Initialization and decoding are outside the reported live timing interval.

Status summaries use the final displayed completed result per clip. All events, including multiple outputs if any, remain in results.json. Classification availability includes below-threshold predictions; confident classifications are reported separately. CORRECTIVE/POSITIVE counts measure substantive output availability, not demonstrated validity. Recording categories are reference labels, not independent Coach C subtype annotations.

Coach C's old ratings apply to the frozen messages he reviewed. They are not reassigned to the phone's new messages. Exact message matching is diagnostic, not new independent feedback validation. The frozen 21/30 substantive-output count is a coverage comparison, not a controlled ablation isolating one cause.

Timing is measured from the app's end-detection timestamp to UI text assignment. A separate source-file-end interval is reported, including negative values. Neither file end nor the motion gate supplies an independently annotated physical serve endpoint. Absent outputs and premature results cannot be counted as two-second successes. Fast abstentions are not successful coaching.

## Suggested next work

1. Save phone pose timestamps and tensors, then replay those exact tensors through the desktop and Android classifiers. Compare pose extraction, sequence boundaries, normalization and probabilities stage by stage. This isolates acquisition/preprocessing problems from model-export problems.
2. Test an explicit start/finish recording control or bounded capture window against automatic gating. Include setup and follow-through. This may recover missed boundaries but does not itself repair low pose throughput. Do not silently lower the 32-frame or confidence gates just to increase output count.
3. Investigate pose throughput and paddle measurement coverage together. MediaPipe live-stream mode can ignore new input frames while busy: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python . Therefore camera FPS is not usable pose FPS. GPU is a candidate, but the earlier pilot's changed classifications and late results require quality validation before adoption.
4. Train on actual phone-produced pose sequences only after the input representation is stable. Use participant-separated development data. These 30 clips are now mobile diagnostics; if used to tune changes, collect a fresh held-out set for final claims.
5. Independently validate changed coaching outputs and measure time from annotated physical serve completion. Preserve the completed Coach C study and report the mobile evaluation separately.

## Reproduce

Run `python scripts/run_coach_c_mobile_live.py` with Xiaomi serial 1fb637d connected, then `python scripts/summarize_coach_c_mobile_live.py` and `python scripts/report_coach_c_mobile_live.py`. Existing completed logs are reused; use a new output directory for an independent rerun. The runner installs the current debug APK and verifies the original five GRU assets before running. `protocol.json` identifies the exact benchmark APK and all source clips. `logs/` retains raw app output, `clip_results.csv` contains exact displayed feedback alongside the frozen messages, and `results.json` retains individual event timestamps.
