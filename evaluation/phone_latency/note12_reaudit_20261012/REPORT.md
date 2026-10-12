# Installed-app live-path re-audit — 12 October 2026

## Finding

The installed app responds in under two seconds **after its own end detector fires**, but this rerun produced no substantive coaching. It is the same original five-fold GRU, not a smaller replacement model. The live input and scheduling differ from the frozen desktop evaluation.

## Observed results

| Clip | Gate-end events | Final classifications | Confidence ≥0.60 | Median / max gate-end to UI update | Usable frames |
|---|---:|---:|---:|---|---|
| C001 | 9 | 4 | 0 | 0.270 / 0.786 s | 24–34 |
| C002 | 7 | 0 | 0 | 0.015 / 0.276 s | 14–23 |
| C007 | 1 | 0 | 0 | 0.281 / 0.281 s | 18–18 |

All 17 recorded gate-end events had a corresponding UI-update log within 0.786 s. Thirteen events returned INSUFFICIENT_DATA; four returned LOW_CONFIDENCE. All four classifications were C001 → Topspin, with confidence 0.412–0.491. No CORRECTIVE or POSITIVE feedback was produced. These are repeated events from only three recordings, not 17 independent serves or a population accuracy estimate. C007 triggered only one end event during its observation; this protocol does not count every replay as a successfully detected serve.

Motion-peak-to-UI times were longer: C001 median 4.116 s (max 4.517 s), C002 median 4.249 s (max 4.559 s), C007 1.029 s. Motion peak is not independently labeled serve completion. Therefore neither this proxy nor gate-end timing proves the thesis requirement from actual capture/serve completion to displayed feedback. The UI timestamp records the runnable updating labels, not an independently measured rendered screen frame.

## Comparison with the original frozen outputs

| Clip | Frozen desktop classification | Confidence | Frozen feedback | Fresh phone outcome |
|---|---|---:|---|---|
| C001 | topspin | 0.515 | LOW_CONFIDENCE | 4 low-confidence classifications / 9 end events |
| C002 | topspin | 0.664 | CORRECTIVE | No classification |
| C007 | topspin | 0.616 | CORRECTIVE | No classification |

The desktop supplied corrective cues for C002 and C007; the phone supplied none in this rerun. This is an output-availability regression on these examples, not proof that every desktop prediction is correct. Coach C did not supply independent subtype labels. His Session 2 ratings describe the frozen desktop packet and cannot be transferred to this live pipeline.

## Mechanism supported by the evidence

1. All five installed GRU asset hashes match the source assets and the Session 2 freeze. Runtime initialization confirms ENSEMBLE, five folds, moment matching enabled. There is no evidence of a weaker replacement network.
2. MainActivity requires at least 32 usable frames for final inference. Every null classification had fewer than 32; classified events had 32–34. Fast unavailable notices must not be counted as successful inference.
3. The live path uses an active-motion window, reduced analysis resolution, latest-frame processing and a bounded asynchronous paddle queue. The desktop packet uses a different first-window extraction. Thus identical weights do not imply identical inputs or outputs.
4. Paddle queue drops were recorded (C001: 14–21 per event; C002: 1–6; C007: 5). Drops and the bounded drain can reduce measurement evidence. The present audit does not isolate the effect of each scheduling change on accuracy.

## Recommendation

Keep the original GRU weights. First test a timestamped pre-roll buffer and complete-serve window so setup and follow-through are retained; measure actual usable-frame rate and detection coverage. Make training and live sampling/window conventions consistent, and compare outputs on fixed clips before adopting changes. Do not simply lower the 32-frame minimum or the 0.60 gate to make more outputs appear.

Then measure from an independently marked serve/capture endpoint to final rendered feedback, while reporting missing classifications and substantive-feedback coverage alongside latency. Preserve the frozen Coach C results as a separate evaluated configuration. No model or app code was changed in this audit.

## Reproduction and limitations

Device: 23021RAAEG, adb 1fb637d. Installed APK SHA-256: `a3165396c5359d2d67cb09100fd2100e0f43214e13c2ce7001632f4aaf6f3f33`.

The already-installed app was tested without rebuilding or reinstalling. Its replay harness substitutes prerecorded frames for camera input while retaining the live gate, pose, paddle, classifier and feedback path. Each clip was observed for 60 seconds after preparation was detected by ten-second polling; the initial partial polling interval is also retained. Preparation is separate, includes expensive file decoding, and is not a camera-processing benchmark. All captured end events are included without removing slow runs or failed predictions. No actual camera session or independent render-time measurement was performed.

The first attempt used an insufficient 90-second preparation/observation window and yielded no completed C001 replay. It was superseded by the preparation-aware rerun. A device-clock mismatch was found: final logs were isolated by application process (C001 PID 17953, C002 PID 21167, C007 PID 23150), rather than trusting wall-clock filtering. C007 startup was retried after waiting for process creation. Results describe the retained final runs only.

Evidence: `C001.log`, `C002.log`, `C007.log`, `results.json`, `device.json`, `frozen_comparison.json`. `summarize_replay.py` regenerates the counts. The installed APK is kept locally for traceability and excluded from Git. Earlier October 11 derivations assumed bounds that these observations do not validate; use measured results here for these tested conditions.
