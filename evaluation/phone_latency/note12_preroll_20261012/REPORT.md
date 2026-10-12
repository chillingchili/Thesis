# Timestamped pre-roll candidate — 12 October 2026

## Outcome

Implemented and installed on Xiaomi 23021RAAEG. Classification availability improved on the tested C001 replay, with post-gate response times still below two seconds. No substantive coaching was produced on any of the three tested clips; this is not a demonstrated accuracy or coaching-quality improvement.

## Changes

- One second of real timestamped pose history is retained before the motion trigger. The trigger sample is counted once; refractory-period samples are excluded and history clears between serves.
- The active classification window retains setup and follow-through rather than dropping its start after 128 observations. Long windows select 128 unique real observations spread over elapsed time. Short windows retain every real observation and trailing zero padding; interpolation does not manufacture evidence.
- The five-fold original GRU runs once at serve end. Intermediate inference no longer competes with pose acquisition.
- Original assets, moment matching, 32-observation minimum, confidence threshold 0.60, feedback wording and end-detector constants remain unchanged. Hip/paddle measurement collection still uses the active gate; the new pre-roll expands classifier context only.

## Same-clip replay comparison

| Clip | Baseline classifications / end events | Candidate classifications / end events | Candidate real frames | Candidate maximum gate-end to UI |
|---|---:|---:|---|---:|
| C001 | 4/9 | 10/10 | 34–50 | 0.415 s |
| C002 | 0/7 | 1/12 | 12–36 | 0.372 s |
| C007 | 0/1 | 0/2 | 18–19 | 0.284 s |

Candidate: 24 end events, 11 classifications, 13 insufficient-data notices, 11 low-confidence notices, zero corrective/positive cues. All 24 events had paired UI-update logs. C001 confidence ranged 0.439–0.534; the sole C002 prediction was Topspin at 0.401. C007 did not reach the 32-frame requirement.

The largest measured final GRU inference was 136 ms; the largest gate-end-to-UI interval was 415 ms. Baseline maximum was 786 ms, but these sequential runs do not isolate scheduling or thermal effects, and a faster maximum is not proof of a general speedup. Clip-level event counts differ; these are repeated gate events, not independent test clips or a matched accuracy denominator.

## Timing and replay limitations

The harness predecodes files and paces frames at the live path. Preparation is excluded from the response interval. Between repeated clips it sleeps for three seconds **without delivering stationary camera frames**. A motion gate waiting for low-motion callbacks can therefore stay open across that gap or even overlap clip boundaries. This particularly limits peak-to-end interpretation and how much pre-roll is available. The same limitation affects the baseline. It must not be called a faithful measurement of actual camera capture completion.

Candidate motion-peak-to-UI maxima were 4.406 s (C001), 4.603 s (C002) and 0.952 s (C007). Motion peak is not ground-truth serve completion. UI logging measures label updates, not independently observed screen presentation. No all-objectives-pass or live two-second thesis claim follows.

## Validation and provenance

- 38 JVM tests passed, including timestamp expiry, trigger inclusion, frame uniqueness, elapsed-time selection, short-window padding, copying and reset behavior.
- Debug APK build and lint passed.
- Candidate APK and original-model asset hashes are recorded in `build.json`; all compared GRU/manifest assets match the baseline.
- Installed on the connected Xiaomi with existing app data preserved; no model training or Coach C output modification.
- Every retained event is in `C001.log`, `C002.log`, `C007.log`; `results.json` and `summarize_replay.py` provide reproducible scoring.

## Next engineering target

Fix acquisition and end-detection validation before considering a lower confidence gate. A camera trial with adequate stationary setup and follow-through, timestamped input frames, an independently labeled serve endpoint, and actual output/render checks is needed. A replay harness can supply explicitly marked idle frames for end detection, but repeated synthetic frames must not be counted toward the 32 real-frame requirement. Once the final capture policy is fixed, evaluate or retrain with matching sampling/window conventions on properly separated data. Retain the original frozen desktop evaluation as a distinct configuration.
