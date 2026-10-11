# Xiaomi recorded-core latency benchmark

Device: Xiaomi 23021RAAEG; SoC SM6225; Android 14; reported RAM 5.51 GiB.

**This measures recorded-file processing, not live capture-end-to-displayed-feedback latency. The thesis live two-second target remains unverified.**

Measured attempts: 9; successful: 9; failures: 0. Warm-up runs excluded: 1.

| Successful-run statistic | Full recorded processing |
|---|---:|
| Median | 68.576 s |
| P95 (nearest rank) | 94.284 s |
| Slowest | 94.284 s |

Completed within two seconds: 0/9 measured attempts. This comparison does not establish the live target.

## Mean component times on successful runs

| Component | Mean |
|---|---:|
| decode_ms | 7.454 s |
| pose_ms | 12.628 s |
| paddle_ms | 54.595 s |
| hip_accumulation_ms | 0.001 s |
| gru_ms | 0.078 s |
| final_measurements_and_feedback_ms | 0.003 s |

Initial GRU/paddle/configuration loading: 0.483 s. Per-clip pose construction is recorded separately in runs.csv.

## Scope

- Median/p95/max describe successful runs only; failed attempts remain in the count and within-2s denominator.
- Nine measured runs across three fixed clips are an initial device benchmark, not a population-wide performance estimate.
- Fresh VIDEO pose tracker per clip; no live camera, motion gate, UI display timing or concurrent live inference.
- Initial GRU/paddle loading and per-run pose loading recorded separately; total_processing_ms excludes these.
- USB/charging, power and thermal conditions are retained in raw.json; no battery-state normalization was applied.

Inputs are C001, C002 and C007, fixed before timing (first recorded Lob, Topspin and Drive selections). All frames are decoded at source dimensions. Clip identity and asset hashes are in raw.json. No Coach C answers affect timing.

Next: measure actual camera completion through final displayed feedback, including queued work and end-detection delay, on this same phone. Do not substitute GRU-only inference time for that measurement.

## Comparison with the original phone baseline

Median: 68.619 s before versus 68.576 s after (0.062% lower). This is effectively unchanged; the crop-only improvement does not provide a meaningful full-pipeline speedup.

Slowest: 104.650 s before versus 94.284 s after. Nine runs in each separate session cannot attribute this change to the optimization; both run order and device conditions can affect timing. P95 equals the maximum at this sample size.

All nine measured attempts succeeded in each session, with identical clip and model asset hashes. Installed APK hash was checked against the optimized build before the run. See build_manifest.json.

Compared classification, feedback status/text, measurement availability and frame counts across matching clip/round pairs: 0 differences. Maximum confidence difference: 0.000000000. Exact comparison is in comparison.json.

This ran the native Android benchmark activity on the Xiaomi using saved clips. It does not exercise the MainActivity live-camera motion gate or measure serve-end-to-displayed-feedback delay. The live two-second target remains unverified.
