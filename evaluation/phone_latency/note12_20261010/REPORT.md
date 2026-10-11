# Xiaomi recorded-core latency benchmark

Device: Xiaomi 23021RAAEG; SoC SM6225; Android 14; reported RAM 5.51 GiB.

**This measures recorded-file processing, not live capture-end-to-displayed-feedback latency. The thesis live two-second target remains unverified.**

Measured attempts: 9; successful: 9; failures: 0. Warm-up runs excluded: 1.

| Successful-run statistic | Full recorded processing |
|---|---:|
| Median | 68.619 s |
| P95 (nearest rank) | 104.650 s |
| Slowest | 104.650 s |

Completed within two seconds: 0/9 measured attempts. This comparison does not establish the live target.

## Mean component times on successful runs

| Component | Mean |
|---|---:|
| decode_ms | 7.149 s |
| pose_ms | 12.827 s |
| paddle_ms | 55.404 s |
| hip_accumulation_ms | 0.001 s |
| gru_ms | 0.083 s |
| final_measurements_and_feedback_ms | 0.003 s |

Initial GRU/paddle/configuration loading: 0.294 s. Per-clip pose construction is recorded separately in runs.csv.

## Scope

- Median/p95/max describe successful runs only; failed attempts remain in the count and within-2s denominator.
- Nine measured runs across three fixed clips are an initial device benchmark, not a population-wide performance estimate.
- Fresh VIDEO pose tracker per clip; no live camera, motion gate, UI display timing or concurrent live inference.
- Initial GRU/paddle loading and per-run pose loading recorded separately; total_processing_ms excludes these.
- USB/charging, power and thermal conditions are retained in raw.json; no battery-state normalization was applied.

Inputs are C001, C002 and C007, fixed before timing (first recorded Lob, Topspin and Drive selections). All frames are decoded at source dimensions. Clip identity and asset hashes are in raw.json. No Coach C answers affect timing.

Next: measure actual camera completion through final displayed feedback, including queued work and end-detection delay, on this same phone. Do not substitute GRU-only inference time for that measurement.

## Device findings and implementation audit

- Paddle processing accounts for 72.9% of aggregate full-processing time; mean 55.404 s per clip.
- Original five-fold GRU inference averages 82.8 ms per clip.
- All 10 trials including warm-up decoded every source frame and produced a computed result. Measured trials are only the nine non-warm-up rows.
- Charging stayed connected; battery temperature ranged from 31.0 to 34.0 C. This was a debug/instrumentation run following debugging, not a fresh cold-start or battery-only test.
- Build, Android JVM tests and lint passed. Both native paddle regression tests passed on this phone. APK/source hashes are in benchmark_build.json; model and clip hashes are in raw.json and were checked against this workspace.
- Fixed Android bitmap/OpenCV type handling, RGBA grayscale conversion and edge-point float conversion. A benchmark bitmap-lifetime error was also corrected. Model weights and frozen Coach C outputs were not changed.
- Earlier failed and interrupted exports are retained in diagnostics/. They belong to superseded development builds, not the final benchmark replicates. One earlier structurally complete export has all failed trials; its completed flag does not establish successful processing. The device test now rejects an entirely failed run.

## Next latency work

Paddle detection currently runs synchronously in the pose-result callback. Test a bounded asynchronous paddle worker with timestamp-matched frames, then verify feedback coverage and values against this baseline. Do not silently skip missing paddle results to achieve a faster time.

Live end detection waits for 15 low-motion processed frames. Replace frame-count assumptions with an explicitly validated time-based rule before making a latency claim. Instrument the actual serve/capture endpoint, pending work and final displayed result; a button measuring only GRU inference cannot verify that endpoint.

This benchmark uses full recorded clips and the Android last-window classifier, whereas the frozen Coach C pipeline uses its documented desktop window. The displayed predictions here are not a replacement Session 2 dataset or a desktop/phone parity result.
