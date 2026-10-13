# Xiaomi recorded-core latency benchmark

Device: Nothing A001; SoC MT6878; Android 16; reported RAM 7.23 GiB.

**This measures recorded-file processing, not live capture-end-to-displayed-feedback latency. The thesis live two-second target remains unverified.**

Measured attempts: 9; successful: 9; failures: 0. Warm-up runs excluded: 1.

| Successful-run statistic | Full recorded processing |
|---|---:|
| Median | 38.003 s |
| P95 (nearest rank) | 53.978 s |
| Slowest | 53.978 s |

Completed within two seconds: 0/9 measured attempts. This comparison does not establish the live target.

## Mean component times on successful runs

| Component | Mean |
|---|---:|
| decode_ms | 4.415 s |
| pose_ms | 11.905 s |
| paddle_ms | 25.403 s |
| hip_accumulation_ms | 0.001 s |
| gru_ms | 0.073 s |
| final_measurements_and_feedback_ms | 0.002 s |

Initial GRU/paddle/configuration loading: 0.017 s. Per-clip pose construction is recorded separately in runs.csv.

## Scope

- Median/p95/max describe successful runs only; failed attempts remain in the count and within-2s denominator.
- Nine measured runs across three fixed clips are an initial device benchmark, not a population-wide performance estimate.
- Fresh VIDEO pose tracker per clip; no live camera, motion gate, UI display timing or concurrent live inference.
- Initial GRU/paddle loading and per-run pose loading recorded separately; total_processing_ms excludes these.
- USB/charging, power and thermal conditions are retained in raw.json; no battery-state normalization was applied.

Inputs are C001, C002 and C007, fixed before timing (first recorded Lob, Topspin and Drive selections). All frames are decoded at source dimensions. Clip identity and asset hashes are in raw.json. No Coach C answers affect timing.

Next: measure actual camera completion through final displayed feedback, including queued work and end-detection delay, on this same phone. Do not substitute GRU-only inference time for that measurement.
