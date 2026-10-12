# Phone latency benchmark

The app has a **Latency Test** button with **Run latency test** and **Export results**.
The initial protocol uses C001, C002 and C007, selected before measuring as the
first available recorded Lob, Topspin and Drive categories. These are not
independently relabeled by Coach C for the benchmark.

## What this measures

Each trial opens a saved clip, decodes every frame at source dimensions, runs Lite
pose tracking, accumulates hip and paddle measurements, evaluates the original
five-fold GRU, and computes feedback. No cached poses or cached predictions are
used. The paddle detector retains its normal every-second-usable-frame stride.
The classifier uses Android's last 128 usable frames, without the live motion gate.
This is distinct from the first-window desktop Coach C inference.

One full C001 warm-up is recorded but excluded from aggregate results. Three rounds
of all three clips follow, giving nine measured attempts. Trackers reset per clip;
GRU and paddle models stay loaded. Initial model loading and per-clip pose loading
are timed separately from file-open-to-feedback processing. All failures and
missing measurements remain in the export. Low-confidence outputs count as
completed processing, not as valid coaching.

**This does not verify the live two-second thesis target.** The live camera
processes frames during the serve and can drop frames under load. A valid live
measurement must include capture completion, outstanding work, end detection and
the final displayed feedback. GRU inference time or post-extraction feedback time
alone cannot establish it. The benchmark also excludes the auxiliary landing stream.

## Reproduce on a connected phone

Build and install the debug APK and Android test APK. Keep the phone idle and
record the actual charging, power-save and thermal conditions; do not silently
discard slow runs. The benchmark records those conditions in its JSON.

Prepare the three clips in the app's private `files/latency_input` folder. With a
debuggable build, copy each source video to `/data/local/tmp/`, then use
`adb shell run-as com.thesis.pickleballserve cp` to copy it into that folder.
On this Xiaomi, direct copies to shared Android/data storage were not readable by
the app. No runtime timing was collected from that unsuccessful setup attempt.

Run with the button, or automate the same screen using:

```powershell
adb shell am instrument -w -r -e class com.thesis.pickleballserve.CoreLatencyTest com.thesis.pickleballserve.test/androidx.test.runner.AndroidJUnitRunner
```

Results are saved incrementally in the private `files/latency_results` directory.
Export using the button, or retrieve the selected JSON with `adb exec-out run-as`.
The test checks completion and record structure; it deliberately does not assert a
two-second pass. Inspect per-run success fields even if instrumentation completes.

From the repository root:

```powershell
python scripts/summarize_phone_latency.py "path/to/core-result.json" --out "evaluation/phone_latency/new_run"
```

The summary keeps the untouched export, produces CSV and a readable report, and
reports median, nearest-rank P95, maximum, failures and completion within two
seconds. With only nine measured runs, P95 equals the slowest observed run. This
is an initial device benchmark, not a guarantee for every player or input.

Benchmark instrumentation changes do not retrain the model or alter Coach C's
frozen outputs. Record the benchmark APK hash separately from the earlier app
snapshot; a new app binary must not be represented as that same frozen binary.

## Android paddle corrections found during device testing

Device testing exposed invalid integer-array writes to an 8-bit OpenCV image,
reversed RGBA grayscale channel interpretation, and missing float conversion in
the edge-point rectangle fallback. These were corrected and covered by device
regression tests. Early failed runs are retained separately from the completed
benchmark. These implementation fixes do not establish desktop/Android output
parity or alter the frozen Coach C feedback files.


Paddle preprocessing optimization is installed and verified on the Xiaomi: grayscale crop preparation fell from 5.20 to 1.38 ms with identical tested pixels and angles. This is a component improvement; the subsequent full recorded-clip benchmark found effectively unchanged latency; live latency remains unmeasured. See [the optimization evidence](../evaluation/phone_latency/note12_optimization_20261010/REPORT.md).


[Full optimized Android rerun](../evaluation/phone_latency/note12_optimized_full_20261010/REPORT.md): 9/9 measured trials passed. Median 68.576 s versus 68.619 s before (0.06% lower), effectively unchanged. This is recorded-video processing, not live-camera latency.


## Installed live-path re-audit — 12 October 2026

[Fresh replay findings](../evaluation/phone_latency/note12_reaudit_20261012/REPORT.md): 17/17 observed gate-end events updated the UI within 0.786 seconds, but 13 had insufficient frames and four were low-confidence; no substantive coaching was produced. The same original GRU weights are installed. This verifies narrow post-gate responsiveness on three replayed clips, not actual capture-to-display latency or Coach C desktop/phone equivalence.


## Pre-roll candidate — 12 October 2026

[Implementation and phone comparison](../evaluation/phone_latency/note12_preroll_20261012/REPORT.md): timestamped one-second classifier pre-roll, complete active window and final-only original-GRU inference. C001 classification availability increased from 4/9 to 10/10 observed end events, but all candidate outputs remained low-confidence or insufficient-data. All 24 candidate post-gate UI updates occurred within 0.415 s. The replay gaps omit idle camera frames, so this does not validate physical serve-end latency. Model weights, 32-observation minimum and confidence gate remain unchanged.


## Five-method follow-up — 12 October 2026

[Full experiment report](../evaluation/mobile_methods_20261012/REPORT.md) covers pose priority, revised end detection, mobile-rate training, GPU pose and GRU distillation. Pose priority is now the default; original weights and 0.60 confidence gate remain. GPU and revised end detection are opt-in research flags; new models remain isolated. All 53 observed gate-end UI updates were below 0.588 s, but one of 54 replays had no end event and source-end timing/quality limits prevent a full live two-second claim.


## Full Coach C replay audit - 12 October 2026

All 30 clips were replayed once through the CPU live app with pose priority and the original GRU. Final statuses: 2 corrective, 2 partial, 2 low confidence, 23 insufficient data, 1 no completed output. Substantive coverage was 2/30 versus 21/30 in the frozen desktop outputs. The 23 insufficient cases retained only 12-30 real poses, below the 32-pose minimum. Two clips split into two insufficient captures. Gate-end to UI assignment was 26 ms median and 397 ms maximum for final displayed outputs; these timings mostly describe notices, not successful coaching. Full physical-capture latency and cue validity are not established. See [full report](../evaluation/coach_c/mobile_live_20261012/REPORT.md), [protocol](../evaluation/coach_c/mobile_live_20261012/PROTOCOL.md), and exact messages in clip_results.csv. Coach C ratings remain attached only to the frozen outputs he reviewed.


## Improved automatic phone configuration - 12 October 2026

Installed GPU pose (CPU startup fallback), time-normalized automatic motion gating, and guarded shared-mechanics feedback despite uncertain serve classification. No manual recording button or new model weights. On the same 30 diagnostic replays: 18 corrective, 3 partial, 3 low confidence, 3 insufficient, 3 no completed output. Eleven corrective outputs use the new classification-independent rule path. Maximum detected-end to UI assignment: 423 ms; C005 was 3126 ms after source-file end. Coverage improved substantially; cue validity and consistent physical-serve-to-display <=2 seconds are not established. See [comparison and exact outputs](../evaluation/coach_c/mobile_gpu_rate_20261012/REPORT.md). This installed configuration supersedes the CPU default described in the earlier audit above.
