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
