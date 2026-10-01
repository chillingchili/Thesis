# Redmi Note 12 landing analysis benchmark

Measured on the connected Redmi Note 12 (23021RAAEG / tapas), Snapdragon 685 (SM6225),
Adreno 610, Android 14, on 2026-09-30. Debug APK, foreground/unlocked screen, CPU/XNNPACK
with two inference threads. No server inference. Tests run through the actual landing activity.

Installed APK SHA-256: `0083421d6c3f8d1b7c2cbbf22fc43c5d5355de6ba70f8fce1a26411a1a605647`.

## Acceptance gate

- Replay elapsed time <= source duration × 1.10 + 200 ms.
- At least 95% of the 30 fps playback frames submitted to the preview.
- Tracking work P95 <25 ms; Android window rendering P95 <33.4 ms.
- A calibrated/tracked court for >=70% of the ten-second clip.
- Ball observations must be present. At least five of eight visible-ball reference samples must
  match within eight pixels in the 640px tracking coordinate system.

This is a near-real-time 30 fps playback/tracking gate, not a requirement to run both neural
networks on every frame. A 60 fps source is sampled at approximately 30 fps using its timestamps.
Inference initialization precedes the replay timer; first-frame preparation is included in elapsed time.

## Final foreground results, 23:06

| Source | Preview frames | Replay elapsed | Effective fps | Tracking P95 | Rendering P95 | Reference samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p / 29.97 fps | 300/300 | 10.144 s | 29.57 | 18.00 ms | 19.20 ms | 7/8 |
| 1080p / 59.94 fps, sampled to 30 | 301/301 | 10.658 s | 28.24 | 20.86 ms | 21.52 ms | 6/8 |

Both passed the gate. Source spans measured from decoded timestamps were 10.009 s and 10.109 s.
The 720p run had 2 frames, and the 1080p run had 27 frames, arriving more than 50 ms after their
target timestamp. Keeping decoder input buffers filled reduced the latter from 61 in the preceding
run. Passing the elapsed-time gate does not mean zero timing jitter.

An earlier implementation took roughly 38.5 seconds for a two-second / 60-frame clip (~1.56 fps).
The separate frame/inference workers remove that serial inference bottleneck. Model-confirmed
160px crops take approximately 50–80 ms on this phone; full-frame refreshes take longer and run
independently of playback. Court camera-motion tracking uses a 320px image and native homography.

## Correctness and limits

The full device suite passed nine tests, covering bitmap refresh/ownership, rotation, decoder EOS,
model class linkage, interpreter thread ownership, delayed detections, lost tracks, and both real
playback benchmarks. Seventeen JVM tests passed. Lint: zero errors, 83 warnings (mostly existing
UI/localization warnings). The two final playback benchmarks and decoder regression were rerun
after packaging cleanup and the decoder buffering change: all three passed.

Reference samples are a small regression check on visible serve frames, not a held-out accuracy
study. They do not measure recall over all balls, all clips, camera angles, lighting, or ball colors.
The 300-image input-size experiment found 361/386 labeled balls at 640px versus 308/386 at 320px
with the original PyTorch checkpoint, so a 320px-only approach was rejected. The runtime instead
uses motion-proposed crops at the 640px image's pixel scale, confirmed by the trained model, with
320px full-frame fallback. Motion alone never becomes a ball detection.

Court coordinates and landing zones remain model estimates. Borderline line calls varied across
runs/encodings, and are not certified accurate by the performance gate. Long-duration thermal
behavior and other footage have not been exhaustively benchmarked.

## Reproduce

Build `:app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest :app:lintDebug`,
install both APKs with `adb install -r`, unlock the phone, then run:

```text
adb shell am instrument -w -r -e class com.thesis.pickleballserve.landing.LandingPipelineTest,com.thesis.pickleballserve.landing.RealtimeTrackerTest com.thesis.pickleballserve.test/androidx.test.runner.AndroidJUnitRunner
```

Read the `LandingBenchmark` and `LandingActivity` log tags. Keep the phone unlocked. The activity
keeps its visible screen awake and pauses analysis if hidden. `connectedDebugAndroidTest` may
uninstall the target app at teardown; direct instrumentation retains the installed app.
