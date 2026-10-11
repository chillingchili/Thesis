# Live serve-end critical path derived from recorded benchmark

Source: evaluation\phone_latency\note12_optimized_full_20261010\raw.json (9 successful measured runs).

**This is a derivation from measured components, not an end-to-end live capture measurement. The live two-second target remains unverified.**

## What is being derived

The recorded benchmark times whole-clip processing, which is dominated by decode and
by pose over every source frame. Live does neither on the critical path. What live
does after the motion gate decides the serve ended is:

```
lastServeEndMs = now                        MainActivity.kt:389
awaitPaddleQuiescence(paddleQuiesceMs)      MainActivity.kt:392  <= cap
runInference(frames, finalResult=true)      MainActivity.kt:395  = gru_ms
updateFeedback(...)                         MainActivity.kt:405  = feedback_ms
log feedback_shown inside runOnUiThread     MainActivity.kt:523  = ui post
```

Note that the `lowHoldMs` end-detection delay is *excluded* from `end_to_shown`:
`lastServeEndMs` is stamped when the gate fires, so `end_to_shown` starts there.
`peak_to_shown` adds it back to describe player-visible delay from motion peak.

## Worst case across all measured runs

| Term | Worst | Basis |
|---|---:|---|
| paddle drain | 250.0 ms | code bound (`paddleQuiesceMs`), not measured |
| final GRU inference | 80.3 ms | measured `gru_ms` |
| paddle verdict + feedback rules | 3.9 ms | measured `final_measurements_and_feedback_ms` |
| UI post before `feedback_shown` | 70.0 ms | assumption: one live pose-frame interval |
| **end_to_shown** | **404.1 ms** | sum |
| + one in-flight pose frame | 73.0 ms | upper bound at source resolution; live is 640x480 |
| + low-hold before end fires | 350.0 ms | code constant |
| **peak_to_shown** | **826.4 ms** | player-visible worst case |

Target 2000.0 ms: peak_to_shown has **2.4x** headroom.

## Per run

| clip | round | end_to_shown | peak_to_shown | final_infer | feedback | pose ms/frame | frames floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| C001.mp4 | 3 | 404.1 ms | 826.4 ms | 80.2 ms | 3.9 ms | 72.4 | 51 |
| C002.mp4 | 3 | 403.1 ms | 825.8 ms | 80.2 ms | 2.8 ms | 72.7 | 34 |
| C007.mp4 | 3 | 402.6 ms | 825.3 ms | 79.7 ms | 2.9 ms | 72.7 | 37 |
| C007.mp4 | 2 | 403.3 ms | 825.3 ms | 80.3 ms | 2.9 ms | 72.0 | 37 |
| C007.mp4 | 1 | 402.9 ms | 823.5 ms | 80.0 ms | 2.9 ms | 70.6 | 38 |
| C001.mp4 | 2 | 398.4 ms | 821.3 ms | 74.8 ms | 3.6 ms | 73.0 | 50 |
| C002.mp4 | 2 | 397.1 ms | 819.7 ms | 74.5 ms | 2.6 ms | 72.6 | 34 |
| C001.mp4 | 1 | 399.0 ms | 818.6 ms | 75.4 ms | 3.6 ms | 69.6 | 53 |
| C002.mp4 | 1 | 398.3 ms | 818.6 ms | 75.6 ms | 2.7 ms | 70.3 | 35 |

## Frame accumulation

`finalMinFrames=32` gates whether the final GRU runs at all;
below it, no classification is produced regardless of latency.

Live pose rate is bounded above by the 30 fps camera and below by pose cost charged at
its measured 1920x1080 figure. Live analysis runs 640x480,
so the real rate sits between those bounds. The table below assumes the gate stayed
open for an entire clip; in practice it opens on motion onset and closes 350 ms after
motion stops, so the real active window is shorter than the clip.

| clip | duration | pose ms/frame | fps floor | fps ceiling | frames floor | frames ceiling |
|---|---:|---:|---:|---:|---:|---:|
| C001.mp4 | 3670 ms | 69.6 | 14.4 | 30 | 53 | 110 |
| C001.mp4 | 3670 ms | 73.0 | 13.7 | 30 | 50 | 110 |
| C001.mp4 | 3670 ms | 72.4 | 13.8 | 30 | 51 | 110 |
| C002.mp4 | 2486 ms | 70.3 | 14.2 | 30 | 35 | 75 |
| C002.mp4 | 2486 ms | 72.6 | 13.8 | 30 | 34 | 75 |
| C002.mp4 | 2486 ms | 72.7 | 13.7 | 30 | 34 | 75 |
| C007.mp4 | 2688 ms | 70.6 | 14.2 | 30 | 38 | 81 |
| C007.mp4 | 2688 ms | 72.0 | 13.9 | 30 | 37 | 81 |
| C007.mp4 | 2688 ms | 72.7 | 13.7 | 30 | 37 | 81 |

The floor of 34 frames assumes the gate is open for a whole
clip, which it is not: it opens on motion onset and closes 350 ms after motion
stops. A ~2 s active window inside C002 would yield ~28 frames, below 32, and the
final inference would be skipped entirely — no classification, whatever the
latency. Live pose at 640x480 is cheaper than the 1920x1080 figure charged here,
which is the only reason this may still clear. This is the least certain input in
the derivation and the first thing a live measurement must settle.

## What this does not establish

- It does not measure capture-to-display latency end to end. No camera, no display,
  no real motion gate were exercised.
- The paddle drain is taken as its code cap, not observed. Under backlog the cap
  guarantees the bound but also discards detections, which is a coaching-quality
  question rather than a latency one.
- The in-flight pose frame is charged at its measured 1920x1080 cost, and live runs
  640x480. That deliberately over-charges
  the bound; it is not a measurement at the live resolution.
- The UI post allowance is an assumption of one pose-frame interval.
- The frame-accumulation floor assumes pose cost measured at 1920x1080 and a gate
  window as long as the clip. Both are the least certain inputs here.

A live measurement is still required to close the target. This only establishes that
compute after the gate fires is not the obstacle.
