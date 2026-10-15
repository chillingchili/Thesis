# Thesis edit brief — 14 October 2026 evidence update

This document gives an AI editor everything needed to update the thesis (Chapters 5–6)
with this week's phone-benchmark and live-pipeline evidence. All paths are relative
to the repo root. Numbers below are measured, not estimated; caveats are load-bearing.

## 1. What changed since the September manuscript

1. **Recorded-file benchmark, Xiaomi Note 12 (SM6225, Android 14), rerun 14 Oct.**
   Median full processing 68.6s -> **23.2s** (9 measured runs of C001/C002/C007).
   Per-run medians: decode 6.9s, pose 11.7s, paddle 55.4s -> **3.4s** (two-pass
   contact window), GRU 0.08s. Output statuses identical before/after
   (C001 POSITIVE, C002/C007 CORRECTIVE). Artifacts:
   `evaluation/phone_latency/note12_20261014/{REPORT.md,summary.json,runs.csv,raw.json}`.
2. **Three more phones benchmarked (v6 protocol, 9 runs each).** S22 Ultra results
   arrived 14 Oct as tester exports (`Benchmark/Samsung S22 Ultra/`, untracked
   source) and were published with the standard summarizer — twice: the first run
   (unplugged, 40.1°C battery, thermal SEVERE) and a charging rerun at the tester's
   request (`Benchmark/Samsung S22 Ultra/new/`, 38-39°C, thermal MODERATE).
   `evaluation/phone_latency/{s22ultra_v6_cpu,s22ultra_v6_gpu}` (first run) and
   `{s22ultra_v6_cpu_charging,s22ultra_v6_gpu_charging}` (rerun).
   Full-processing medians: Nothing GPU 9.7s < S22 GPU 10.3s/10.5s < S22 CPU
   12.6s/13.0s < Nothing CPU 13.4s < Infinix GPU 24.0s ~= Note 12 two-pass 23.2s <
   Infinix CPU 53.0s < Note 12 baseline 68.6s. The two S22 runs agree within noise
   with byte-identical per-clip predictions — keep both cited as a stability pair;
   the phone's sustained heat (38°C+ even while charging) is itself a documented
   finding, and Nothing leads on merit, not on a throttling technicality.
   GPU-vs-CPU outputs differ slightly (pose landmarks shift GRU inputs; e.g. S22
   C002 LOW_CONF 0.49 CPU vs CORRECTIVE 0.69 GPU) — same phenomenon as the earlier
   pilot, not a code change.
3. **30-clip live-path evaluation with paddle recovery (record: `mobile_pass2`).**
   Config = previous GPU audit config + contact-window full-res second pass at
   source resolution (same detector/rules) + 60s cooldowns, 350ms paddle spacing.
   Results (`evaluation/coach_c/mobile_pass2_20261014/`): outputs 21/30,
   substantive 16/30 (15 corrective + C008 positive), paddle measured 16/30
   (was 1/30), same-status-as-frozen 15/30, exact text matches 2/30, end-to-UI
   median 5.1s (includes the 4-7s pass by design). Runner flags added:
   `--relaxed-paddle/--fullres-paddle/--cooldown-secs` in
   `scripts/run_coach_c_mobile_live.py`; reproduce via `evaluation/coach_c/mobile_pass2_20261014/PROTOCOL.md`.
4. **Superseded 150ms-spacing experiment (`mobile_showcase_20261014`).** Paddle
   14/30 but availability fell to 18/30; 9/10 recovery at 350ms proved spacing-load
   (plus heat) was the cause, not the pass. Marked superseded in its REPORT.
5. **Showcase demo feature (for the thesis video, demo-only).** Menu item plays
   C002/C005/C023 through the live pipeline with video visible and hands-free
   chaining (`android/app/src/main/java/com/thesis/pickleballserve/MainActivity.kt`,
   `ServeClipReplay.kt`, menu/strings/layout, `android/README.md`). Demo config =
   normal 350ms spacing + full-res pass; disclosed as demo-only in the README.

## 2. Numbers the editor may cite (with their exact scope)

- Paddle recall, live path: **1/30 -> 16/30** (record run). Thirteen OPTIMAL, three
  OUTSIDE_BASELINE. The 640x360 live worker still measures almost nothing alone;
  resolution, not scheduling, is the bottleneck (evidence: full-res pass logs
  `measured=1`; 150ms spacing alone recovered nothing).
- Availability: GPU audit 27/30 outputs and 18/30 substantive vs pass2 record 21/30
  and 16/30. Misses = cold-start C001-C005 after fresh install (both record runs;
  runner needs an unlogged warmup replay) + residual gate variance. One verdict
  (C027, topspin 0.85, paddle measured) rendered after the runner's +2s dump and
  went uncounted — runner cutoff must cover the pass (use +10s).
- End-to-UI median 126ms -> 5.1s is the declared speed-for-recall trade. **Neither
  figure verifies the 2s live capture-to-display target** (both exclude capture;
  recorded-file processing is a different metric — do not conflate).
- No independent Coach C subtype accuracy exists (instrument omits subtype labels
  by design). Recording-category agreement and exact-match counts are diagnostic
  only. Coach C Valid ratings attach to frozen Session 2 messages only and do not
  transfer to any live-app outputs, including cases where fuller evidence flips a
  verdict (e.g. frozen C005 PARTIAL vs live POSITIVE with paddle measured).

## 3. What must NOT be claimed

- Do not present 85%+ CV as generalizing to unseen players (held-out ceiling stands).
- Do not present the 9.7-23s recorded-processing figures as live-latency results.
- Do not convert 90.0% Session 2 Valid into corrective-feedback accuracy.
- Do not present demo/showcase outputs as validated; label the demo config.
- Do not claim the 150ms arm; it was reverted (kept reproducible via flag only).

## 4. Suggested chapter updates

- Ch.5 phone-performance section: replace/augment the 68.6s Note 12 figure with
  the 23.2s two-pass rerun (outputs identical); add the 4-phone leaderboard and
  the S22 GPU-vs-CPU output note.
- Ch.5 live-pipeline section: add the pass2 record (16/30 paddle, 21/30 outputs,
  5.1s end-to-UI incl. pass) alongside the GPU audit, with the cold-start and
  dump-cutoff methodology notes.
- Ch.6 limitations: paddle recall at 640x480 as the standing bottleneck; 2s live
  target still unverified; gate run-to-run variance; pass2 as thorough-mode, not
  default.
- Methods: one paragraph on the second pass (contact +/-5 usable frames, even
  stride, one forward batch at source resolution, verdict-only, ~4-7s cost).

## 5. Open items (do not present as done)

- Runner warmup replay + extended post-finish wait (specified, not implemented).
- C025 verdict wedge (pass2 measured=1, then no verdict, no crash line) — single
  occurrence under watch.
- Retraining/fine-tuning the paddle detector (the actual recall fix); GPU delegate
  already measured and rejected (17% faster, detection-changing precision loss —
  see `PaddleStream.kt` header comment).
