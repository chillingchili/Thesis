# 30 clips through the live pipeline with relaxed paddle + full-res second pass

> Superseded as the record by `../mobile_pass2_20261014/` (350ms spacing + pass2 +
> cooldowns): paddle 16/30 with availability 21/30. This directory remains the
> valid record of the 150ms-spacing experiment (paddle 14/30, availability 18/30).

Date: 14 October 2026. Device: Xiaomi 23021RAAEG (1fb637d), SM6225, Android 14.
Config: original five-fold GRU, GPU pose attempt, pose priority, motion-rate gating,
0.60 confidence, 32-pose minimum — identical to the `mobile_gpu_rate_20261012`
audit — plus paddle submit spacing 150ms (was 350ms) and a contact-window
full-res second pass at source resolution (same detector and rules). Intent extras
`research_relaxed_paddle` / `research_fullres_paddle`; showcase demo uses the same
settings. One replay per clip, C001..C030, fresh app start each time. All 30 source
hashes and all five GRU asset hashes verified by the runner before the run.

## Findings vs the two previous full-set runs

| Measure | CPU audit (`mobile_live`) | GPU audit (`mobile_gpu_rate`) | This run (`mobile_showcase`) |
|---|---:|---:|---:|
| Clips with completed displayed feedback | 29/30 | 27/30 | 18/30 |
| Substantive coaching (CORRECTIVE/POSITIVE) | 2/30 | 18/30 | 14/30 |
| Frozen Session 2 substantive | 21/30 | 21/30 | 21/30 |
| Available paddle measurements | 2/30 | 1/30 | 14/30 |
| Same status as frozen | 4/30 | 16/30 | 10/30 |
| Exact full feedback-text matches | 2/30 | 2/30 | 1/30 |
| Median / max detected-end to UI | 26 / 397 ms | 126 / 423 ms | 5453 / 6871 ms |

New status counts: 13 CORRECTIVE, 1 POSITIVE (C008), 2 LOW_CONFIDENCE, 2
INSUFFICIENT_DATA, 12 NO_OUTPUT. Ten paddle OPTIMAL, four OUTSIDE_BASELINE.

## Interpretation

The intervention did what it claimed: measured paddles went from 1/30 to 14/30.
The full-res pass logs `measured=1` on the recovered clips (e.g. C002); the 640x360
live worker still measures almost nothing by itself. Resolution, not scheduling,
is the paddle recall bottleneck.

Three costs, all expected except the third:

1. End-to-UI now includes the second pass (median 5.5s vs 0.1s). Speed was
   explicitly traded for recall.
2. Fuller evidence changes verdicts (same-status 16 -> 10, e.g. C005 PARTIAL ->
   POSITIVE-class output when the paddle measures). Coach C ratings attach to the
   frozen messages, not to these outputs; they cannot be transferred.
3. Availability dropped 27 -> 18, and the 9 newly-missing outputs skew mid/late
   run (C011-C030 region). The pass cannot cause this directly (it runs after
   serve end), but sustained max load — GPU pose plus denser paddle submits plus
   a 4-7s full-res pass per serve over 30 back-to-back runs — plausibly throttles
   pose throughput and starves the gate. Disentangling heat from scheduling load
   needs a rerun with cooldowns and a spacing-only arm.

Recording-category agreement and exact-match counts are diagnostic only: recording
categories are not independent Coach C subtype labels, and his Valid ratings do
not transfer to new messages. All 30 clips stay in the denominator.

## Per-clip status (new / gpu / cpu)

C001 NO/NO/LOW · C002 CORR/CORR/INSUF · C003 NO/NO/NO · C004 CORR/CORR/INSUF ·
C005 NO/PARTIAL/PARTIAL · C006 CORR/CORR/INSUF · C007 CORR/LOW/INSUF ·
C008 POS/PARTIAL/PARTIAL · C009 CORR/CORR/INSUF · C010 CORR/CORR/INSUF ·
C011 NO/INSUF/INSUF · C012 NO/CORR/INSUF · C013 INSUF/CORR/INSUF ·
C014 NO/LOW/INSUF · C015 INSUF/INSUF/CORR · C016 NO/CORR/INSUF ·
C017 CORR/CORR/INSUF · C018 NO/CORR/INSUF · C019 LOW/LOW/INSUF ·
C020 CORR/CORR/INSUF · C021 CORR/CORR/INSUF · C022 LOW/INSUF/INSUF ·
C023 NO/CORR/LOW · C024 NO/CORR/INSUF · C025 CORR/CORR/INSUF ·
C026 NO/PARTIAL/INSUF · C027 CORR/CORR/CORR · C028 CORR/NO/INSUF ·
C029 CORR/CORR/INSUF · C030 NO/CORR/INSUF.

New paddle-measured clips: C002, C004, C006-C010, C017, C019, C022, C025, C027-C029.

## Artifacts

`protocol.json` (APK/source identities), per-clip `logs/` (raw app output),
`clip_results.csv` (displayed feedback alongside frozen messages), `results.json`
(all events). Summarized with `scripts/summarize_coach_c_mobile_live.py`.
Raw logs in `logs/`. See `PROTOCOL.md` for the exact run method.
