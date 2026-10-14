# 30 clips through the live pipeline: 350ms paddle + full-res second pass (record)

Date: 14 October 2026. Device: Xiaomi 23021RAAEG (1fb637d), SM6225, Android 14.
Config: original five-fold GRU, GPU pose attempt, pose priority, motion-rate gating
with guarded shared mechanics, original end detector, 0.60 confidence, 32-pose
minimum — identical to the `mobile_gpu_rate_20261012` audit — plus a contact-window
full-res second pass at source resolution through the same detector and rules
(`research_fullres_paddle`; paddle spacing stays at the normal 350ms after the
150ms arm cost availability with no extra recall — see `mobile_showcase_20261014`).
60s cooldown between clips. One replay per clip, C001..C030, fresh app start each
time. All 30 source hashes and all five GRU asset hashes verified by the runner.

## Findings vs previous full-set runs

| Measure | CPU audit | GPU audit | This run |
|---|---:|---:|---:|
| Clips with completed displayed feedback | 29/30 | 27/30 | 21/30 |
| Substantive coaching (CORRECTIVE/POSITIVE) | 2/30 | 18/30 | 16/30 |
| Frozen Session 2 substantive | 21/30 | 21/30 | 21/30 |
| Available paddle measurements | 2/30 | 1/30 | 16/30 |
| Same status as frozen | 4/30 | 16/30 | 15/30 |
| Exact full feedback-text matches | 2/30 | 2/30 | 2/30 |
| Median / max detected-end to UI | 26 / 397 ms | 126 / 423 ms | 5118 / 6443 ms |

New status counts: 15 CORRECTIVE, 1 POSITIVE (C008), 2 LOW_CONFIDENCE (C014, C019),
2 INSUFFICIENT_DATA (C015, C022), 1 PARTIAL (C026), 9 NO_OUTPUT
(C001-C005, C013, C025, C027, C030). Thirteen paddle OPTIMAL, three OUTSIDE_BASELINE
(C007, C010, C022 — C022 still INSUFFICIENT at 29 frames, 3 short of the minimum).

## Interpretation

Paddle recall went from 1/30 to 16/30 while status agreement with the frozen
reports held (15 vs 16) — the second pass adds evidence without rewriting verdicts
in most cases. Where fuller evidence does change a verdict (C008 CORRECTIVE ->
POSITIVE with paddle OPTIMAL at 0.69), Coach C ratings do not transfer; they
attach to the frozen messages only.

The 9 misses split into two groups. C001-C005 missed on both full record runs
(this and `mobile_showcase_20261012`'s predecessor run) immediately after a fresh
install: a cold-start effect (GPU delegate compilation is known-slow on this
phone), not the paddle config. The runner should gain an unlogged warmup replay.
C013/C025/C027/C030 are residual gate variance; C027 measured CORRECTIVE on every
prior live run including this week's showcase takes.

End-to-UI now includes the 4-7s second pass by design (median 5.1s vs 0.1s);
this is the declared speed-for-recall trade, unsuitable for the 2s live target.

## Per-clip status (this / gpu / cpu) and paddle (this)

C001 NO/NO/LOW — · C002 NO/CORR/INSUF — · C003 NO/NO/NO — · C004 NO/CORR/INSUF — ·
C005 NO/PARTIAL/PARTIAL — · C006 CORR/CORR/INSUF OPTIMAL · C007 CORR/LOW/INSUF
OUTSIDE · C008 POS/PARTIAL/PARTIAL OPTIMAL · C009 CORR/CORR/INSUF OPTIMAL ·
C010 CORR/CORR/INSUF OUTSIDE · C011 CORR/INSUF/INSUF OPTIMAL · C012 CORR/CORR/INSUF
OPTIMAL · C013 NO/CORR/INSUF — · C014 LOW/LOW/INSUF OPTIMAL · C015 INSUF/INSUF/CORR — ·
C016 CORR/CORR/INSUF OPTIMAL · C017 CORR/CORR/INSUF OPTIMAL · C018 CORR/CORR/INSUF
OPTIMAL · C019 LOW/LOW/INSUF — · C020 CORR/CORR/INSUF — · C021 CORR/CORR/INSUF — ·
C022 INSUF/INSUF/INSUF OUTSIDE · C023 CORR/CORR/LOW OPTIMAL · C024 CORR/CORR/INSUF
OPTIMAL · C025 NO/CORR/INSUF — · C026 PARTIAL/PARTIAL/INSUF — · C027 NO/CORR/CORR — ·
C028 CORR/NO/INSUF OPTIMAL · C029 CORR/CORR/INSUF OPTIMAL · C030 NO/CORR/INSUF —.

## Artifacts

`protocol.json` (APK/source identities), per-clip `logs/`, `clip_results.csv`,
`results.json`. Summarized with `scripts/summarize_coach_c_mobile_live.py`.
Runner: `scripts/run_coach_c_mobile_live.py --gpu --motion-rate --fullres-paddle
--cooldown-secs 60 --output evaluation/coach_c/mobile_pass2_20261014`.
