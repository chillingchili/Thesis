# Active Session 2 — original GRU ensemble

The researcher selected the original five-fold GRU before sending Session 2.
The prior matched diagnostic comparison favored original GRU (51.4%) over hybrid
(49.7%) and retrained GRU (46.9%). This is a selection rationale, not proof of
statistically significant superiority or new independent test accuracy.

The switch occurred **after receipt and inspection of Session 1**. That exposure
must be disclosed. No model fitting, threshold adjustment, clip replacement or
coach-comment-driven cue editing was performed. The generator does not read
coach responses. Session 1 remains valid as a pre-output form review; Session 2
will evaluate this newly frozen version. Nothing has been sent automatically.

All 30 predictions were recomputed with the original ensemble from preserved pose
caches. Hip/paddle/contact measurements, the 0.60 gate and approved wording remain
the same. The Android and desktop defaults now select the original ensemble;
their live/recorded capture windows still differ, so phone parity and latency
remain unverified. Android feedback wording now matches the approved wording.

| Output | Original GRU | Earlier hybrid |
|---|---:|---:|
| Corrective | 19 | 16 |
| Positive on assessed measurements | 2 | 1 |
| Low confidence | 8 | 12 |
| Partial without supported cue | 1 | 1 |
| Substantive coaching coverage | 21/30 (70.0%) | 17/30 (56.7%) |

Coverage is not accuracy or coach-rated validity. The same missing paddle
measurements remain in C005, C020, C026 and C027. No clips were dropped.

Use [index.html](index.html) to preview the complete outputs with videos in
`../videos/`. Use the [current form script](../../../scripts/google_apps_script/coach_c_session2.gs)
to create a **new form**, with key `COACH_C_SESSION2_ORIGINAL_GRU_V1`. Build it
three times, preview it, and send only that new respondent link after the planned
interval. Do not send the older hybrid ZIP or old form link.

After submission, run `python scripts/score_coach_c_session2.py "responses.xlsx"`
from the repository root. It now defaults to these GRU reports. Three response
categories and all 30 clips remain in the accounting.

`freeze.json` records the inputs, inference code and assets; `summary.json`
records counts and the amendment. `feedback_strings.csv` includes the prior
hybrid status for comparison. Original evidence under `../baseline/` and
`../session2/` remains unchanged and is superseded for release, not erased.
