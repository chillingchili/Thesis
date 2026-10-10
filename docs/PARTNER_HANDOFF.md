# Partner handoff — 10 October 2026

Start here for the current state. Older reports document earlier experiments and
may describe a protocol or status that has since changed.

## What is being evaluated

The current evaluation uses the original five-fold GRU ensemble plus kNN5,
combined 50/50, with feedback withheld below 60% fused confidence. Hip movement
and image-plane paddle orientation feed deterministic feedback rules. Ball landing
is disabled for this 30-clip evaluation. The separate retrained GRU and development
measurement adapter are not the evaluated system.

The selected set contains 12 Beginner 3, 12 Beginner 4 and six fresh Coach A clips,
with ten per recorded category. The 24 beginner clips have prior diagnostic exposure.
Do not describe this as an untouched final test or all-new-player evaluation.

## Current progress

| Item | State |
|---|---|
| 30 selected videos and original inference | Complete and frozen |
| Session 1 form judgments | Received: 27 Bad Form, 3 Good Form (C008, C019, C022) |
| Independent serve-type labels | Intentionally omitted from the actual coach instrument |
| Independent Coach C classification accuracy | Unavailable under that revised protocol |
| Revised feedback wording | Prepared before reading Coach C responses; no prediction/threshold changes |
| Session 2 Google Apps Script | Ready locally; deployment and collection in Google must be confirmed by researchers |
| Session 2 ratings | Not yet received in this repository |
| Phone latency / live accuracy | Physical-device validation still pending |

Session 1 is complete under the researcher's clarified scope. The earlier request
for missing subtype labels and the corresponding HOLD status were superseded.
The researcher reports that this label question was removed starting with Coach B;
reconcile the actual training-label provenance before final manuscript claims.

## What to use now

1. [Session 2 Apps Script](../scripts/google_apps_script/coach_c_session2.gs) is the
   current form implementation. Follow [its instructions](../scripts/google_apps_script/README.md).
   Existing form: replace the script and run `updateCoachCSession2Display()`.
   New form: run `createCoachCSession2Form()` three times.
2. Titles contain clip IDs only. Each output shows the actual frozen prediction
   followed by the approved feedback; low-confidence predictions are marked.
   Choices are Valid, Not valid, and No coaching feedback to assess.
3. [Coach C evidence folder](../evaluation/coach_c/README.md) contains the workbook,
   normalized comments, original and revised reports, and one copy of each video.
4. [Alignment notes](THESIS_EVALUATION_ALIGNMENT.md) identify manuscript changes.
   [Report index](../analysis/README.md) explains which results answer which questions.

The historical ZIP uses feedback-only presentation. The latest Apps Script adds
classification to the output block. Use the Apps Script for the latest requested
Session 2 instrument and record which version the coach actually sees.

## How to interpret results

The original GRU's reported stratified CV is 86.7%; the separate retrained GRU's
is 90.9%. The comparable fresh-Lite diagnostic accuracies are 51.4% for original
GRU, 46.9% for retrained GRU and 49.7% for the original hybrid. These concern
different evaluation conditions, not contradictory measurements of one test.
Three training participants limit observed variation; augmentation does not add
independent players. CV success does not establish a reliable coach replacement.

All 30 evaluation clips produced classifications; coaching availability is weaker:
16 corrective, one positive, 12 low-confidence and one partial output. Thus
corrective-or-positive coaching coverage is 17/30 (56.7%). Do not call this accuracy.
All three Coach C Good Form clips received low-confidence messages. The 16 corrective
cases were judged Bad Form, but only Session 2 can assess the specific advice.

Report all three Session 2 response counts out of 30. Report Valid/30, and label
any assessed-only percentage with its separate denominator. Do not count absence
of coaching as valid advice or silently discard those clips. Form judgments are
not serve-type labels, and model feedback status is not a validated whole-form classifier.

## Remaining work

- Preview the actual Google form and complete Session 2. Do not use Session 1
  comments to tune this frozen model or rewrite its cues during evaluation.
- Import Session 2 ratings keyed by C001–C030, then report appropriateness and coverage.
- Update Chapters 4–6 to the actual instrument, hybrid configuration, timing change,
  label sources and evaluation limits. The existing chapter drafts are historical
  working drafts, not submission-ready conclusions.
- Measure Android latency and live behavior on the intended phone.

## Repository versus local workspace

`evaluation/coach_c/` is the tracked handoff copy; `outputs/` retains the original
local run history. Source paths inside frozen JSON describe that original layout.
`provenance.json` maps the handoff copies to their original files and hashes.
Regenerate the handoff with `python scripts/export_coach_c_handoff.py` only on the
researcher's full workspace. It does not rerun inference or change responses.

Repeated ZIPs, cached poses, previews, logs, build products and detector datasets
remain local. Detector datasets previously exceeded the Git LFS budget; ask for
the dataset transfer separately when training those models. A clone contains the
30 evaluation videos and the small evidence needed to understand the current review.

Run checks from the repository root:

```powershell
python -m unittest desktop.test_coaching_development desktop.test_desktop desktop.test_hybrid
node scripts/google_apps_script/test_coach_c_session2.cjs
```

The Apps Script checks use mocks; they do not establish that the form was deployed
or that every YouTube upload is playable. No messages are sent by these checks.
