# Session 2 scoring plan — fixed before receipt of ratings

No Session 2 results have been supplied. This is an analysis plan, not a result.
The selected model is now the original GRU ensemble; source reports are in
../session2_gru/reports. The earlier hybrid reports in this folder are historical.
The evaluated presentation is the current Apps Script: predicted subtype followed
by approved wording-v2 feedback, with uncertain classifications marked.
Record the actual Google form version and session dates when collecting responses.
The researchers plan a one-day interval after Session 1; elapsed time alone does
not eliminate recall or establish blinded assessment.

## Run after receiving the Google Forms export

```powershell
python scripts/score_coach_c_session2.py "C:/path/to/Session 2 (Responses).xlsx"
```

CSV exports work with the same command. The script also accepts a long table with
`review_id,feedback_validity`. For multiple submissions, explicitly choose the
response row, for example `--response-row 2`. For multiple matching worksheets,
specify `--sheet "Form Responses 1"`. No responses are silently combined or selected.

The output folder under `outputs/coach_c_session2_scores/` contains `REPORT.md`,
`scores.json` and `clip_scores.csv`. `--out` can set a new destination; an existing
folder is rejected. Original exports and frozen model reports are not modified.

## Definitions

Let V = Valid, I = Not valid, U = No coaching feedback to assess, and M = Missing.
Every clip C001–C030 is accounted for, so V + I + U + M = 30.

| Measure | Formula | Interpretation |
|---|---|---|
| Primary full-set validity | 100 × V / 30 | Final only after all 30 responses are present |
| Secondary validity among assessed outputs | 100 × V / (V + I) | Unavailable if V + I = 0; never substitute for the primary result |
| No-coaching assessment percentage | 100 × U / 30 | Explicit coach choice, not a missing response |
| Missing percentage | 100 × M / 30 | Incomplete collection, not an invalid or no-coaching judgment |
| Frozen substantive coaching coverage | 100 × 21 / 30 = 70.0% | 19 corrective + two positive output; independent of coach ratings |

The script retains every original response and flags, without changing it, a
Valid/Not valid judgment on a notice-only output or a no-coaching judgment on a
substantive output. Resolve interpretation through documented clarification, never
automatic recoding. This instrument does not separately assess whether withholding
feedback was appropriate. A serve-type prediction alone is not coaching advice.

Report counts and denominators alongside percentages. No Session 2 pass threshold
has been introduced. Do not claim a significant improvement, independent subtype
accuracy, inter-rater reliability, or a binary whole-form accuracy from these ratings.
Compare the two sessions descriptively without treating Session 1 comments as
Session 2 answers. Keep the evaluated model, gate and wording fixed.

Verification uses synthetic fixtures only:

```powershell
python -m unittest discover -s scripts -p test_score_coach_c_session2.py
```
