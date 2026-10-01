# Rule-based feedback implementation

Implemented from thesis sections 4.2.1, 4.3.2, 4.5.1 (FR6/FR7, NFR3),
4.5.2 and 4.10.1. This is a research implementation awaiting Coach C's
independent validation, not a claim of validated coaching performance.

## Runtime

`android/app/src/main/java/com/thesis/pickleballserve/FeedbackEngine.kt` is a
deterministic evaluator. The hand-authored rule table is
`android/app/src/main/assets/feedback_rules.json`. Each rule carries its
condition, applicable subtypes, fixed text and source clip IDs. Table order
determines message order (weight shift, then paddle); all matching cues are
joined into one result. The thesis does not specify severity ranking, so no
ranking is inferred from category frequency. No model generates advice.

At serve completion, MainActivity passes the **final GRU** subtype and maximum
probability, independent shift verdict and contact-window paddle state to the
engine before clearing buffers. The final result display uses that same GRU
prediction. Hybrid/kNN and EMA smoothing remain live preview options; neither
can override the thesis's GRU confidence gate. Changing mode resets capture.

- GRU confidence <0.60: withhold all corrective and positive feedback. Retain
  independent measurement outputs and name absent signals.
- Missing/invalid GRU result: withhold coaching; keep independent measurements.
  This conservatively resolves FR7 against the mandatory confidence gate.
- Missing shift or paddle: apply only rules supported by the surviving signal.
- Both measured mechanics acceptable: scoped encouragement, never a whole-serve
  good-form diagnosis. A subtype prediction does not classify form quality.
- No matching supported rule: explicitly say overall form was not assessed.
- Feedback is held after completion and cleared at the next capture/reset.
- Paddle detection exceptions invalidate that stream for the current serve;
  classification and weight-shift processing continue.
- A scrollable results overlay accommodates combined cues and missing signals.

The engine accepts the seven thesis placement zones with a separate
`placementActive` flag. An active missing/invalid zone is reported unavailable;
a fault prevents the all-acceptable encouragement rule. Placement is reported
as an observation, without inferring a biomechanical cause. The Android landing
screen processes a separate clip, so its last result is **not** fused into the
live serve. A synchronized secondary-camera integration must pass a result for
the same serve before enabling this input.

## Encoded rules and evidence

| Rule | Trigger | Coach B evidence |
|---|---|---|
| C4_WEIGHT_TRANSFER | Insufficient hip displacement; any supported subtype | Beginner1_Lob_016: minimal transfer; Beginner2_Drive_019: no transfer; Beginner2_Topspin_022: add transfer |
| C6_PADDLE_REVIEW | Contact angle outside annotated baseline; any supported subtype | Beginner2_Drive_010 and Beginner1_Lob_019: upward face; Beginner2_Topspin_002: closed face |
| C6_TOO_OPEN | Verified too-open state; Drive/Lob only | Beginner2_Drive_010, Beginner1_Lob_019 |
| C6_TOO_CLOSED | Verified too-closed state; Topspin only | Beginner2_Topspin_002, Beginner2_Topspin_004 |
| ASSESSED_MECHANICS_OK | All required assessed states acceptable | Thesis section 4.5.2 positive-confirmation requirement; researcher-authored wording |

Directional rules are encoded for review but **inactive with the bundled
configuration**: `paddle_config.json` sets `direction_validated: false`.
An image-plane long-axis angle is not a verified measurement of face tilt/yaw.
Until directional calibration is independently checked, the adapter produces
`OUTSIDE_BASELINE`, and the UI says outside/within reference range. Even the
neutral paddle cue and positive wording require Coach C validation.

The engine excludes toss placement, wrist lag, follow-through and swing-path
diagnoses because those states are not supplied by the current runtime.
The existing shift metric supports only part of C4; it does not independently
measure knee bend, stance, balance or ground reaction force. No legality rules
are implemented (thesis scope section 1.4).

## Annotation audit and calibration

`annotation_audit.json` records the workbook hash, counts and eligible baseline
IDs. The source workbook is never modified. Its 210 rows contain 77 Good Form
and 133 Bad Form judgments, including 60 good and 12 bad Coach A clips.
Category flags count 26/25/22/38/23/11 for C1 through C6. Twenty-eight bad-form
rows have no category flagged. Some flags refer to positively described aspects
inside a bad-form clip (for example Beginner1_Drive_023's weight transfer).
Consequently the flags are audit aids, not per-category runtime ground truth.

Both derivation scripts now join the explicit CoachA Good Form rows against the
training manifest. They do not fit on evaluation clips. Each asset records the
workbook SHA-256, selected IDs and excluded IDs, so calibration is reproducible.
Existing percentile methods are retained; no threshold search against Coach C
or the evaluation set is performed.

| Calibration | Previous | Annotated training baseline |
|---|---|---|
| Shift p10 | 142 assumed-good clips, 0.1438 | 59 usable clips, approximately 0.154266 |
| Paddle p5–p95 | 147 assumed-good clips, 10.39–162.17 degrees | 52 contact-window clips, approximately 13.4105–165.7387 degrees |

Shift excludes CoachA_Lob_019 for missing/invalid keypoints. Paddle excludes
eight eligible clips without a usable contact-window scalar; IDs are recorded
in its config. Runtime feedback also rejects the full-series paddle fallback
so that a backswing angle cannot stand in for an unavailable contact angle.
The broad paddle band and underlying direction ambiguity remain measurement
limitations; these updates do not establish detection accuracy.

Reproduce from the project root:

```powershell
python scripts/coach_annotations.py
python scripts/derive_shift_threshold.py
python scripts/derive_paddle_threshold.py
python scripts/test_feedback_provenance.py
```

Android checks: `:app:testDebugUnitTest`, `:app:assembleDebug`, `:app:lintDebug`
using Gradle 8.9 with JDK 17 or 21. The unit tests exercise confidence boundaries,
missing/invalid signals, deterministic multiple-rule results, subtype restrictions,
placement availability, scoped encouragement, and contact-only paddle evidence.
The Python checks verify asset provenance against the actual workbook.

## Independent evaluation still required

1. Freeze the rule version, threshold assets and model version before evaluation.
2. Coach C Session 1: independently label the 30 evaluation videos without any
   system output. Keep these labels separate from rule construction.
3. Run each evaluation video through the pipeline and retain its actual fused
   input, rule IDs, rule version, final feedback and abstention status. Debug
   Logcat emits rule version/status/IDs; a full research record must also retain
   clip ID and the measurements. A live replay is not a substitute for the
   controlled pre-recorded evaluation protocol.
4. Session 2: show the same videos and resulting feedback only, without Session 1
   labels, and collect appropriateness judgments and comments. Do not substitute
   hand-entered category flags for runtime detections.
5. Report appropriate judgments out of 30 as specified by the thesis, with
   abstentions/missing-data cases explicitly accounted for. Preserve the first
   evaluation when refining rules, rather than presenting tuning results as
   untouched held-out performance.
6. Validate every deployed wording, including encouragement. Change the
   validation metadata only when supported by documented coach review.

Phone rendering, full-video evaluation, independent coaching validation and the
end-to-end two-second latency target require separate measurements; passing the
logic tests establishes none of those outcomes.
