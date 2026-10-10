# Aligning the final thesis with the frozen implementation

The final manuscript must describe the actual experiment. The proposal is a
source of intended requirements, not evidence that each requirement was met.

## Primary Coach C evaluation

The frozen run selected before Coach C's responses uses the Android-compatible
five-GRU-fold ensemble combined 50/50 with distance-weighted kNN5. The feedback
gate applies to the fused probability at 0.60. The paper currently specifies a
GRU and its softmax gate; explain the hybrid extension and actual gate explicitly.
The legacy kNN bank has 438 records, which differs from the proposal's 210-clip
Coach B annotated training partition. Document actual training membership and
label sources. A GRU-only CV score does not establish the hybrid's CV accuracy.
The separate retrained GRU's development-CV score belongs to that separate model.

The evaluation comprises 12 Beginner 3, 12 Beginner 4 and six fresh Coach A
recordings. The 24 beginner clips were previously inspected during diagnostics;
this is retrospective independent coach review, not a pristine sealed test.
Fresh Coach A recordings are new clips from an already represented participant.

The user authorized early private inference while Session 1 was underway. Its
timing change is recorded in the freeze and status records. No returned coach
responses were used for inference. Session 1 has now been submitted: 27 Bad Form
and three Good Form judgments. The researcher clarified that subtype labeling
was intentionally removed from the actual coach instrument starting with Coach B.
Record this protocol amendment and reconcile historical label provenance.

Session 2 is ready under that revised scope. The approved wording revision keeps
predictions, measurements and thresholds unchanged. The latest Google form shows
clip-only titles and the actual predicted subtype alongside the feedback, marking
low confidence. It asks Valid / Not valid / No coaching feedback to assess.
Session 1 answers are not shown. Google deployment and receipt of Session 2
ratings have not been verified in this workspace. The earlier feedback-only ZIP
and historical HOLD notices remain preserved as previous snapshots.

## Output and reporting scope

- All 30 clips produced classification and hip-motion results; this is availability,
  not proof of correct subtype or physical weight transfer.
- Paddle measurements were available for 26/30. The measured image-plane axis is
  not validated face tilt. Directional cues remain disabled.
- Feedback statuses were 16 corrective, one positive, 12 low-confidence and one
  partial. Keep all 30 in the accounting and show withheld/missing cases.
- Independent Coach C classification accuracy is unavailable: no subtype labels
  were collected. Do not infer them from comments or treat recording categories
  as independent coach labels. Form judgments do not establish subtype accuracy.
- Report all three Session 2 response counts and Valid/30. Any assessed-only
  validity rate must state its separate denominator. No coaching feedback to
  assess is not Valid and must not disappear from the full-set denominator.
  Corrective-or-positive coaching coverage is 17/30 (56.7%).
- Describe hip displacement as a proxy, not measured force or verified forward
  transfer. Describe paddle reference membership without calling overall form good.
- The auxiliary landing stream was disabled because no paired second-camera
  recordings were supplied; it remains outside this core-stream batch.
- Desktop runtimes include playback preparation and do not establish the Android
  two-second target. Report hardware measurements separately when available.

## Development v1 is a separate result

The new measurement safeguards and calibration audit are separate development
work. They do not replace the original measurements or retrospectively improve
its score. The approved actionable wording was applied uniformly to the original
30 outputs before Coach C responses were read, with the original run preserved.
The latest requested form assesses overall feedback validity using that wording.
Future mechanics-level labels may support later refinements; those refinements
need another independent evaluation before a coaching-validity claim.

Calibration was not adopted: the training-only nested temperature experiment did
not improve recognition, and accepted-case accuracy at 0.60 declined. Additional
mechanics cues remain disabled pending measurements and expert labels. These
findings should be reported as limits and development outcomes, not omitted to
present an entirely successful model.

[Frozen primary run](../outputs/coach_c_evaluation_v2/private_session2_hybrid_20261010T045855Z/summary.json)
· [Development results](../analysis/COACHING_DEVELOPMENT_V1.md)
· [Thesis source](Pascua-Leones_Thesis.md)

For a GitHub clone, use the [tracked evidence](../evaluation/coach_c/README.md)
and [partner handoff](PARTNER_HANDOFF.md); the original `outputs/` paths are local.
