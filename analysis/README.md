# Research report index

For the current workflow, read [the partner handoff](../docs/PARTNER_HANDOFF.md)
and [Coach C evidence](../evaluation/coach_c/README.md) first.
The [current thesis revision](../docs/thesis/README.md) supersedes the historical
chapter draft listed below.

| Report | What it establishes |
|---|---|
| [Pose/classifier audit](POSE_CLASSIFIER_AUDIT.md) | Legacy preprocessing, pose extraction and reporting limitations |
| [GRU retraining v2](GRU_RETRAINING_V2.md) | 90.9% development CV, but no improved diagnostic accuracy versus the matched old baseline |
| [Hybrid audit](HYBRID_CLASSIFIER_AUDIT.md) | Android-style GRU/kNN fusion, disagreements and reduced feedback coverage |
| [Four follow-up study](FOUR_FOLLOWUP_STUDY.md) | Raw-video augmentation, richer inputs and participant-held-out comparisons |
| [Refinement v4](SERVE_REFINEMENT_V4.md) | Windowing and fusion experiments; separate research candidates |
| [Coaching development](COACHING_DEVELOPMENT_V1.md) | Measurement checks and rejected calibration experiment; no proven coaching gain |
| [Chapters 5–6 draft](THESIS_CHAPTERS_5_6_DRAFT.md) | Historical working draft requiring the current protocol and Coach C results to be incorporated |

Dates and populations matter. Do not compare cached Heavy-pose scores directly
with fresh Lite-pose scores as a model improvement. Do not substitute diagnostic,
training, accepted-only, or development-CV accuracy for independent evaluation.
The `.docx` draft is also historical. Numeric scripts, plots and feature caches in
this directory support the named studies; they are not separate final results.
