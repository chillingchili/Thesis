# Current thesis working revision — 10 October 2026

Use [Pascua-Leones_Thesis_Current.md](Pascua-Leones_Thesis_Current.md) for the
full working manuscript. It is based on the supplied **Pascua-Leones Thesis (5).md**.
The [original copy](source/Pascua-Leones_Thesis_2026-10-10.md) is preserved byte for
byte. This is a working revision, not a claim that remaining evaluation is complete.

## Current model amendment

The selected configuration is now the original GRU ensemble. The researcher
authorized the switch before Session 2 release, after Session 1 receipt.
[The amendment and new counts](../../evaluation/coach_c/session2_gru/README.md)
supersede the earlier hybrid selection. All chapters and diagrams now distinguish
the selected GRU from the archived hybrid comparison.

## What changed

- The abstract and Chapters 1–3 now identify the original GRU and hybrid comparison, distinguish it
  from multimodal measurement integration, and align objectives and limitations
  with the actual instruments. GRU-only scores remain labeled as GRU results.
- Figures 4–7 and 12 use editable Mermaid diagrams; Figures 8–10 specify output
  states without unsupported whole-form verdicts. Result tables are now 7–12,
  avoiding duplicate numbering. The contents and figure/table lists were refreshed.
- Chapter 4 now distinguishes the proposed design from the selected original GRU, limited measurement rules, actual coach instruments and timing amendment.
- New [Chapters 5–6](CHAPTERS_5_6.md) report the completed experiments, all 30 frozen
  outputs, and Session 1's 27 Bad Form / three Good Form judgments.
- Session 2 counts and physical-phone timing remain pending. The planned one-day
  interval is recorded as a plan, not an already completed delay or guarantee against bias.
- The scoring definitions retain all 30 clips and distinguish no assessable coaching
  from unanswered questions. See the [scoring plan](../../evaluation/coach_c/session2/SCORING_PLAN.md).
- The classifier changed from hybrid to original GRU. Measurements, confidence
  threshold, cue templates, clip membership and coach responses remain unchanged.

## Evidence behind the findings

| Material | Repository source |
|---|---|
| Legacy preprocessing and CV interpretation | [Pose/classifier audit](../../analysis/POSE_CLASSIFIER_AUDIT.md) |
| Retraining and matched old/new comparison | [GRU retraining v2](../../analysis/GRU_RETRAINING_V2.md) |
| Hybrid performance and withholding | [Hybrid audit](../../analysis/HYBRID_CLASSIFIER_AUDIT.md) |
| Augmentation and richer features | [Four follow-up study](../../analysis/FOUR_FOLLOWUP_STUDY.md) |
| Calibration audit | [Coaching development](../../analysis/COACHING_DEVELOPMENT_V1.md) |
| Selected recordings and frozen outputs | [Coach C evidence](../../evaluation/coach_c/README.md) |
| Original Session 1 answers | [Response workbook](../../evaluation/coach_c/session1/coach_c_responses.xlsx) |

## Remaining manuscript work

1. Import Session 2 after the coach submits it. Fill Table 12 and finalize the
   corresponding discussion and conclusions using actual ratings.
2. Record the actual session dates and deployed Google form version. The local
   script cannot verify what the coach was shown.
3. Measure physical-device latency and live behavior before claiming the two-second
   target or equivalent desktop/phone behavior.
4. Review the amended objectives and scope with the adviser. The numerical GRU
   target was not silently reassigned to the hybrid. Historical appendix invitations
   retain their original subtype-label requests, qualified in the main text.
5. Render the editable diagrams for the final Word/PDF, and generate page numbers
   after typesetting. The Markdown contents list intentionally has no stale page
   numbers. The literature citations were retained, not independently re-verified
   in this manuscript-consistency pass.
6. Reconcile the recorded-category and legacy label provenance before submission.
   Coach form judgments must not be described as independent subtype labels.

The earlier draft in `analysis/THESIS_CHAPTERS_5_6_DRAFT.md` is historical.
This revision supersedes it for current findings and protocol status.

## Revision trace

[revision_manifest.json](revision_manifest.json) records the source and working
hashes and the replaced source-line numbers. The bibliography, appendices and
embedded image definitions were retained. The builder is
[`scripts/revise_thesis_20261010.py`](../../scripts/revise_thesis_20261010.py).
The subsequent consistency pass is recorded by
[`scripts/align_thesis_manuscript.py`](../../scripts/align_thesis_manuscript.py).
Both tools protect the completed consistency pass from accidental rerunning.
Use the current manuscript for further editing; preserve manual changes before
attempting any reconstruction.
