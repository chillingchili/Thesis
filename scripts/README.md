# Script guide

Run Python scripts from the repository root. Original script paths remain stable
because frozen evaluation manifests record their hashes and locations.

| Task | Script or folder |
|---|---|
| Build/update the current Coach C Google form | [google_apps_script/](google_apps_script/README.md) |
| Export tracked evidence for the partner | `export_coach_c_handoff.py` |
| Select/finish the original 30 clips | `prepare_coach_c_evaluation.py`, `finalize_coach_c_30.py` (historical preparation) |
| Original frozen inference | `prepare_coach_c_session2.py` (do not rerun to replace the evaluated snapshot) |
| Approved wording revision | `revise_coach_c_session2.py` (historical transformation of saved outputs) |
| Import the supplied Session 1 workbook | `import_coach_c_session1.py` (historical first import; its missing-label HOLD was superseded) |
| Development coaching audit | `build_coaching_development.py` (training clips only; separate from Coach C evaluation) |
| Research website generation | `build_research_site.py` (historical experiment exporter; review current evaluation text after regenerating) |
| Word draft export | `export_thesis_draft.py` |

Other `train_*`, `extract_*`, `evaluate_*` and `export_*` scripts are training,
feature extraction, study evaluation and model-export utilities. Follow the
corresponding report in [analysis/](../analysis/README.md) before running a study.

Do not rerun historical preparation/import scripts as a routine next step: they
can generate old protocol status. Current state is in
[the handoff](../docs/PARTNER_HANDOFF.md), with immutable evidence under
[`evaluation/coach_c/`](../evaluation/coach_c/README.md).
