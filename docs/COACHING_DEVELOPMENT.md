# Coaching development v1

This opt-in development version implements the six coaching improvements approved
on 10 October 2026. It preserves the original hybrid model, rules, assets, and
Coach C's Session 1/2 evaluation. It is not a replacement result for that study.

## Changes and evidence boundaries

1. Describe hip displacement as measured hip motion; do not infer forward weight
   transfer, ground forces, or arm reliance from image-plane displacement.
2. Check landmark visibility, continuity, and contact-proxy plausibility. Remove
   the arbitrary 70%-of-clip fallback in the development contact estimator. Use
   axial circular statistics for paddle angles. A motion peak remains a proxy,
   never a verified paddle–ball impact. Face-direction advice remains disabled.
3. Name the actual missing/uncertain signal and supply capture guidance only when
   tracking quality establishes a visibility problem. Preserve the 0.60 gate.
4. Use the existing coach-created form to judge overall feedback appropriateness.
   The model estimates its signals automatically; Coach C is not asked to enter
   contact frames, angles, or individual mechanics. Unsupported diagnoses stay disabled.
5. Assess calibration with saved, explicitly separated nested training folds.
   Do not transplant v4 policies to the old Android hybrid. Keep policies out of
   runtime unless matched-model, independently evaluated evidence supports them.
6. Use short measurement-bound observations and proposed coaching prompts, with
   rule provenance and review status. New wording is awaiting expert review.

## Implementation and verification

- `desktop/coaching_development.py`: opt-in measurements, guarded feedback and
  development-only pipeline adapter. The existing desktop/Android paths stay intact.
- `scripts/build_coaching_development.py`: training-only audit, reference assets,
  cached nested calibration experiment and real development
  smoke runs. Evaluation clip IDs and hashes are excluded.
- `desktop/test_coaching_development.py`: visibility failure, temporal gaps,
  stationary/early motion peaks, angle wraparound, missing modalities, confidence
  gates, unsupported diagnoses, leakage checks, and model-contract isolation.

The builder verifies the frozen Coach C code/assets and Session 1 artifacts before
and after its work. Returned Coach C annotations are not read. All calibration
metrics are development measurements, not independent Coach C results.

New expert labels and a later independent evaluation are still required to claim
better contact accuracy, physical paddle-face diagnoses, or validated coaching.
Additional mechanics-level validation is future work, not a prerequisite imposed
on Coach C for the current thesis review. No new coach form is generated.
Desktop processing time is not an Android latency measurement. No current
classification accuracy improvement is promised by rewriting feedback.

## Running and review

Run from the repository root:

```powershell
python -m unittest desktop.test_coaching_development desktop.test_desktop desktop.test_hybrid
python scripts/build_coaching_development.py
python scripts/build_coaching_development.py --smoke 4
```

Alternatively use `launchers/Coaching Development.bat`. The default audits the
training partition without repeating video inference.
`--smoke 4` checks four real training videos, covering all three training players.

Use the existing overall-feedback form for the independent coach review.
The software supplies the clip and feedback string; raw measurements remain
researcher-side. This workflow requests no additional frame/angle entries.

The scripts create `models/coaching_development_v1/assets.json` and private
audit reports under `outputs/coaching_development_v1/`. Expert measurement labels
are not generated automatically. `pending_mechanics` explicitly records the
diagnoses that remain disabled. The adapter has not been integrated into or
validated on the physical Android phone.
