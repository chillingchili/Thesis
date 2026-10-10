# Coaching development v1 — 10 October 2026

The development version improves measurement safeguards and feedback wording.
It does not establish better classification, physical-mechanics diagnosis, or
independently validated coaching. The original Coach C evaluation remains frozen.

## Implemented changes

- Hip feedback now describes the image-plane measurement and proposes reviewing
  weight transfer with a coach. It does not assert forward transfer or arm reliance.
- Hip and serve-quality checks use finite, visible, in-frame landmarks. Hip
  smoothing never joins visibility gaps. Inconsistent body-height references fail.
- The contact candidate uses adjacent visible wrist samples, timestamps, aspect
  correction and median-filtered speeds. A stationary/early/boundary motion peak
  becomes unavailable rather than an arbitrary 70%-of-clip contact frame.
- Paddle-angle aggregation uses 180-degree axial circular statistics. A 179/1
  degree pair no longer creates a 90-degree median. Sparse or inconsistent angles
  become unavailable. No paddle-face direction is inferred from the long axis.
- Messages distinguish poor tracking, uncertain subtype and missing measurements.
  Capture guidance is conditional on actual tracking evidence. The 0.60 gate stays.
- Measurement-bound prompts carry source clip IDs and pending expert-review status.
  Reference agreement never becomes a whole-serve good-form diagnosis.
- The existing coach-created form is used to judge overall feedback appropriateness.
  No extra contact-frame, angle or mechanics entries are required from Coach C.

## Training-only audit

The audit reads the 210 Coach B annotated training records and corresponding fresh
Lite pose caches. It verifies source video hashes and preprocessing fingerprints.
The 30 Coach C videos are excluded by original IDs and SHA-256, including renamed
copies. No returned Session 1/2 responses are opened.

| Audit result | Count |
|---|---:|
| Training clips examined | 210 |
| Pose tracking passes the development quality gate | 209 |
| Hip-motion measurement available | 209 |
| Wrist-motion contact candidate available | 206 |
| Contact candidate unavailable | 4 |
| Eligible good-form Coach A hip-reference clips | 60 |

The visible hip-motion reference p10 is **0.161224**. It differs from the old
reference because it uses the fresh Lite cache and visibility/continuity checks.
It is a motion-reference threshold, not a validated physical transfer threshold.

None of the subtype-specific cached paddle reference pools met the fixed
development selectivity criteria (at least five usable observations and an axial
p95 radius no greater than 45 degrees). Directional feedback and reference-based
paddle prompts therefore remain disabled. Cached detector-weight provenance and
true contact-frame labels are also insufficient for an accuracy claim.

## Calibration experiment

The experiment uses the saved v4 angles+augmentation candidate's nested training
folds. Each inner/outer model and kNN bank excludes its held-out player. Membership,
model hashes, feature hashes and saved outer predictions are checked. Temperature
is fitted on inner predictions with equal player weighting; outer predictions are
used for assessment only. Fusion remains 50/50 and the gate remains 0.60.

| Participant-held-out development metric | Raw fixed hybrid | Nested temperature only |
|---|---:|---:|
| Classification accuracy | 57.9% | 57.9% |
| Player-balanced log loss | 1.430 | 1.185 |
| Brier score | 0.655 | 0.691 |
| Expected calibration error | 0.162 | 0.266 |
| Coverage at 0.60 | 60.8% | 19.1% |
| Accepted-case accuracy | 59.8% | 35.0% |

Log loss improved, but the other probability/acceptance measures did not support
adoption. Temperature preserved every predicted class; it did not improve
recognition. No policy was deployed. These are results for the v4 candidate,
whose preprocessing/model differ from the old Android ensemble; they cannot
certify or calibrate that ensemble. A matched nested OOF calibration dataset for
the current hybrid remains necessary.

## Coach review and evaluation boundaries

The model computes its internal measurements from the videos. Coach C reviews
whether the overall feedback is appropriate using the existing coach-created form.
No extra raw-measurement labeling form is part of this workflow. Unsupported
mechanics remain future validation topics, not additional tasks required from
Coach C to complete the current thesis evaluation.

The existing 30-clip packet, reports, model/rule assets and source files are
hash-verified before and after development. Do not replace its feedback strings
or present a later tuned version as that frozen evaluation. The thesis must also
disclose the hybrid configuration, early private inference and prior diagnostic
exposure of the 24 beginner clips. No phone latency or generalization gain is
established by these desktop development checks.

## Inspectable artifacts

- [Development specification and commands](../docs/COACHING_DEVELOPMENT.md)
- [Measurement summary](../outputs/coaching_development_v1/measurement_summary.json)
- [Per-clip measurement audit](../outputs/coaching_development_v1/training_measurement_audit.csv)
- [Calibration audit and fold provenance](../outputs/coaching_development_v1/calibration_audit.json)
- [Development assets](../models/coaching_development_v1/assets.json)
- [Completion record](../outputs/coaching_development_v1/completion.json)
- [Final-thesis alignment checklist](../docs/THESIS_EVALUATION_ALIGNMENT.md)
