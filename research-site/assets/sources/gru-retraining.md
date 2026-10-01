# GRU retraining result — 2026-10-01

Retrained the GRU using **209 Coach B annotated training clips only**. No Beginner 3/4 testing clip was passed to model fitting, early stopping, normalization fitting, confidence tuning, or checkpoint selection. BlazePose weights were not trained or modified.

The new five-fold ensemble scores **46.9% on the existing 175 diagnostic test clips**, versus 51.4% for the old desktop ensemble on the same freshly extracted Lite landmarks. This is the relevant desktop baseline; the earlier 38.3% audit used cached Heavy poses and the old training preprocessing, so it is a different comparison. **The retrain did not improve unseen-player accuracy; the old desktop model remains the default.**

## Dataset and separation

- Located all 210 annotated raw training videos, including the five absent from old feature caches.
- 209 passed the predeclared tracking quality gate (at least 32 source frames, at least 80% valid source and resampled frames; required landmark visibility ≥0.50).
- Excluded from fitting: CoachA_Drive_018. Its annotation and quality statistics remain in the frozen inventory; no label was changed.
- Training class counts in drive/lob/topspin order: [69, 70, 70].
- Froze video SHA-256 hashes, Coach B workbook hash, labels, player IDs and split membership in data/serve_v2/dataset.json. Extracted all clips with separate tracker instances.
- The existing 175 Beginner 3/4 clips are diagnostic testing data only. They were already inspected during earlier project experiments. Model weights were frozen before this run's test-label evaluation. No fitting code opens diagnostic_test.npz.
- The user confirmed that Coach C's independently annotated 30-clip set does not exist yet. Its evaluation remains pending; neither this model nor these 175 labels represent that final thesis evaluation. The actual 209 eligible training count also differs from the planned 210 usable clips and must be documented in the thesis.

## Pipeline corrections

1. Shared scripts/serve_sequence.py between training and desktop: the same Tasks VIDEO BlazePose Lite asset, settings, and fresh tracker per selected recording.
2. Resample the complete selected serve into 128 steps over elapsed time. Preserve both ends and normalize interpolated sin/cos pairs. Do not take only the first 128 frames or estimate contact to crop the classifier input. Clips must contain one serve from setup through follow-through, at most 12 seconds.
3. Retain visibility and time metadata. Mask unreliable required joints; do not interpolate through a missing source frame. Apply the same quality rejection in training and desktop.
4. Remove inference-only moment matching for the candidate. The model sees the same feature representation during training and inference.
5. Fix SequenceAugment so padding and dropped frames stay zero after sin/cos encoding and noise. Previously cos(atan2(0,0)) populated missing frames with artificial poses.
6. Reset the original legacy Heavy extractor before each new video too. Existing cached Heavy files and original model checkpoints were preserved.

## Training and validation

An initial training-only pilot used inner validation to select an epoch and refit each outer training fold. Its stopping point varied from 12 to 94 epochs; the early-stopped fold was undertrained. The final candidate uses a fixed 100 epochs, selected from this training-only development evidence. The two runs and their histories are preserved. Therefore the final stratified score is a **development CV score**, not a pristine independent estimate after hyperparameter selection. The diagnostic test was not used to choose between these schedules.

The final GRU has 32 units, dropout 0.15, L2 0.0001, batch size 16, and conservative training-only angle/noise/time/frame-drop augmentation. No horizontal flip is used. The current implementation augments extracted pose features; it does not implement the paper's brightness/contrast/crop augmentation of raw videos, and this remains a methodological difference to reconcile. Seeds, fold membership, training histories, and model hashes are saved. Each outer model is fitted on its training indices only; outer validation is never passed to fit in the final run.

| Evaluation | Initial nested pilot | Fixed 100-epoch candidate |
|---|---:|---:|
| Stratified five-fold training CV | 73.7% | 90.9% |
| Leave-one-training-player-out | 33.0% | 36.4% |

Final stratified fold accuracies: 88.1%, 88.1%, 92.9%, 97.6%, 87.8%.

The leave-one-player-out check trains three additional models. Every held-out player's clips are excluded from that model's fitting. These diagnostic models are not included in the deployment ensemble. A separate single GRU is fitted on all eligible training clips for 100 epochs; the primary ensemble was chosen before testing.

## Frozen-model diagnostic results

Scores below count quality rejection as an incorrect/unavailable result, rather than silently dropping difficult test clips.

| Model | Beginner 3 (85) | Beginner 4 (90) | All testing (175) |
|---|---:|---:|---:|
| Old desktop ensemble, fresh Lite poses | 43.5% | 58.9% | 51.4% |
| Retrained five-fold ensemble | 43.5% | 50.0% | 46.9% |
| Retrained single GRU | 29.4% | 37.8% | 33.7% |

The retrained ensemble accepts tracking quality on 174/175 clips. At confidence ≥0.60 its coverage is 82.3%, confident-only accuracy is 47.2%, and 76 accepted predictions are wrong. Confidence must not be interpreted as validated correctness.

### Previously reported examples

These are training clips, so this table is a regression illustration, not evidence of generalization to new players.

| Clip | Coach B label | Old desktop ensemble | Retrained ensemble |
|---|---|---|---|
| Beginner2_Drive_001 | drive | lob (69.1%) | drive (82.5%) |
| Beginner2_Drive_002 | drive | lob (86.3%) | topspin (58.6%) |

## Desktop use and deployment status

Analysis settings → Serve classifier → **Retrained GRU · research candidate** uses models/serve_v2_fixed/tflite. The old model options remain available for comparisons. Refresh/restart Serve Lab if it was already running. The candidate forces moment matching off, validates the preprocessing contract, and rejects poor tracking explicitly.

The candidate meets the 85% training development-CV threshold, but independent Coach C validation and physical Android evaluation remain pending.

Android still has its original model assets. Its asynchronous live stream and motion gate have not been validated against this complete-recording contract; copying the new files into Android without implementing and validating that path would recreate the mismatch. No Android latency or live accuracy improvement is claimed.

The retrain changes serve classification, not the repertoire of biomechanical diagnoses. Wrist-lag and early-wiper coaching rules still need validated measurements; this GRU does not learn those diagnoses from deviation comments.

The next evidence needed is additional independently annotated **training** players and an assessment of whether these five 2D angles preserve the differences between serve subtypes. Keep the current testing partition out of fitting. These results do not justify tuning on it or claiming that another epoch increase will solve the remaining player/domain gap.

## Verification and artifacts

- Exported all five fold models plus the single model. TFLite/Keras parity checked on 1260 real training windows: zero label disagreements, maximum probability difference 2.89082527e-06.
- Sequence tests cover late motion, endpoints, frame-rate invariance, angle wraparound, missing joints, invalid time axes, and preservation of padding/frame-drop masks.
- Desktop tests cover candidate preprocessing parity, isolation from cached legacy models, contract mismatch rejection, below-target coaching suppression, and existing API/stream behavior.
- A real end-to-end desktop run on Beginner2_Drive_001 completed all core streams: the candidate classified Drive at 82.5%, resampled all 142 frames without truncation, and produced the analysis report, playback and contact image. The running desktop health endpoint confirms that candidate files are available.
- Original checkpoints and cached feature arrays remain intact. New data are under data/serve_v2; pilot models under models/serve_v2; final candidate under models/serve_v2_fixed.

[Final training configuration](../models/serve_v2_fixed/training_config.json) · [CV results](../models/serve_v2_fixed/cv_results.json) · [Frozen model hashes](../models/serve_v2_fixed/frozen_models.json) · [Diagnostic evaluation](../models/serve_v2_fixed/evaluation.json) · [Per-clip predictions](../models/serve_v2_fixed/predictions.csv)

```powershell
python scripts/prepare_serve_v2.py --workers 3
python scripts/retrain_serve_v2.py
python scripts/retrain_serve_v2.py --out-dir models/serve_v2_fixed --schedule fixed100
python scripts/evaluate_serve_v2.py --src models/serve_v2_fixed
python scripts/report_serve_v2.py
python scripts/test_serve_sequence.py
python -m unittest desktop.test_desktop
```

The dataset/contract is frozen: source or configuration changes require a new version. Resume checks reject stale pose provenance or a different training configuration.

![Retraining comparison](../outputs/serve_v2_results.png)

Implementation references: [TensorFlow determinism](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism), [MediaPipe PoseLandmarker options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions).
