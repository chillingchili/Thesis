# BlazePose and serve-classifier audit — 2026-10-01

The inconsistency is confirmed. The saved Heavy-pose/GRU pipeline scores 89.3% on its training clips, 86.8% in reconstructed stratified cross-validation, and 38.3% on the 175 existing test clips. These are **serve classification** scores, not BlazePose joint-location accuracy. BlazePose is the pretrained landmark extractor; the GRU is the classifier trained on this project's sequences.

No model was trained, no checkpoint or threshold was chosen using these results, and the desktop/Android model settings were left unchanged. Only evaluation reporting was corrected. All results below use the existing five-model ensemble unless marked OOF.

## Scope and protocol

- Re-evaluated all 438 available training sequences and 175 test sequences (Beginner 3: 85; Beginner 4: 90).
- Re-extracted both legacy BlazePose Heavy and Tasks Lite on 33 videos: two randomly selected clips per player × class using seed 42 (30 total), plus Beginner2_Drive_001, _002 and _007. Fresh sample results are diagnostic; they are not full-dataset desktop accuracy estimates.
- Used manifest serve-type labels. The 205 available Coach B annotated clips have 0 label conflicts with their manifest labels. There is no manual landmark ground truth or identified Coach C 30-clip annotation set in this audit.
- OOF is reconstructed from the current manifest order, StratifiedKFold(5, shuffle=True, random_state=42), and saved fold checkpoints. Original fold IDs were not saved; the reconstruction reproduces the previously reported fold scores but is conditional on that order/seed. Validation also guided early stopping. Players occur on both sides of stratified folds, so this does not measure generalization to a new player.
- Four of five models have seen each training clip. Ensemble training accuracy is a resubstitution diagnostic, not an independent evaluation.
- The existing docs/RESULTS_SUMMARY.md and analysis scripts already examined Beginners 3/4 and compared adaptation methods on them. Treat these as previously inspected test sets, not a pristine final holdout.

## Full-dataset results

| Input pipeline using cached Heavy poses | Training (438) | Beginner 3 (85) | Beginner 4 (90) | Combined test (175) |
|---|---:|---:|---:|---:|
| Original: contact + 10 trim, normalized angles, no matching | 89.3% | 34.1% | 42.2% | 38.3% |
| Untrimmed first 128, no matching | 80.4% | 27.1% | 48.9% | 38.3% |
| Desktop preprocessing: untrimmed first 128 + moment matching | 60.7% | 38.8% | 60.0% | 49.7% |
| Trimmed first 128 + moment matching | 66.7% | 42.4% | 54.4% | 48.6% |

The desktop-preprocessing rows still use **cached Heavy landmarks** to isolate preprocessing. The actual desktop uses Lite; its fresh-sample results appear below. Moment matching rescales each channel to training mean/std at inference, although the GRU was trained on unmodified sin/cos features. It changes the signal the model sees. Its mixed effects do not establish it as a validated fix.

OOF fold accuracies: 90.9%, 92.0%, 85.2%, 89.7%, 75.9%.

### Original test confusion matrix

Rows are manifest labels; columns are predicted classes.

| Actual / predicted | Drive | Lob | Topspin |
|---|---:|---:|---:|
| drive | 17 | 0 | 44 |
| lob | 7 | 18 | 33 |
| topspin | 23 | 1 | 32 |

At the existing 0.60 confidence threshold, 128/175 predictions are accepted (73.1% coverage). **81 of those accepted predictions are wrong**; accepted-only accuracy is 36.7%. Accuracy with abstentions counted as wrong is 26.9%. Mean softmax confidence is 74.5%, despite only 38.3% accuracy. A high confidence number is therefore not a calibrated probability of correctness.

## Pose quality and temporal coverage

| Cached poses | Frames | Frames with pose | Estimated contact outside first 128 |
|---|---:|---:|---:|
| train | 79,242 | 79,236 (99.992%) | 159/438 |
| test_beg3 | 12,415 | 12,415 (100.000%) | 7/85 |
| test_beg4 | 13,125 | 13,125 (100.000%) | 0/90 |

All saved trims exactly reproduce the current contact heuristic, and every saved angle array exactly reproduces angles regenerated from normalized trimmed landmarks (max difference 0.0). No duplicate raw-pose file hashes were found among the 613 available clips. This is a cache-consistency check, not proof of independence between adjacent recorded serves.

The contact estimate is a peak wrist-speed heuristic, not an annotated paddle/ball impact. For 159 training clips the estimated impact is entirely outside the classifier's first 128 frames. Merely trimming after impact does not bring a late impact into that window.

Very high detection coverage does not validate joint locations, wrist tracking, player identity, or biomechanics. The original extractor stores x/y only and discards visibility. The fresh audit also records visibility, but model visibility is not human ground truth.

## Fresh Heavy vs Lite comparison

| Pose extractor / preprocessing | Training sample (21) | Test sample (12) | All sampled (33) |
|---|---:|---:|---:|
| cached_heavy / training_preprocessing | 17/21 (81.0%) | 4/12 (33.3%) | 21/33 (63.6%) |
| cached_heavy / desktop_default | 13/21 (61.9%) | 6/12 (50.0%) | 19/33 (57.6%) |
| heavy / training_preprocessing | 17/21 (81.0%) | 4/12 (33.3%) | 21/33 (63.6%) |
| heavy / desktop_default | 13/21 (61.9%) | 7/12 (58.3%) | 20/33 (60.6%) |
| lite / training_preprocessing | 16/21 (76.2%) | 2/12 (16.7%) | 18/33 (54.5%) |
| lite / desktop_default | 13/21 (61.9%) | 6/12 (50.0%) | 19/33 (57.6%) |

Fresh Heavy uses the training script's model/thresholds but resets the tracker between clips. Lite uses the desktop's model/thresholds. Differences therefore include the model variant, graph/API, and tracking settings; this is not a controlled benchmark of network weights alone. Each variant gets the same decoded full video and GRU.

- Heavy/Lite predicted-label disagreements with training_preprocessing: **3/33**.
- Heavy/Lite predicted-label disagreements with desktop_default: **3/33**.

Median across clips of per-clip median angle disagreement (degrees): elbow 2.23, shoulder 1.51, wrist 4.84, torso 0.53, twist 5.79. Disagreement does not identify which extractor is more accurate.

### Repeated-run and clip-order checks

- Beginner2_Drive_001 / lite: maximum repeated-run coordinate change 0; identical missing-frame masks: True.
- Beginner2_Drive_001 / heavy: maximum repeated-run coordinate change 0; identical missing-frame masks: True.
- Beginner2_Drive_002 / lite: maximum repeated-run coordinate change 0; identical missing-frame masks: True.
- Beginner2_Drive_002 / heavy: maximum repeated-run coordinate change 0; identical missing-frame masks: True.

The original extractor creates one Heavy tracker outside its video loop and does not reset it across videos. In the controlled Drive 2 check, processing Drive 1 first changes Drive 2's coordinates by up to 0.254793 normalized units. Resetting the tracker brings that maximum difference to 0. This establishes an input-order dependency; cached arrays can depend on extraction order and which earlier files were skipped. The predicted labels did not change in this controlled order check. It does not establish that this alone causes all errors.

## Beginner 2 Drive 1 and 2

Both are labeled Drive by Coach B and by the manifest. The table below uses freshly extracted landmarks; percentages are model confidence, not correctness.

| Clip | Heavy + training preprocessing | Lite + training preprocessing | Lite + desktop defaults |
|---|---|---|---|
| Beginner2_Drive_001 | topspin (50.3%) | topspin (56.5%) | lob (69.1%) |
| Beginner2_Drive_002 | topspin (48.3%) | topspin (72.2%) | lob (86.3%) |

The current serve classifier does not predict Coach B's wrist-lag or early-wiper annotations. Those detailed feedback disagreements cannot be resolved just by raising the serve-classification confidence threshold.

## Conversion, dataset, and reporting checks

- TFLite vs Keras: 132 real windows × 5 folds = 660 comparisons; zero label disagreements, maximum probability difference 5.96046448e-07. Deployed model assets are byte-identical to exported files. These checks were on this Windows CPU runtime, not a physical Android device.
- Nine training-manifest rows have no usable cached arrays; their IDs are in inventory.json.
- Coach B workbook has 210 rows, but only 205 have available model features. Unavailable IDs: Beginner1_Topspin_009, Beginner1_Topspin_011, CoachA_Lob_019, CoachA_Lob_023, CoachA_Topspin_021. The existing model was evaluated/trained on 438 available training clips, which does not match the thesis's 210 annotated training and independent 30-clip evaluation protocol. A subset score cannot retroactively make this a model trained under that protocol.
- Fixed scripts/testrun.py: its report previously said abstentions were counted wrong while actually reporting unconditional argmax accuracy. It now prints both explicitly, reports coverage and confident errors, handles missing classes in the confident subset, and defaults to its documented 0.60 threshold instead of 0.40. The old behavior can still be requested explicitly with --threshold 0.40. Two regression checks pass.

## What should change next

1. Freeze a versioned split manifest that matches the thesis, including Coach B labels and the independently annotated Coach C evaluation set. Recover or document missing clips; use participant-held-out development checks as well as the required stratified CV. Obtain a new untouched final set if inspected test results guide improvements.
2. Use one explicit extraction and preprocessing contract for training, desktop, and Android: same pose variant, tracker reset per recording, visibility handling, frame sampling rate, and serve window. Retrain and validate under that contract. Do not assume swapping Lite for Heavy fixes the classifier without validation.
3. Validate contact detection and place the whole serve into a consistent time window. The phone additionally uses asynchronous live frames and a rolling buffer of valid poses, unlike the offline full-frame sequences. Android live accuracy remains unmeasured.
4. Annotate representative shoulder/elbow/wrist/index/hip locations, especially near contact and during occlusion, to measure dataset-specific landmark error/PCK. Review the contact overlays saved by this audit before attributing all error to BlazePose.
5. Validate confidence/abstention on development data and gate specific coaching rules on validated measurements. Accurate serve type alone cannot validate wrist-lag or swing-path diagnoses.

## Reproduce and inspect

```powershell
python scripts/audit_pose_classifier.py cached
python scripts/audit_pose_classifier.py fresh --per-stratum 2
python scripts/audit_pose_classifier.py order
python scripts/report_pose_audit.py
python scripts/test_testrun_reporting.py
```

Fresh extraction resumes files under outputs/pose_audit/fresh. Remove or use a separate output directory before evaluating changed weights, videos, or extraction settings; the saved audit is a snapshot of the current artifacts.

Machine-readable evidence: [cached metrics](../outputs/pose_audit/cached_metrics.json), [per-clip predictions](../outputs/pose_audit/cached_predictions.csv), [fresh metrics](../outputs/pose_audit/fresh_metrics.json), [fresh predictions](../outputs/pose_audit/fresh_predictions.csv), [clip-order check](../outputs/pose_audit/order_check.json), [provenance](../outputs/pose_audit/provenance.json).

![Accuracy and confusion matrices](../outputs/pose_audit/accuracy.png)

![Pose overlays at cached estimated contact](../outputs/pose_audit/drive_examples.png)

MediaPipe reference: [Pose solution and model complexity](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md), [Pose Landmarker overview](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker). Published benchmark accuracy is not measured accuracy on these pickleball videos.
