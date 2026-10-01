# CHAPTER 5
# RESULTS AND DISCUSSION

**Working draft — evidence available on 1 October 2026.** Chapter 5 is drafted as Results and Discussion, and Chapter 6 as Summary, Conclusions, and Recommendations. These titles should be aligned with the department's final format. Independent Coach C judgments and physical-device measurements have not yet been collected; passages describing those evaluations remain explicitly pending. This draft does not replace the existing Chapters 1–4.

## 5.1 Overview of the Evaluation

This chapter presents the completed computational experiments for the pickleball serve evaluation system. The general objective was to combine pose kinematics, a bodyweight-shift measurement, and paddle orientation to classify Drive, Lob, and Topspin serves and generate expert-derived corrective feedback. Ball placement was implemented as an auxiliary stream requiring a suitable secondary-camera view for the intended evaluation. The results address classification performance, input representation, augmentation, multimodal integration, confidence handling, and the status of independent validation.

The thesis specifies a minimum overall stratified cross-validation accuracy of 85% for the GRU classifier in Section 4.5.1, with further evaluation described in Section 4.10.1. This requirement concerns classification across the three subtypes under the stated cross-validation procedure. It does not establish an 85% accuracy requirement on every new player, nor does satisfying it establish the appropriateness of the resulting feedback. Accordingly, stratified development results, participant-separated results, and diagnostic results are reported separately.

Stratified cross-validation preserves subtype proportions across folds but can place recordings from the same player in both training and validation. Participant-held-out cross-validation instead excludes an entire player's recordings from a fold's model fitting. The diagnostic set consists of 175 recordings from Beginners 3 and 4. Although these players are excluded from the recent model-fitting sets, their recordings were inspected in earlier experiments. They therefore cannot be treated as an untouched final test. The separate 30-clip Coach C evaluation described in Chapter 4 remains incomplete.

Accuracy represents the proportion of clips correctly classified. Precision measures how often a predicted subtype is correct, and recall measures how often a reference subtype is recovered. Macro-F1 gives each subtype equal weight. Tracking failures are counted as unavailable or incorrect in the recent all-clip diagnostic accuracy results. These distinctions prevent apparently favorable overall scores from concealing weak subtype recognition or excluded difficult clips.

## 5.2 Dataset Preparation and Actual Evaluation Boundaries

The original experiments used 438 available training sequences from Coach A, Beginner 1, and Beginner 2. The later retraining and refinement experiments used the 210-clip Coach B annotated inventory corresponding to the proposed 70 clips per subtype. Fresh extraction and the predeclared tracking gate accepted 209 clips for fitting. CoachA_Drive_018 did not pass that gate and was excluded from model fitting without changing its annotation. The resulting training counts were 69 Drive, 70 Lob, and 70 Topspin clips.

**Table 5.1. Data quantities used in the completed experiments**

| Data component | Quantity | Interpretation |
|---|---:|---|
| Original available training sequences | 438 | Historical training inputs; not an additional independent player population |
| Coach B annotated retraining inventory | 210 | The intended annotated training subset |
| Eligible originals for recent fitting | 209 | Three source players; one tracking rejection |
| Frozen training video variants | 418 | Two augmented variants per eligible original |
| Beginner 3 diagnostic recordings | 85 | Previously inspected; excluded from recent fitting |
| Beginner 4 diagnostic recordings | 90 | Previously inspected; excluded from recent fitting |
| Selected beginner clips for Coach C staging | 24 | Four clips per recorded subtype from each beginner |
| Additional coach clips required | 6 | Two per recorded subtype; to be supplied from unused recordings |

These quantities describe different inventories and must not be added as if they were disjoint recordings. Augmented variants are derived views of existing clips, not new serves or new participants. The planned recording totals in Chapter 4 require reconciliation against the final recording inventory before submission.

Recent models were fitted using training data only. The augmentation variants inherited their original clip's fold membership, and validation used original, unaugmented clips. This separation prevents an augmented view of a validation clip from entering the same model's training fold. However, separation from fitting does not undo earlier diagnostic inspection. The 24 staged beginner review clips have no overlap with the checked original GRU, original kNN bank, or recent training inventories, but all 24 were previously present in diagnostic development. Their eventual Coach C assessment should therefore be reported as a retrospective independent coach review.

## 5.3 Original GRU and Classical Baselines

The historical results report an original GRU mean stratified five-fold accuracy of approximately 86.7%. A subsequent reconstruction of out-of-fold predictions yielded approximately 86.8% pooled accuracy. The small difference reflects reporting and aggregation conventions rather than a separate performance improvement. The original fold memberships were reconstructed from the available manifest order and random seed; the original validation procedure also guided early stopping. These qualifications should accompany the original score.

The September experiment record included several classical models using 32 timing features. Their reported results are summarized below to retain the broader experimental context.

**Table 5.2. Selected historical classical-model results**

| Model or family | Reported stratified CV accuracy | Reported diagnostic mean across Beginners 3 and 4 |
|---|---:|---:|
| Gradient Boosting | 92.9% | 44.9% |
| Random Forest | 91.8% | 48.9% |
| Scaled RBF SVM | 89.9% | 48.1% |
| Scaled kNN7, Manhattan distance | 87.9% | 48.7% |
| Scaled kNN7, Euclidean distance | 85.4% | Not reported in the summary |
| Scaled kNN5 | 84.0% | Not reported in the summary |
| Logistic Regression | 83.1% | 48.5% |
| Raw kNN5 family | 78.3% | 56.1% for the distance-weighted variant |

The diagnostic column retains the historical report's unweighted mean of the two player accuracies. It is not the pooled 175-clip accuracy used in the recent evaluations. The raw kNN CV entry and the distance-weighted diagnostic entry are family-level results rather than a matched ablation. These historical comparisons also used older cached inputs, so their scores should not be substituted for the current fresh-Lite pipeline results.

The original exploration also tested strong subject-style augmentation, metric or center-loss learning, adversarial subject-invariance, and adaptation methods. The reported diagnostic means were 41.5% for strong augmentation and 52.0% for metric learning. Adversarial runs were unstable, with reported results roughly in the 44–48% range; combining adversarial training with strong augmentation produced roughly 36–37.5%. A moment-matching adaptation mixture reached a reported 58.3% mean. These were exploratory uses of the same diagnostic players. They did not establish an independently validated replacement for the deployed pipeline.

The classical and recurrent experiments both demonstrated that strong stratified scores could coexist with substantially weaker recognition on different players. This pattern is consistent with sensitivity to player style and recording conditions. It does not prove that every error has a single dataset-related cause.

## 5.4 Corrected GRU Retraining

The corrected retraining used fresh BlazePose Lite VIDEO extraction with a separate tracker for each recording. A complete selected serve was resampled to 128 time steps. Required-joint visibility and valid-frame gates were retained, unreliable observations were masked, and interpolation did not bridge missing observations. Inference-only moment matching was removed for the candidate so that training and inference used the same representation. Padding and dropped-frame handling were also corrected to avoid artificial nonzero pose features.

A training-only pilot used inner validation to determine stopping epochs. The final retraining candidate used a fixed 100-epoch schedule, informed by that development evidence. Because this schedule was selected during development, the resulting cross-validation score is a development estimate rather than an untouched estimate after all hyperparameter choices.

**Table 5.3. Training-development results for the corrected retraining**

| Evaluation | Initial retraining pilot | Fixed 100-epoch candidate |
|---|---:|---:|
| Stratified five-fold accuracy | 73.7% | 90.9% |
| Participant-held-out accuracy | 33.0% | 36.4% |

The fixed candidate achieved 190 correct classifications out of 209 out-of-fold training cases, corresponding to 90.9% accuracy. Its five stratified fold accuracies were approximately 88.1%, 88.1%, 92.9%, 97.6%, and 87.8%. It therefore exceeded the paper's 85% stratified-CV requirement under this development procedure. The pilot did not meet that requirement.

**Table 5.4. Frozen-model diagnostic results using fresh Lite poses**

| Model | Beginner 3 | Beginner 4 | All 175 diagnostic clips |
|---|---:|---:|---:|
| Original GRU ensemble | 43.5% | 58.9% | 51.4% |
| Retrained five-fold GRU ensemble | 43.5% | 50.0% | 46.9% |
| Retrained single GRU | 29.4% | 37.8% | 33.7% |

The retrained ensemble did not improve aggregate diagnostic accuracy. Its all-clip accuracy was approximately 4.6 percentage points lower than the original ensemble under the fresh-Lite comparison. Correcting an individual clip did not imply a general improvement: Beginner2_Drive_001 changed from an incorrect Lob prediction to a correct Drive prediction, whereas Beginner2_Drive_002 remained incorrect under the retrained ensemble. Both examples are training clips and are illustrations rather than independent performance evidence.

The original cached-Heavy audit's 38.3% diagnostic result should not be used to claim that the retrain improved performance to 46.9%. The two scores use different extraction and preprocessing conditions. The relevant original-model comparison for this retraining is the fresh-Lite 51.4% result.

## 5.5 Raw-Video Augmentation and Richer Pose Inputs

The next study implemented the thesis's augmentation concept before fresh pose extraction. It created two frozen variants per eligible original, for 418 variants. The transformations included brightness and contrast changes, per-pixel Gaussian noise, slight isotropic cropping and translation, and temporal subsampling. Flips, shears, and nonuniform stretching were excluded. All variants passed the specified tracking gate.

Each condition used the same 100-epoch training budget, with one sampled original or variant per original clip per epoch. This controlled optimizer-update count instead of simply giving the augmented condition more training steps. The comparison included five angle pairs, normalized skeleton coordinates with visibility, and skeleton coordinates with adjacent-step motion. Richer representations also increased parameter count, so differences cannot be attributed solely to additional information.

**Table 5.5. Matched GRU input and augmentation study**

| Condition | Stratified development CV | Participant-held-out CV | Diagnostic accuracy |
|---|---:|---:|---:|
| Five angles, control | 87.6% | 36.4% | 46.3% |
| Five angles with video augmentation | 85.6% | 47.8% | 47.4% |
| Full skeleton | 91.9% | 35.4% | 55.4% |
| Full skeleton with motion | 91.9% | 32.1% | 56.0% |

Video augmentation increased participant-held-out accuracy by approximately 11.5 percentage points against the matched angle control. The change on diagnostic clips was approximately 1.1 percentage points. Augmentation therefore helped in a specific development comparison, while its transfer benefit on the previously inspected diagnostic players was smaller.

The full-skeleton and skeleton-motion conditions increased overall diagnostic accuracy, but did not provide balanced subtype recognition. The skeleton GRU's Drive, Lob, and Topspin recalls were approximately 88.5%, 70.7%, and 3.6%, respectively. For skeleton with motion, they were approximately 88.5%, 74.1%, and 1.8%. Consequently, the 56.0% skeleton-motion result cannot be interpreted as reliable recognition of all three serve types. Pose geometry also does not directly measure the ball's spin.

Training-only selection favored the augmented angle condition using the declared participant-held-out development criterion. The model selection and hashes were frozen before that run's diagnostic evaluation. Nevertheless, the diagnostic recordings had been inspected in earlier work, and the study used a single seed. Statistical significance and robustness across random initializations have not been established.

## 5.6 GRU–kNN Fusion and Input-Window Sensitivity

The original Android classifier combines a five-fold GRU ensemble with distance-weighted kNN5. The kNN bank contains 438 historical training examples. The fixed hybrid assigns equal weights to the GRU and kNN probabilities. Final Android classification and feedback were corrected to use the combined probabilities, and the first timing-feature velocity was changed to zero to match offline extraction. These are implementation-consistency corrections; they do not independently demonstrate improved live-camera accuracy.

**Table 5.6. Bundled classifiers on recorded first-128 windows**

| Classifier | Diagnostic accuracy | Macro-F1 |
|---|---:|---:|
| Original GRU ensemble | 51.4% | 0.479 |
| Original kNN5 | 45.7% | 0.413 |
| Fixed GRU–kNN5 hybrid | 49.7% | 0.496 |

The hybrid produced a slightly higher macro-F1 than the GRU but a lower overall diagnostic accuracy. This illustrates that model choice can depend on the metric and that adding kNN does not automatically resolve errors. At the existing 0.60 confidence gate, the original hybrid accepted 89 of 175 clips and was wrong on 41 accepted clips. Accepted-case accuracy was 53.9%, which does not support interpreting confidence as a verified probability of correctness.

Input windows were also important. The same bundled assets produced the following recorded-window results.

**Table 5.7. Diagnostic accuracy of unchanged original assets under different windows**

| Window | GRU | Fixed hybrid |
|---|---:|---:|
| First 128 valid pose frames | 51.4% | 49.7% |
| Last 128 valid pose frames | 40.6% | 50.3% |
| Complete timeline resampled to 128 steps | 48.0% | 47.4% |

The first- and last-window paths discarded 3,789 detected poses across 142 clips. Complete-timeline resampling avoided that cropping, but did not automatically improve accuracy when applied to assets trained under the legacy representation. Capturing the entire serve and matching the training input contract are related but separate requirements. Recorded VIDEO replay also differs from the app's asynchronous live tracker and motion gate.

## 5.7 Lean Features and Nested Calibrated Fusion

The refinement study combined raw-video augmentation with compact arm/torso features and richer skeleton representations. The lean representation contained 45 channels: angle pairs, selected normalized joint coordinates, visibility, and adjacent displacements. Fixed fusion was compared with a policy fitted using nested participant-separated development predictions. The held-out outer player's labels were excluded from weight selection, temperature fitting, and confidence-threshold fitting.

**Table 5.8. Participant-held-out refinement results**

| Condition | GRU accuracy | Fixed hybrid accuracy | Nested tuned hybrid accuracy | Mean-player tuned macro-F1 |
|---|---:|---:|---:|---:|
| Angles with video augmentation | 47.8% | 57.9% | 53.1% | 0.481 |
| Lean arm/torso, control | 35.9% | 53.6% | 56.9% | 0.522 |
| Lean arm/torso with augmentation | 38.3% | 54.5% | 53.6% | 0.489 |
| Full skeleton with augmentation | 33.5% | 52.6% | 55.5% | 0.509 |
| Skeleton/motion with augmentation | 32.5% | 50.7% | 53.6% | 0.489 |

The selected research condition was lean arm/torso control, using mean-player tuned macro-F1 as the selection criterion. Its final policy assigned 25% weight to GRU and 75% to kNN, with temperature approximately 9.244. Selection among conditions using outer development results can still favor a winner whose apparent score is optimistic; a separate new-player evaluation remains necessary.

**Table 5.9. Final single-fit diagnostic results for the refinement study**

| Condition | GRU | Fixed hybrid | Tuned calibrated hybrid |
|---|---:|---:|---:|
| Angles with video augmentation | 49.7% | 46.3% | 42.9% |
| Lean arm/torso, control | 45.7% | 40.6% | 44.0% |
| Lean arm/torso with augmentation | 45.7% | 41.7% | 42.9% |
| Full skeleton with augmentation | 56.6% | 46.9% | 42.9% |
| Skeleton/motion with augmentation | 52.0% | 41.7% | 41.1% |

These results use single models refitted on the full eligible training set, whereas the preceding richer-input diagnostic study used five-fold GRU ensembles. They are therefore not a controlled comparison of augmentation alone. The 56.6% skeleton result was accompanied by only 5.4% Topspin recall, again demonstrating the importance of class-specific evaluation.

Calibration improved some probability-quality measures without establishing reliable recognition. For lean control, nested log loss decreased from approximately 2.233 to 1.100 relative to fixed fusion. However, confidence rules that achieved the inner fitting criterion of at least 80% observed accepted accuracy achieved only 25.6–35.0% accepted accuracy on outer-player evaluation across the five conditions. Every final research policy therefore disables confidence acceptance. Temperature scaling changes probability sharpness while preserving the fused class ranking; it cannot repair a wrong class ranking by itself.

## 5.8 Equipment Detection and Deterministic Mechanics Checks

The system includes pretrained pose extraction, trained equipment and court detectors, and deterministic geometric checks. Their evaluation measures are different from serve classification. In particular, pose detection coverage does not validate anatomical landmark accuracy; no manually annotated landmark ground truth was available for this work.

**Table 5.10. Detector validation values from the final logged epoch**

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Ball detector | 95.8% | 92.1% | 93.8% | 50.4% |
| Paddle detector, base | 85.3% | 77.1% | 80.9% | 36.7% |
| Paddle detector, fine-tuned | 98.5% | 73.7% | 84.7% | 54.5% |
| Court landmarks, base | 99.9% | 100.0% | 99.5% | 97.7% |
| Court landmarks, fine-tuned | 99.9% | 100.0% | 99.5% | 98.4% |

Ball and paddle rows use bounding-box metrics; court rows use keypoint metrics. These figures come from the individual training-run validation logs. They are not a common test-set comparison, best-checkpoint test result, or measured end-to-end accuracy on Coach C's evaluation clips. The one-epoch paddle smoke run is an engineering check rather than a substantive model result.

The current weight-shift proxy measures hip-center image displacement normalized by body height. Its threshold is approximately 0.1543, derived from 59 explicitly good-form Coach A baseline clips under Coach B's annotations. The paddle reference band is approximately 13.4°–165.7°, based on 52 baseline clips with usable measurement windows. These are empirical geometric references. They do not directly measure ground reaction force, ball spin, or paddle-face yaw. Being recorded by the professional coach was not treated as sufficient evidence of good form; explicit annotations determined baseline membership.

The paddle measurement describes a long-axis angle in the image plane. Directional open/closed advice remains inactive without validated direction. Current feedback can request review of an angle outside the reference band without asserting a verified three-dimensional face orientation. A measured mechanics cue also does not amount to an assessment of the player's overall serve form.

## 5.9 System Verification and Pending Independent Validation

The latest automated verification record reports 20 passing Python checks and 35 passing Kotlin unit tests. Android debug and phone-test APK builds and lint completed. The selected research TFLite model was numerically compared with Keras across 175 diagnostic inputs, with maximum probability difference approximately 2.68 × 10⁻⁶. These checks support implementation consistency and export correctness. They do not demonstrate biomechanical validity or physical-device performance.

No connected-phone test was completed. Decoder behavior, pose extraction, native inference performance, and real live-capture callbacks still require device measurements. Desktop processing times cannot be substituted for the thesis's core-pipeline latency target of at most two seconds on the specified smartphone. The auxiliary ball-placement stream must be timed separately.

Coach C's Session 1 subtype and binary-form annotations, Session 2 appropriateness judgments, and final descriptive 30-clip classification results are pending. Twenty-four beginner clips have been randomly selected; six additional unused coach clips remain to be incorporated and checked. The intended quota is ten per recorded subtype, but independent annotation may alter the observed label distribution. No feedback-appropriateness percentage can yet be reported.

The taxonomy distinguishability gate described in Chapter 4 has no completed outcome documented in the available artifacts. Consequently, this chapter does not claim that independent expert agreement on the three-subtype taxonomy has been established. It also does not claim that all professional recordings have good form or all beginner recordings have bad form.

## 5.10 Discussion in Relation to the Objectives

The computational work supports the feasibility of constructing a multimodal proof of concept and meeting the stated stratified-CV classification gate in several GRU conditions. The original GRU and the fixed-epoch retrained ensemble exceed the 85% criterion. All four matched richer-input GRU conditions also exceed it. This finding should be reported alongside the participant-separated and diagnostic results, because the threshold's evaluation design does not represent new-player reliability.

The dataset-preparation objective was implemented through an annotated inventory, explicit exclusions, frozen source hashes, and original-level augmentation splits. The final evaluation boundary departs from the proposal's intended untouched holdout because the beginner review recordings were previously inspected. That departure is material and must be acknowledged rather than corrected only by changing terminology in a final table.

The multimodal-feedback objective was implemented through independent streams and expert-derived rules, but independent confirmation of feedback appropriateness remains outstanding. The three-subtype taxonomy's expert distinguishability outcome and physical-device latency are also unresolved. Thus, the study currently supports a working research prototype and a qualifying development classification result, with incomplete evidence for validated mobile coaching.

The remaining errors are consistent with limited independent player diversity, representation limitations, and capture or preprocessing differences. The controlled augmentation experiment demonstrated a meaningful improvement in one participant-held-out comparison; subsequent diagnostic results and class collapse demonstrate that further training changes did not yield a consistently superior model. The completed evidence does not establish that the dataset is the sole cause or that additional training epochs would resolve the problem.

# CHAPTER 6
# SUMMARY, CONCLUSIONS, AND RECOMMENDATIONS

## 6.1 Summary of the Study

This study developed a pickleball serve evaluation proof of concept using pose-based sequence classification, deterministic weight-shift measurement, paddle-angle estimation, and an expert-derived rule engine. An optional ball-placement module uses court geometry and ball tracking. The research addressed the computational distinction among Drive, Lob, and Topspin, the construction of an annotated video dataset, the integration of multimodal measurements, classification performance, and independent feedback validation.

The original development explored GRUs and several timing-feature baselines. Corrected retraining used 209 eligible annotated originals from three players and a shared complete-serve preprocessing contract. Follow-up studies added 418 raw-video augmentation variants, richer skeletal representations, compact arm/torso features, and nested calibrated fusion. Evaluation separated stratified development CV, participant-held-out CV, and 175 previously inspected diagnostic recordings.

## 6.2 Summary of Findings

The original GRU achieved approximately 86.7% mean stratified CV accuracy, with a later reconstructed pooled value of approximately 86.8%. The fixed 100-epoch retrained ensemble achieved 90.9%. Both exceeded the thesis's 85% stratified-CV requirement. The retraining pilot achieved 73.7% and did not meet that requirement. The original and retrained scores use different training inventories and preprocessing, so their numerical difference does not isolate the effect of retraining alone.

The qualifying development scores did not establish reliable transfer to different players. Under the fresh-Lite diagnostic comparison, the original GRU ensemble achieved 51.4%, the retrained ensemble 46.9%, and the retrained single GRU 33.7%. The recent participant-held-out GRU score was only 36.4% for the fixed retrain.

The matched raw-video augmentation experiment increased participant-held-out angle-GRU accuracy from 36.4% to 47.8%, while diagnostic accuracy changed from 46.3% to 47.4%. Augmentation helped that comparison but did not add independent player diversity. Richer inputs reached 55.4–56.0% diagnostic accuracy in the ensemble study while nearly failing to recognize Topspin. A later full-skeleton augmented single fit reached 56.6% with only 5.4% Topspin recall.

The original fixed GRU–kNN hybrid achieved 49.7% pooled diagnostic accuracy and macro-F1 of 0.496. Its aggregate accuracy was lower than the original GRU's 51.4%, despite a slightly higher macro-F1. Nested fusion selected lean arm/torso control on training-player macro-F1, but its final tuned diagnostic accuracy was 44.0%. No consistently superior replacement was established across the relevant evaluations.

Calibration improved some probability-quality scores, but the fitted confidence-acceptance rules did not transfer successfully to held-out players. Final research policies disable confidence acceptance. Accordingly, high model confidence cannot be used as independent proof that either a subtype prediction or a coaching cue is correct.

Equipment-detector validation logs and numerical parity checks support parts of the implementation. They do not establish end-to-end serve-evaluation accuracy, overall good form, or physical-phone latency. Independent Coach C annotation and feedback review, the documented taxonomy gate, and real-device evaluation remain pending.

## 6.3 Conclusions

The study demonstrates the implementation of a pose-based, multimodal pickleball serve evaluation prototype and satisfaction of the stated stratified-CV classification threshold under several completed development procedures. The threshold finding is limited to its specified evaluation design. It should not be extended to a claim of at least 85% accuracy on unfamiliar players or to a claim that the complete feedback system is validated.

Retraining did not produce a consistent improvement in diagnostic recognition. Some later treatments improved particular measures, including participant-held-out accuracy under matched video augmentation, but the results varied by player, representation, and model aggregation. Describing the work as only two training runs with uniformly minimal improvement would omit these distinctions. It is more accurate to report an original development stage, a corrected retraining stage, and additional augmentation, representation, and fusion studies, each involving multiple fold fits.

Limited independent player coverage is a plausible contributor to the observed generalization gap. Feature sufficiency, extraction quality, input-window compatibility, and taxonomy ambiguity are also plausible contributors. The completed experiments do not establish their individual causal contributions. The remaining problem therefore cannot be attributed exclusively to insufficient clip quantity, and a larger recurrent model cannot be assumed to solve it.

The rule engine provides bounded, reproducible cues from measurable conditions. Its appropriateness must still be established through the planned independent review. In the absence of those judgments and physical-phone measurements, the defensible outcome is a qualifying development classifier and a working proof of concept with documented limitations. Final conclusions about feedback validity, taxonomy agreement, and mobile performance must be updated when their evidence is available.

## 6.4 Recommendations Within the Existing Dataset and Budget

### 6.4.1 Complete evaluation before changing the final candidate

Complete the intended 30-clip packet using the six unused coach recordings. Check source identities and hashes against training inventories, collect Coach C's independent Session 1 labels, freeze the chosen evaluation pipeline, and then generate the complete outputs. Conduct Session 2 using the feedback strings without access to Session 1 annotations. Report classification and feedback judgments separately, retain missing-output cases in the accounting, and identify the beginner review as retrospective because of prior diagnostic exposure.

The present gallery's curated correct examples and single error illustration should not become the evaluation sample. Demonstration selection and evaluation selection serve different purposes. The final review sample must remain independent of model success or confidence.

### 6.4.2 Verify actual phone preprocessing and serve boundaries

Use the existing recorded-replay function to compare the phone and Python pipelines on the same training recordings before measuring live performance. Examine decoded frame counts, timestamps, visibility, input windows, model probabilities, and source-video hashes. Then verify that the live motion gate captures setup through follow-through instead of systematically omitting a useful phase. These checks require the existing device and comparison tooling; they do not require a new training dataset.

Once replay agreement is established, measure core-pipeline latency under the specified hardware conditions and report the timing boundary. If the baseline fails the two-second target, follow the paper's declared hardware-escalation procedure and document the result. Do not infer a phone latency improvement from desktop timings or successful compilation.

### 6.4.3 Diagnose existing training errors by subtype and recording session

Use participant-separated out-of-fold predictions to identify repeated errors, missing wrist or paddle observations, unusual framing, and subtype confusion within the existing training set. Where recording-session information can be recovered, keep adjacent serves and derived views within the same development partition. Review annotation questions with an expert when that review becomes feasible; do not automatically relabel clips to agree with a model.

This recommendation concerns targeted diagnostic use of the existing data rather than fitting on the already inspected Beginner 3/4 recordings. Group-wise evaluation is particularly appropriate when samples share a subject or recording context (scikit-learn developers, n.d.-a).

### 6.4.4 Test compact, phase-aware features under a fixed protocol

A remaining low-cost development direction is to test whether a compact set of relative arm/torso motion and serve-phase features retains subtype differences better than the current representations. For example, training-only inspection can define setup, acceleration, and follow-through phases and assess consistent alignment without assuming that the current wrist-speed proxy is annotated ball contact. Any phase extractor must be implemented identically in training and inference and must preserve missing-data behavior.

Full-skeleton coordinates and adjacent motion have already been tested. This recommendation is therefore a narrower test of phase alignment and feature selection, not a claim that simply adding skeleton motion is untried. A small temporal-convolution baseline or a regularized GRU can be compared with the current classifier under a similar parameter and training budget. The comparison should be declared in advance, use participant-separated development folds, and include class-specific precision, recall, and macro-F1. No improvement is guaranteed.

### 6.4.5 Check repeatability before expanding the architecture search

Repeat a small, preselected set of promising conditions with several declared random seeds and summarize variation across participants and initializations. This can establish whether a one-seed gain is repeatable and whether class collapse persists. Use a limited experiment budget, freeze augmentation membership, and keep fitting, checkpoint selection, and normalization inside the appropriate training folds. Repeatedly searching the same diagnostic players would weaken the independence of subsequent claims (scikit-learn developers, n.d.-b).

### 6.4.6 Refit confidence policies only when their transfer can be evaluated

Keep the final research acceptance policy disabled until a separate reliability assessment supports its use. Examine probability reliability by subtype and player, rather than selecting a larger confidence threshold on the diagnostic set. Probability calibration requires appropriately separated fitting evidence and does not by itself improve the underlying class ranking (scikit-learn developers, n.d.-c).

## 6.5 Longer-Term Recommendations

When additional collection is feasible, prioritize independent training participants, recording sessions, player styles, and camera conditions. Record and annotate new training examples separately from the final evaluation set. Additional transformations of the same three players do not provide equivalent independent coverage.

The three-subtype taxonomy should also receive the expert distinguishability assessment specified in the methodology. If independent experts cannot consistently distinguish the categories using the available evidence, report that outcome and follow the thesis's declared scope contingency instead of forcing agreement through relabeling. Optional ball-flight or outcome features should only be investigated under a revised, explicitly evaluated input scope; the current pose-only classifier does not directly observe spin.

Further mechanics rules should be added only when the necessary measurements and expert interpretation can be validated. Directional paddle cues, contact timing, and ball-placement judgments require their own supporting evidence. The prototype should retain stream independence so that an unavailable measurement does not become an invented positive result.

## 6.6 Required Updates Before Final Submission

The final manuscript must incorporate the completed 30-clip composition, Coach C's independent class and form labels, per-class final classification results, output coverage, and Session 2 feedback-appropriateness count and percentage. It must also include the actual core and auxiliary latency measurements, device conditions, and taxonomy-gate outcome. Missing results should remain identified as pending until collected.

Chapter 4 should be reconciled with the implemented system: native Android implementation rather than the proposed React Native stack where applicable; a serve-subtype GRU with separate deterministic weight-shift measurement rather than a claimed joint learned shift head; 209 usable recent training clips rather than 210 usable clips; inactive directional paddle advice without direction validation; and retrospective independent beginner review rather than an untouched holdout. Assertions that the devices were already validated should be revised to reflect the actual evidence status.

## Evidence and Reference Notes for the Draft

Project results are drawn from the following saved artifacts. These are primary study records, not external literature; their relative links are provided for editorial checking.

| Draft content | Study record |
|---|---|
| Original model families and exploratory adaptations | [September results summary](../../../docs/RESULTS_SUMMARY.md) |
| Original GRU score reconstruction and Heavy/Lite qualifications | [Pose/classifier audit](POSE_CLASSIFIER_AUDIT.md) |
| Retraining counts, CV, diagnostic results and limitations | [Retraining report](GRU_RETRAINING_V2.md), [CV metrics](../models/serve_v2_fixed/cv_results.json) |
| Original hybrid and confidence coverage | [Hybrid audit](HYBRID_CLASSIFIER_AUDIT.md) |
| Matched augmentation and richer-input study | [Follow-up report](FOUR_FOLLOWUP_STUDY.md) |
| Nested fusion, capture comparison and verification | [Refinement report](SERVE_REFINEMENT_V4.md), [Diagnostic metrics](../outputs/serve_refinement_v4/diagnostic_results.json) |
| Current Coach C preparation status | [Status record](../outputs/coach_c_evaluation_v2/status.json) |
| Detector figures | Individual `runs/**/results.csv` final validation rows, also exported in the website dataset |

scikit-learn developers. (n.d.-a). *Cross-validation: evaluating estimator performance*. https://scikit-learn.org/stable/modules/cross_validation.html (accessed 1 October 2026).

scikit-learn developers. (n.d.-b). *Common pitfalls and recommended practices*. https://scikit-learn.org/stable/common_pitfalls.html (accessed 1 October 2026).

scikit-learn developers. (n.d.-c). *Probability calibration*. https://scikit-learn.org/stable/modules/calibration.html (accessed 1 October 2026).

Editorial note: Retain the thesis's existing citation style and integrate these engineering references into its bibliography if the corresponding recommendations are retained. The completed study results do not require invented literature citations or invented Coach C responses.
