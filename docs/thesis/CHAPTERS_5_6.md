# **CHAPTER 5** {#chapter-5}

# **RESULTS AND DISCUSSION** {#results-and-discussion}

## **5.1 Scope of the Reported Results** {#5.1-scope-of-the-reported-results}

This chapter reports the completed classifier experiments, the frozen multimodal outputs for the 30 selected evaluation clips, and Coach C's completed Session 1 form judgments. Session 2 feedback-validity ratings and physical-phone latency measurements were not available when this draft was prepared on 10 October 2026. Consequently, the chapter distinguishes measured findings from pending evaluation outcomes. No Session 2 ratings have been estimated or inferred from Session 1 comments.

Three kinds of evidence are reported separately. Stratified cross-validation measures discrimination among clips sampled from the available training participants. Participant-held-out development and the existing Beginner 3/4 diagnostic set examine transfer under different sampling conditions. Coach C's evaluation examines form judgments and the contextual validity of feedback on the selected 30 clips. Because the actual coach instrument omits independent serve-type labels, it cannot provide independent Coach C subtype-classification accuracy.

## **5.2 Dataset and Evaluation Composition** {#5.2-dataset-and-evaluation-composition}

The annotated training inventory contains 210 clips: 72 from Coach A and 69 each from Beginners 1 and 2, with 70 clips per recorded subtype. The v2 retraining quality checks retained 209 clips and rejected one Coach A Drive recording. This rejection remains documented rather than being relabeled. The legacy classifiers use a different training population; the bundled kNN bank contains 438 samples. Results from these populations are not interchangeable.

**Table 7. Evaluation populations and their interpretation**

| Population | Size | Interpretation |
|---|---:|---|
| Annotated training inventory | 210 clips | Coach B form judgments and comments; three training participants |
| Quality-accepted v2 training inventory | 209 clips | Separate retrained GRU development population |
| Legacy kNN training bank | 438 samples | Existing hybrid asset; distinct from the 210-clip inventory |
| Beginner 3/4 diagnostic set | 175 clips | Previously inspected diagnostic material, not an untouched final test |
| Coach C evaluation set | 30 clips | 12 Beginner 3, 12 Beginner 4, six fresh Coach A recordings |

The Coach C selection contains ten clips per recorded category: four from each evaluation beginner and two from Coach A. The 24 beginner clips had prior diagnostic exposure. The six Coach A recordings were verified as fresh clip additions, but Coach A is already represented in training. The evaluation is therefore described as a retrospective review by an independent coach, not a wholly new-player or pristine sealed test.

## **5.3 GRU Cross-Validation and Generalization** {#5.3-gru-cross-validation-and-generalization}

The original GRU experiments reported a mean stratified five-fold accuracy of 86.7%, exceeding the original 85% numerical target. A later reconstruction produced approximately 86.8% pooled out-of-fold accuracy. These are differently aggregated historical results, and the reconstruction depends on the documented sample order and split seed. They do not establish the accuracy of the fused hybrid, nor do they make the legacy training population identical to the proposal's 210 clips.

The fixed 100-epoch v2 ensemble achieved 190 correct out-of-fold classifications among 209 quality-accepted training clips, or 90.9%. Its reported fold accuracies were approximately 88.1%, 88.1%, 92.9%, 97.6%, and 87.8%. However, its leave-one-training-player-out accuracy was 36.4%. The gap shows that a high clip-stratified score did not translate into comparable performance when an entire participant was excluded from training.

**Table 8. Matched diagnostic comparison using fresh Lite pose extraction**

| Classifier | Accuracy on 175 diagnostic clips |
|---|---:|
| Original GRU ensemble | 51.4% |
| Retrained v2 GRU ensemble | 46.9% |
| Retrained v2 single GRU | 33.7% |
| Original GRU + kNN5 hybrid | 49.7% |

The retrained ensemble did not improve diagnostic accuracy relative to the original ensemble under the matched fresh-Lite comparison. The older 38.3% result used cached Heavy poses and different preprocessing; treating a change from 38.3% to 46.9% as a retraining gain would confound model and input-pipeline changes. The original GRU ensemble was selected before Session 2 release on the basis of this existing comparison. The previous hybrid outputs remain archived; no claim of statistically established superiority follows from the 1.7-point diagnostic difference.

Only three participants contribute to the principal training population. The results are consistent with limited participant variation and sensitivity to serving style. Nevertheless, the experiments do not isolate the dataset as the sole cause of error. Pose extraction, temporal windows, feature representation, label taxonomy and probability behavior also affect the implemented pipeline.

## **5.4 Augmentation, Representation and Calibration Experiments** {#5.4-augmentation-representation-and-calibration-experiments}

The raw-video augmentation study generated two variants for each of the 209 eligible training clips, yielding 418 additional training views. Original clip identities determined fold membership before augmentation, and variants remained in their source clip's training partition. These additional views did not increase the number of independent participants.

**Table 9. Matched input and video-augmentation study**

| Condition | Stratified development CV | Participant-held-out development | Diagnostic accuracy |
|---|---:|---:|---:|
| Five angles, no video augmentation | 87.6% | 36.4% | 46.3% |
| Five angles, thesis video augmentation | 85.6% | 47.8% | 47.4% |
| 33-joint coordinates and visibility | 91.9% | 35.4% | 55.4% |
| Coordinates, motion and visibility | 91.9% | 32.1% | 56.0% |

Video augmentation improved the participant-held-out angle-model comparison by approximately 11.4 percentage points while changing diagnostic accuracy by approximately 1.1 points. Richer representations achieved higher diagnostic scores in some conditions, but lower participant-held-out development scores. These results do not identify a uniformly superior replacement. They also come from a separate controlled experiment; its unaugmented 87.6% score must not be substituted for the v2 ensemble's 90.9% result.

A later training-only temperature-calibration audit of a separate v4 candidate preserved classification accuracy at 57.9%. Although player-balanced log loss decreased from 1.430 to 1.185, expected calibration error increased from 0.162 to 0.266. At the fixed 0.60 acceptance threshold, coverage decreased from 60.8% to 19.1%, and accepted-case accuracy decreased from 59.8% to 35.0%. That policy was not adopted for the selected original GRU or the archived hybrid. Calibration and rewritten feedback therefore cannot be presented as demonstrated improvements in its recognition accuracy.

## **5.5 Frozen Multimodal Outputs and Coaching Coverage** {#5.5-frozen-multimodal-outputs-and-coaching-coverage}

All 30 evaluation clips produced a serve classification and a hip-movement measurement. Paddle measurements were available for 26 clips and unavailable for C005, C020, C026 and C027. The ball-landing stream was disabled because paired secondary-camera recordings were not supplied for this evaluation. Availability establishes that a signal was produced, not that its classification or physical interpretation was correct.

**Table 10. Frozen output availability and feedback status**

| Output or status | Clips | Percentage of all 30 |
|---|---:|---:|
| Serve classification available | 30 | 100.0% |
| Hip-movement measurement available | 30 | 100.0% |
| Paddle measurement available | 26 | 86.7% |
| Corrective feedback | 19 | 63.3% |
| Positive feedback on assessed measurements | 2 | 6.7% |
| Low-confidence coaching withheld | 8 | 26.7% |
| Partial output without a supported corrective cue | 1 | 3.3% |

The four feedback-status rows are mutually exclusive and total 30. Measurement-availability rows overlap. The selected GRU supplies corrective or positive feedback on 21/30 clips (70.0%), including corrective feedback on 19/30 (63.3%). A missing paddle measurement did not necessarily suppress an otherwise supported hip-movement cue.

The earlier hybrid run withheld coaching on 12 clips and supplied substantive coaching on 17. Recomputing original GRU reduced low-confidence outputs to eight and increased substantive coaching to 21, using the same 30 clips, measurements and threshold. This is a coverage comparison, not a determination that the newly accepted predictions or suggestions are correct. Selection occurred after the researchers had seen Session 1, so it is disclosed as a retrospective pre-release amendment. The selection rationale is the previously available 175-clip diagnostic comparison, not fitting to the coach’s form comments.

The earlier wording-only revision retained the hybrid predictions and triggers. The subsequent GRU selection changed classification and potentially triggered feedback; it reused the same measurements and approved message templates. It replaced generic instructions with a measurement observation and a practical next-attempt suggestion. For example, the hip cue begins, “Your hips showed limited movement during the serve,” and suggests shifting toward the front foot during the next swing. This is an instructional revision, not evidence that physical weight distribution was measured or that coaching effectiveness improved.

## **5.6 Coach C Session 1 Findings** {#5.6-coach-c-session-1-findings}

Coach C submitted one complete form-review response covering all 30 clips. Twenty-seven clips were judged Bad Form and three were judged Good Form. The Good Form clips were C008, C019 and C022. All 27 Bad Form judgments included deviation comments. The source workbook and comments were preserved unchanged.

**Table 11. Session 1 form judgments by frozen feedback status**

| Coach C form judgment | Corrective | Positive | Low confidence | Partial | Total |
|---|---:|---:|---:|---:|---:|
| Bad Form | 18 | 0 | 8 | 1 | 27 |
| Good Form | 1 | 2 | 0 | 0 | 3 |
| Total | 19 | 2 | 8 | 1 | 30 |

With the selected GRU configuration, 18 of 19 corrective outputs occur on clips Coach C judged Bad Form; the remaining corrective output is on C008, judged Good Form. C019 and C022, also judged Good Form, receive positive feedback on the limited assessed measurements. These associations do not establish that individual cues are valid or that the model classifies whole-form quality. Session 2 must judge the displayed suggestions independently of this descriptive comparison.

The actual coach instrument did not collect independent Drive/Lob/Topspin labels. Comments disputing an intended category, including the description of C030 as an under-spin serve, were preserved as comments and were not converted into a complete classification reference set. No independent Coach C subtype accuracy or whole-form classifier accuracy is reported.

## **5.7 Session 2 Analysis Plan and Pending Results** {#5.7-session-2-analysis-plan-and-pending-results}

The researchers plan to deliver Session 2 after a one-day interval from Session 1. The coach will view the same videos with the frozen model prediction and approved feedback, without access to Session 1 answers through the instrument. This separation may reduce immediate recall but does not guarantee that the coach will forget earlier judgments. The actual session dates and presentation version must be recorded before reporting the completed procedure.

Each clip requires one response: Valid, Not valid, or No coaching feedback to assess. Let V, I and U denote those counts and M denote unanswered items. All clips remain in the accounting, so V + I + U + M = 30. The primary result is 100V/30. A secondary result, 100V/(V + I), describes validity among assessed outputs and is unavailable when V + I = 0. U and M are distinct: the former is an explicit judgment about absence of assessable coaching, whereas the latter denotes incomplete data collection.

**Table 12. Session 2 results to be completed after submission**

| Measure | Result at this draft |
|---|---|
| Valid, Not valid, and No coaching feedback to assess counts | Pending |
| Primary validity percentage, V/30 | Pending |
| Secondary validity among assessed outputs, V/(V + I) | Pending |
| Missing responses | Not yet determined from a Session 2 export |
| Independently measured coaching effectiveness over subsequent attempts | Not measured by this instrument |

The scoring script preserves the source export, matches answers by clip ID, and rejects duplicate IDs or unknown response values. Multiple submissions require explicit selection. It does not silently combine attempts or convert Session 1 comments into ratings. It flags inconsistent use of the no-coaching option for documented review without changing the submitted answers. Synthetic test data used to verify the script are not research observations.

## **5.8 Verification, Limitations and Objective Attainment** {#5.8-verification-limitations-and-objective-attainment}

The current local verification includes 36 targeted Python checks for the desktop, hybrid and development-feedback code, together with Apps Script mock checks for exact feedback, batching, recovery and migration. The Session 2 scorer additionally passes 12 synthetic-input tests covering CSV/XLSX reading, denominator definitions, missing values, duplicate fields, multiple submissions and source preservation. These are implementation checks, not evidence of coaching validity. Physical-phone latency and live-camera performance remain unmeasured in the available record.

The evidence supports implementation of a multimodal prototype and achievement of the numerical stratified-CV target under documented GRU conditions. It does not establish feedback validity on all observed mechanics or reliable coach replacement. The selected GRU gate withholds coaching on 26.7% of the clips. Hip displacement remains an image-plane movement proxy, and paddle orientation remains an image-plane axis estimate rather than verified physical face tilt. One independent expert and a small, previously exposed evaluation set further limit generalization.

The final interpretation of contextual feedback validity must wait for Session 2. Results below a target or with limited coverage should be reported as unmet or partially met objectives, not removed from the evaluation. The distinction between a functioning implementation and an effective coaching system is central to the conclusions.

# **CHAPTER 6** {#chapter-6}

# **SUMMARY, CONCLUSIONS, AND RECOMMENDATIONS** {#summary-conclusions-and-recommendations}

## **6.1 Summary of the Study** {#6.1-summary-of-the-study}

The study developed a recorded-video prototype using the original five-fold GRU ensemble, a 0.60 confidence gate, separate hip/paddle measurements and deterministic practice cues. The earlier GRU–kNN5 hybrid remains an archived comparison. Auxiliary landing is excluded from the 30-clip evaluation. Original GRU was selected after Session 1 receipt but before Session 2 release, without retraining or adapting cues to Coach C’s comments.

The work included legacy-model audits, GRU retraining, augmentation and feature experiments, hybrid evaluation, and preparation of a two-session expert review. The final review instrument differs from the initial proposal: Coach C supplies overall-form judgments and comments in Session 1, then evaluates displayed feedback in Session 2. Independent subtype labels were intentionally omitted. This change limits external classification validation and must remain explicit in the final manuscript.

## **6.2 Summary of Findings** {#6.2-summary-of-findings}

The original GRU's historical mean stratified-CV accuracy was 86.7%, and the separate v2 retrained ensemble achieved 90.9% on 209 quality-accepted clips. Both exceed the original numerical target, but their training populations and procedures differ. The high development scores did not establish reliable transfer: the matched diagnostic comparison yielded 51.4% for the original GRU, 46.9% for the retrained GRU, and 49.7% for the comparison hybrid.

Raw-video augmentation improved a matched participant-held-out angle-model result from 36.4% to 47.8%, while its diagnostic result changed from 46.3% to 47.4%. Richer inputs and calibration produced mixed outcomes. These experiments support continued investigation of participant variation and representation, but do not establish a consistently superior replacement for the current evaluated configuration.

All 30 selected clips produced classifications and hip measurements; 26 produced paddle measurements. The selected original GRU configuration yields 19 corrective, two positive, eight low-confidence and one partial output. Substantive coaching coverage is 21/30 (70.0%), compared with 17/30 (56.7%) for the archived hybrid. This change measures availability rather than validated coaching quality.

Coach C judged 27 clips Bad Form and three Good Form in Session 1. The agreement between a Bad Form judgment and the presence of corrective feedback cannot establish the validity of that correction. Session 2 ratings and physical-device performance results remain pending. No final percentage for feedback validity, independent Coach C classification accuracy, or phone latency is claimed.

## **6.3 Conclusions** {#6.3-conclusions}

The completed work demonstrates that the proposed components can be assembled into a recorded-video analysis prototype with explicit classification, measurement and feedback outputs. The stated stratified-CV numerical target was reached in identified GRU experiments. However, the selected original GRU's external reliability, feedback coverage and physical-device performance must be assessed separately rather than inferred from those development scores.

The present evidence does not support a claim that the system replaces a coach. It provides a limited set of automatic practice cues and uncertainty notices. Substantial withholding, imperfect participant transfer, unvalidated physical interpretations of some proxies, and pending Session 2 evidence prevent a stronger effectiveness conclusion. A complete thesis can report these limitations and partially attained objectives, but the pending outcomes must be filled from collected evidence rather than favorable assumptions.

These conclusions are provisional with respect to contextual feedback validity and deployment latency. They should be finalized only after the remaining measurements are available, while retaining all 30 selected clips and the frozen evaluated version in the reported analysis.

## **6.4 Recommendations** {#6.4-recommendations}

First, complete Session 2 with the current form and a documented interval, preserving the model, confidence gate and feedback wording. Report the three response categories and full-set validity alongside substantive coaching coverage. If the coach provides clarification, retain the original response and document any interpretation separately. Refinements based on this evaluation should be identified as a later development version requiring new validation.

Second, measure end-to-end core-pipeline latency and actual behavior on the intended Android devices. Separate camera acquisition, inference and optional landing processing where relevant, and retain the device specifications and measurement procedure. Desktop runtimes and successful builds are insufficient substitutes.

Third, when resources permit, prioritize additional independent training participants and clearer label provenance over repeatedly generating variants of the same players. Use participant-separated development checks and consistent extraction/windowing to evaluate proposed improvements. Independently labeled mechanics or contact events may support later validation, but they are not additional requirements imposed on Coach C for the present review.

Fourth, restrict each feedback rule to measurements the implementation can support. Do not claim verified ground-force transfer, paddle-face opening or closing, ball spin, serve legality, wrist lag or whole-form correctness without appropriate validation. Preserve useful partial results while making the scope of every cue clear.

Finally, reconcile the final manuscript with the selected original GRU and separate measurement streams, actual instrument, training populations, prior diagnostic exposure and recorded timing change. Maintain the distinction between intended recording categories, coach form judgments, predicted subtypes, and feedback-validity ratings. These records make the final conclusions traceable even when some intended performance objectives are not met.
