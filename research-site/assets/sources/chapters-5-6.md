# **CHAPTER 5** {#chapter-5}

# **RESULTS AND DISCUSSION** {#results-and-discussion}

## **5.1 Scope of the Reported Results** {#5.1-scope-of-the-reported-results}

This chapter reports the completed classifier experiments, frozen multimodal outputs for all 30 selected clips, Coach C's two-session review, and recorded-video measurements on the Xiaomi Note 12. Results were consolidated on 12 October 2026. Submitted ratings are preserved without reinterpretation. Recorded-file processing and the proposed live capture-end-to-displayed-feedback latency target are treated as different measurements.

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

With the selected GRU configuration, 18 of 19 corrective outputs occur on clips Coach C judged Bad Form; the remaining corrective output is on C008, judged Good Form. C019 and C022, also judged Good Form, receive positive feedback on the limited assessed measurements. These associations do not establish that individual cues are valid or that the model classifies whole-form quality. The Session 2 ratings below assess the displayed outputs separately from this descriptive comparison.

The actual coach instrument did not collect independent Drive/Lob/Topspin labels. Comments disputing an intended category, including the description of C030 as an under-spin serve, were preserved as comments and were not converted into a complete classification reference set. No independent Coach C subtype accuracy or whole-form classifier accuracy is reported.

## **5.7 Coach C Session 2 Findings** {#5.7-coach-c-session-2-findings}

The Session 2 export received on 12 October 2026 contains one complete submission covering all 30 clips. The form presents each video, its frozen original-GRU classification and feedback, and three choices: Valid, Not valid, or No coaching feedback to assess. Session 1 answers are not displayed within this instrument. A one-day separation was planned to reduce immediate recall; the available presentation check does not independently establish the actual viewing interval or eliminate recall bias.

Let V, I and U denote the three response counts and M denote missing answers. The fixed primary measure is 100V/30; the secondary measure is 100V/(V + I). All clips remain in the accounting, with V + I + U + M = 30. These definitions were retained when scoring the submitted export.

**Table 12. Session 2 submitted ratings and denominators**

| Measure | Observed result |
|---|---:|
| Valid | 27 |
| Not valid | 1 |
| No coaching feedback to assess | 2 |
| Missing responses | 0 |
| Primary: rated Valid among all clips, V/30 | 27/30 (90.0%) |
| Secondary: rated Valid among Valid/Not valid answers, V/(V + I) | 27/28 (96.4%) |
| Frozen corrective or positive output coverage | 21/30 (70.0%) |

C001 received Not valid. C021 and C024 received No coaching feedback to assess even though both frozen outputs contain a hip-movement corrective cue. Among the 21 corrective or positive outputs, 19 were rated Valid and two received the no-coaching response. Among nine uncertainty or partial notices without substantive coaching, eight were rated Valid and one Not valid. Thus, the 90.0% primary result describes the coach's submitted ratings across displayed outputs; it is not an estimate of actionable-coaching accuracy, subtype accuracy, or improvement in subsequent serves.

These 11 category/status discrepancies were retained as interpretation flags, not automatically corrected. The form combines validity and availability in one question. A Valid answer to an uncertainty notice may concern the notice itself, but the coach's intent cannot be inferred from the export. Equally, the two no-coaching answers cannot be silently changed to invalid or excluded. The secondary 96.4% therefore also requires the category-interpretation limitation.

On 12 October, all 30 classification and feedback texts in the current Google Form editor matched the frozen reports after whitespace normalization. C021 and C024 displayed the expected corrective text. This rules out a discrepancy in the current editor; it does not independently reconstruct what was displayed at the exact time of submission. The original workbook, source hash, clip-level scores, frozen report hashes and subsequent form-verification record are preserved. No model, threshold, feedback text or submitted answer was changed during this analysis.

The review provides descriptive evidence from one expert on a small, previously exposed sample. No new numerical passing threshold was imposed after receipt. It does not establish inter-rater reliability, participant-independent coaching effectiveness or the ability to replace a coach.

## **5.8 Verification, Limitations and Objective Attainment** {#5.8-verification-limitations-and-objective-attainment}

The Session 2 scorer passed 12 synthetic-input tests covering export parsing, denominator definitions, missing values, duplicate fields, multiple submissions and source preservation. Android build, unit, lint and targeted device checks supported the benchmark implementation. Software checks establish implementation behavior, not biomechanical correctness or coaching effectiveness.

**Table 13. Recorded-video Android benchmark**

| Measure | Baseline | Optimized |
|---|---:|---:|
| Successful measured runs | 9/9 | 9/9 |
| Median processing time | 68.619 s | 68.576 s |
| Slowest measured run | 104.650 s | 94.284 s |
| Compared outputs changed after optimization | — | 0/9 |

Each configuration processed three saved clips (C001, C002 and C007) in three measured rounds, excluding one warm-up run. The Xiaomi Note 12 ran Android 14. The timed interval begins when the recorded file is opened and ends when feedback is computed, including decoding, pose extraction, hip/paddle analysis, GRU inference and feedback construction. Initial model/pose setup is recorded separately. These debug-build, charging-device measurements do not time live capture completion or the final screen display; repeated runs of three clips are not nine independent recordings.

The crop-preparation optimization reduced its measured component time from approximately 5.20 to 1.38 ms with identical tested pixels and angles. Overall median processing fell by only 0.06%, which does not demonstrate a meaningful total speedup. The same model assets and clips were used, and all nine compared predictions and feedback outputs were unchanged. Separate sequential runs do not isolate thermal or scheduling variation. Android's temporal window also differs from the Coach C recorded-clip path, so these checks do not establish desktop/phone output parity.

### Post-review mobile replay and automatic feedback refinement

After the completed Coach C review, the same 30 recordings were replayed once per configuration through the live Android processing path on the Xiaomi. The source files and original five GRU assets were hash-verified. CPU pose with priority scheduling was compared with GPU pose, followed by GPU pose with a time-normalized automatic trigger and a revised feedback policy. These are development comparisons on previously inspected clips, not a new independent test. Coach C did not evaluate these updated phone outputs.

| Mobile configuration | Clips with a classification, including low confidence | Corrective/positive feedback | Insufficient-data notices | No completed output |
|---|---:|---:|---:|---:|
| CPU pose and pose-priority scheduling | 6/30 | 2/30 | 23/30 | 1/30 |
| GPU pose with the existing trigger | 25/30 | 5/30 | 1/30 | 4/30 |
| GPU pose, time-normalized trigger and guarded general cues | 24/30 | 18/30 | 3/30 | 3/30 |

The final configuration produced 18 corrective, three partial, three low-confidence and three insufficient-data outputs; three clips had no completed result. Thus, corrective-feedback availability was 60.0%, compared with 6.7% in the CPU configuration. The 18/30 mobile result must remain separate from the 21/30 frozen-output coverage and 27/30 Valid ratings in the expert study.

The GRU weights and 0.60 classification threshold were unchanged. Motion was normalized to an 80 ms reference interval and smoothed with a 200 ms time constant to reduce dependence on processing frame rate. General mechanics cues could be emitted despite uncertain classification only when the capture retained at least 32 real poses spanning at least 1.5 seconds, the relevant measurement was available, and the existing rule explicitly applied to all three serve types. Subtype-specific advice remained withheld. Eleven of the 18 corrective outputs used this classification-independent path and explicitly disclosed classification uncertainty. The coverage gain therefore partly reflects a feedback-policy change; it is not an equivalent improvement in classifier accuracy or independently established advice validity.

Ten final classifications met the 0.60 threshold; seven agreed with the stored recording category. Across all 24 available classifications, including low confidence, 14 agreed. Recording categories are not independent Coach C subtype labels. Paddle measurements were available for only one of the 30 final results, leaving an important multimodal limitation. Low-confidence subtype labels are displayed as uncertain in the updated app.

Median detected-end-to-UI-assignment time was 126 ms and the maximum was 423 ms among the 27 final displayed outputs. However, C005 appeared 3.126 seconds after the source file ended; three outputs appeared before file end and three clips had no completed output. These observations do not establish consistent physical-serve-end-to-visible-feedback latency within two seconds. The replay supplied synthetic idle images before and after each video, excluded them from classifier and mechanics evidence, and excluded initial decoding/model setup from live timing. It did not test camera exposure, orientation or actual player capture. The final summary uses each clip's last displayed result; all 28 displayed events, including a multiple-output clip, were preserved.

The comparisons were single-pass and sequential, with no counterbalancing for device temperature. Trigger and feedback changes were combined, so their separate effects were not isolated. Repeated use of these 30 clips for development further limits generalization claims. Forty-three JVM tests, debug build and lint checks passed for the updated implementation; software checks do not validate biomechanical advice. The original expert-review artifacts and ratings remain unchanged. Reproducible logs, per-clip messages and build identities are retained in `evaluation/coach_c/mobile_gpu_rate_20261012/`.

A subsequent bounded pilot increased the automatic quiet-hold interval from 350 to 650 ms on eight selected failure/control clips. Corrective/positive output coverage rose from 2/8 to 3/8 because one insufficient-data result became positive. However, the same three clips had no completed result and C005 remained late at 3.106 seconds after file end. With no independent validation or repetition, this small availability gain was insufficient to justify replacing the current setting; the longer hold was not adopted. This selected, single-pass pilot is reported as a negative development result, not a new independent evaluation.

**Table 14. Objective attainment and evidence limits**

| Objective | Supported conclusion |
|---|---|
| At least 85% stratified GRU CV | Numerically reached in historical original-GRU and separate v2 experiments; populations differ from one another and cannot be treated as the same proposal dataset |
| Transfer to other players | Limited: selected original GRU achieved 51.4% on the previously inspected 175-clip diagnostic set |
| Contextual feedback review | Completed: 27/30 rated Valid, with category-interpretation limitations; no prespecified numerical pass threshold |
| Substantive feedback availability | Frozen expert-review outputs: 21/30 (70.0%); post-review phone outputs: 18/30 (60.0%), not independently rated |
| Live feedback within two seconds after capture | Not established: replay UI updates were fast after detection, but one occurred 3.126 s after file end and three clips had no completed output |
| Capture retention and subtype-distinguishability gates | Final gate outcomes not established by the available evidence |
| Independent Coach C subtype accuracy | Unavailable: independent subtype labels were intentionally omitted |

The evidence supports an implemented multimodal prototype and a completed descriptive expert review. It does not establish reliable coach replacement. Hip movement is an image-plane displacement proxy; paddle orientation is an image-plane axis estimate rather than verified physical face tilt. Low-confidence withholding, limited participant diversity, retrospective model selection before Session 2 release, prior exposure of evaluation clips, one expert and the response-category ambiguity constrain the conclusions. These limitations remain part of the final findings rather than reasons to discard unfavorable clips.

# **CHAPTER 6** {#chapter-6}

# **SUMMARY, CONCLUSIONS, AND RECOMMENDATIONS** {#summary-conclusions-and-recommendations}

## **6.1 Summary of the Study** {#6.1-summary-of-the-study}

The study developed a recorded-video prototype using the original five-fold GRU ensemble, a 0.60 confidence gate, separate hip/paddle measurements and deterministic practice cues. The earlier GRU–kNN5 hybrid remains an archived comparison. Auxiliary landing is excluded from the 30-clip evaluation. Original GRU was selected after Session 1 receipt but before Session 2 release, without retraining or adapting cues to Coach C’s comments.

The work included legacy-model audits, GRU retraining, augmentation and feature experiments, hybrid evaluation, and completion of a two-session expert review and recorded-video Android benchmarks. The final review instrument differs from the initial proposal: Coach C supplies overall-form judgments and comments in Session 1, then evaluates displayed feedback in Session 2. Independent subtype labels were intentionally omitted. This change limits external classification validation and must remain explicit in the final manuscript.

## **6.2 Summary of Findings** {#6.2-summary-of-findings}

The original GRU's historical mean stratified-CV accuracy was 86.7%, and the separate v2 retrained ensemble achieved 90.9% on 209 quality-accepted clips. Both exceed the original numerical target, but their training populations and procedures differ. The high development scores did not establish reliable transfer: the matched diagnostic comparison yielded 51.4% for the original GRU, 46.9% for the retrained GRU, and 49.7% for the comparison hybrid.

Raw-video augmentation improved a matched participant-held-out angle-model result from 36.4% to 47.8%, while its diagnostic result changed from 46.3% to 47.4%. Richer inputs and calibration produced mixed outcomes. These experiments support continued investigation of participant variation and representation, but do not establish a consistently superior replacement for the current evaluated configuration.

All 30 selected clips produced classifications and hip measurements; 26 produced paddle measurements. The selected original GRU configuration yields 19 corrective, two positive, eight low-confidence and one partial output. Substantive coaching coverage is 21/30 (70.0%), compared with 17/30 (56.7%) for the archived hybrid. This change measures availability rather than validated coaching quality.

Coach C judged 27 clips Bad Form and three Good Form in Session 1. Session 2 contained 27 Valid, one Not valid and two No coaching feedback to assess answers, with none missing. The fixed primary result was 90.0% (27/30); the secondary result was 96.4% (27/28). Eight Valid answers concern notice-only outputs, while two corrective outputs were marked no coaching. All answers were retained, so these figures describe submitted output ratings rather than an unqualified measure of corrective-feedback accuracy.

The Xiaomi recorded-file benchmark completed all nine measured runs per configuration. Median processing was 68.619 seconds before and 68.576 seconds after crop optimization, with unchanged compared outputs. The component improvement did not materially improve overall processing time. The proposed live two-second boundary remains unverified.

The post-review phone refinement increased corrective-feedback availability from 2/30 to 18/30 on reused diagnostic recordings. Eleven outputs used guarded, classification-independent mechanics rules. Three clips still produced no completed output and one result was late relative to file end. These updated outputs have not received independent expert validation and do not inherit the earlier 90.0% submitted-validity result.

## **6.3 Conclusions** {#6.3-conclusions}

The study implemented an original-GRU multimodal recorded-video prototype that produces classifications, limited movement/orientation measurements, practical cues and explicit uncertainty notices. Identified GRU experiments reached the historical stratified-CV numerical target, but diagnostic performance shows that this result cannot be extended to reliable recognition across new participants.

Coach C's completed review supplies favorable submitted ratings for most displayed outputs, accompanied by a material interpretation limitation. Only 70.0% of the selected clips contain substantive coaching, and the combined availability/validity choices were not used consistently with frozen output status. The study therefore supports the feasibility of producing and reviewing limited automated practice cues, not a claim of comprehensive biomechanical assessment, improved player performance or coach replacement.

The mobile follow-up shows that acquisition and feedback policy materially affect output availability even when GRU weights remain fixed. It supports feasibility of limited automatic practice cues, while exposing unresolved capture, paddle-measurement, classification and latency failures.

The thesis objectives were partially attained. Implementation and descriptive expert review are complete within their stated scope; reliable participant transfer and the live latency requirement have not been demonstrated. Reporting these outcomes completes the present findings without converting missing evidence or limited results into successful objectives.

## **6.4 Recommendations** {#6.4-recommendations}

First, preserve the completed Session 2 responses and report validity alongside coverage and the category/status cross-tabulation. Any later coach clarification should be recorded separately without overwriting the original answers. A future instrument should ask feedback availability and validity separately, and distinguish substantive coaching from uncertainty notices. Changes informed by this review require a later model version and new validation data.

Second, measure live capture-end-to-displayed-feedback latency on the intended Android devices using a clearly defined timing boundary. Optimize the dominant decoding/pose/paddle work and verify output behavior after each change. The completed recorded-video benchmark and small crop optimization do not substantiate the two-second live target. Additional clips, devices and controlled repeated measurements would improve deployment evidence.

Third, when resources permit, prioritize additional independent training participants and clearer label provenance over repeatedly generating variants of the same players. Use participant-separated development checks and consistent extraction/windowing to evaluate proposed improvements. Independently labeled mechanics or contact events may support later validation, but they are not additional requirements imposed on Coach C for the present review.

Fourth, restrict each feedback rule to measurements the implementation can support. Do not claim verified ground-force transfer, paddle-face opening or closing, ball spin, serve legality, wrist lag or whole-form correctness without appropriate validation. Preserve useful partial results while making the scope of every cue clear.

Finally, maintain the final manuscript with the selected original GRU and separate measurement streams, actual instrument, training populations, prior diagnostic exposure and recorded timing change. Maintain the distinction between intended recording categories, coach form judgments, predicted subtypes, and feedback-validity ratings. These records make the final conclusions traceable even when some intended performance objectives are not met.
