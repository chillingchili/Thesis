"""Apply the evidence-based October 12 manuscript and site update once."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return (ROOT / path).read_text(encoding='utf-8-sig')

def write(path, value):
    (ROOT / path).write_text(value, encoding='utf-8')

def main():
    chapter_path = 'docs/thesis/CHAPTERS_5_6.md'
    old = read(chapter_path)
    if 'Table 13. Recorded-video Android benchmark' in old:
        raise SystemExit('Already applied; edit the current manuscript directly.')
    ch = old
    start = ch.index('This chapter reports')
    end = ch.index('\n\nThree kinds', start)
    ch = ch[:start] + "This chapter reports the completed classifier experiments, frozen multimodal outputs for all 30 selected clips, Coach C's two-session review, and recorded-video measurements on the Xiaomi Note 12. Results were consolidated on 12 October 2026. Submitted ratings are preserved without reinterpretation. Recorded-file processing and the proposed live capture-end-to-displayed-feedback latency target are treated as different measurements." + ch[end:]
    ch = ch.replace('Session 2 must judge the displayed suggestions independently of this descriptive comparison.', 'The Session 2 ratings below assess the displayed outputs separately from this descriptive comparison.')
    start = ch.index('## **5.7 ')
    end = ch.index('# **CHAPTER 6**', start)
    ch = ch[:start] + '''## **5.7 Coach C Session 2 Findings** {#5.7-coach-c-session-2-findings}

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

**Table 14. Objective attainment and evidence limits**

| Objective | Supported conclusion |
|---|---|
| At least 85% stratified GRU CV | Numerically reached in historical original-GRU and separate v2 experiments; populations differ from one another and cannot be treated as the same proposal dataset |
| Transfer to other players | Limited: selected original GRU achieved 51.4% on the previously inspected 175-clip diagnostic set |
| Contextual feedback review | Completed: 27/30 rated Valid, with category-interpretation limitations; no prespecified numerical pass threshold |
| Substantive feedback availability | 21/30 clips (70.0%); availability is distinct from validity |
| Live feedback within two seconds after capture | Not verified by the recorded-file benchmark; no claim of target attainment |
| Capture retention and subtype-distinguishability gates | Final gate outcomes not established by the available evidence |
| Independent Coach C subtype accuracy | Unavailable: independent subtype labels were intentionally omitted |

The evidence supports an implemented multimodal prototype and a completed descriptive expert review. It does not establish reliable coach replacement. Hip movement is an image-plane displacement proxy; paddle orientation is an image-plane axis estimate rather than verified physical face tilt. Low-confidence withholding, limited participant diversity, retrospective model selection before Session 2 release, prior exposure of evaluation clips, one expert and the response-category ambiguity constrain the conclusions. These limitations remain part of the final findings rather than reasons to discard unfavorable clips.

''' + ch[end:]
    ch = ch.replace('preparation of a two-session expert review', 'completion of a two-session expert review and recorded-video Android benchmarks')
    start = ch.index('Coach C judged 27 clips', ch.index('## **6.2'))
    end = ch.index('## **6.4', start)
    ch = ch[:start] + '''Coach C judged 27 clips Bad Form and three Good Form in Session 1. Session 2 contained 27 Valid, one Not valid and two No coaching feedback to assess answers, with none missing. The fixed primary result was 90.0% (27/30); the secondary result was 96.4% (27/28). Eight Valid answers concern notice-only outputs, while two corrective outputs were marked no coaching. All answers were retained, so these figures describe submitted output ratings rather than an unqualified measure of corrective-feedback accuracy.

The Xiaomi recorded-file benchmark completed all nine measured runs per configuration. Median processing was 68.619 seconds before and 68.576 seconds after crop optimization, with unchanged compared outputs. The component improvement did not materially improve overall processing time. The proposed live two-second boundary remains unverified.

## **6.3 Conclusions** {#6.3-conclusions}

The study implemented an original-GRU multimodal recorded-video prototype that produces classifications, limited movement/orientation measurements, practical cues and explicit uncertainty notices. Identified GRU experiments reached the historical stratified-CV numerical target, but diagnostic performance shows that this result cannot be extended to reliable recognition across new participants.

Coach C's completed review supplies favorable submitted ratings for most displayed outputs, accompanied by a material interpretation limitation. Only 70.0% of the selected clips contain substantive coaching, and the combined availability/validity choices were not used consistently with frozen output status. The study therefore supports the feasibility of producing and reviewing limited automated practice cues, not a claim of comprehensive biomechanical assessment, improved player performance or coach replacement.

The thesis objectives were partially attained. Implementation and descriptive expert review are complete within their stated scope; reliable participant transfer and the live latency requirement have not been demonstrated. Reporting these outcomes completes the present findings without converting missing evidence or limited results into successful objectives.

''' + ch[end:]
    start = ch.index('First, complete Session 2')
    end = ch.index('Third, when resources permit', start)
    ch = ch[:start] + '''First, preserve the completed Session 2 responses and report validity alongside coverage and the category/status cross-tabulation. Any later coach clarification should be recorded separately without overwriting the original answers. A future instrument should ask feedback availability and validity separately, and distinguish substantive coaching from uncertainty notices. Changes informed by this review require a later model version and new validation data.

Second, measure live capture-end-to-displayed-feedback latency on the intended Android devices using a clearly defined timing boundary. Optimize the dominant decoding/pose/paddle work and verify output behavior after each change. The completed recorded-video benchmark and small crop optimization do not substantiate the two-second live target. Additional clips, devices and controlled repeated measurements would improve deployment evidence.

''' + ch[end:]
    ch = ch.replace('Finally, reconcile the final manuscript', 'Finally, maintain the final manuscript')
    full_path = 'docs/thesis/Pascua-Leones_Thesis_Current.md'
    full = read(full_path)
    assert old.strip() in full
    full = full.replace(old.strip(), ch.strip(), 1)
    replacements = {
        '5.7 Session 2 Analysis Plan and Pending Results': '5.7 Coach C Session 2 Findings',
        '#5.7-session-2-analysis-plan-and-pending-results': '#5.7-coach-c-session-2-findings',
        'Table 12. Pending Session 2 Results': 'Table 12. Session 2 Submitted Ratings and Denominators',
        'Session 2 and physical-device timing remain pending.': 'Session 2 is complete; recorded-file Android timing is available, while the live two-second timing boundary remains unverified.',
        'Session 2 after planned interval; ratings pending': 'Session 2 completed; category interpretation qualified',
        'Android original-GRU default; phone validation pending': 'Android original-GRU default; recorded-file benchmark complete',
        'nor pending Session 2 judgments': 'nor Session 2 judgments',
        'Expert review and pending phone validation': 'Expert review complete; live timing unverified',
        'pending expert review': 'completed expert review with interpretation limitations',
    }
    for a, b in replacements.items():
        full = full.replace(a, b)
    lines = full.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('This study develops and evaluates a multimodal computer-vision prototype'):
            lines[i] = "This study developed and evaluated a multimodal computer-vision prototype for pickleball serve classification and rule-based practice feedback. The selected original five-fold GRU ensemble operates with a 0.60 confidence gate and separate hip-movement and image-plane paddle measurements. The earlier GRU–kNN5 hybrid remains a comparison. Historical original-GRU stratified cross-validation was 86.7%, while a matched, previously inspected 175-clip diagnostic set yielded 51.4%, compared with 46.9% for the retrained v2 ensemble and 49.7% for the hybrid. On 30 selected clips, the system produced 19 corrective, two positive, eight low-confidence and one partial output, giving 70.0% substantive coaching coverage. Coach C judged 27 clips Bad Form and three Good Form in Session 1. Session 2 returned 27 Valid, one Not valid and two No coaching feedback to assess responses: 90.0% rated Valid across all clips and 96.4% among Valid/Not valid answers. Eight Valid answers concerned notice-only outputs and two corrective outputs were marked no coaching, limiting interpretation of these percentages. The current form's 30 outputs matched frozen reports; historical submission-time presentation was not independently reconstructed. Xiaomi recorded-video processing had a 68.576-second optimized median, with no meaningful total improvement over baseline. This timing does not verify the proposed live two-second target. The study demonstrates a limited prototype and completed descriptive review, with partially attained objectives; it does not establish independent Coach C subtype accuracy, improved serving performance or coach replacement."
        elif line.startswith('The implemented mobile proof of concept'):
            lines[i] = line[:line.index('Planned phone measurements')] + 'Recorded-file benchmarking is complete on the Xiaomi Note 12 running Android 14; the optimized median was 68.576 seconds across nine measured runs of three clips. The second planned device profile and live capture-end-to-displayed-feedback timing remain unverified. Successful builds do not establish the live latency target.'
        elif line.startswith('* **Hardware-Specific Latency Constraints:**'):
            lines[i] = '* **Hardware-Specific Latency Constraints:** Recorded-file measurements are available for one Xiaomi Note 12. Live capture-to-display behavior and the second planned device profile remain unverified; these measurements do not establish minimum operating specifications.'
        elif line.startswith('This chapter describes the research setting'):
            lines[i] = 'This chapter describes the research setting, data preparation, implemented multimodal pipeline and expert-review procedure. It distinguishes the proposed production design from the recorded-clip configuration actually evaluated. Both Coach C sessions and the Xiaomi recorded-file benchmark are complete as of 12 October 2026; live timing remains unverified. Changes to the initial proposal are disclosed explicitly.'
        elif line.startswith('4. The two-second core-pipeline target'):
            lines[i] = '4. The two-second live core-pipeline target requires a capture-end-to-displayed-feedback measurement with hardware and timing boundaries reported. The completed recorded-file benchmark measures a different interval and does not establish this target. Auxiliary landing time, if exercised, is reported separately.'
        elif line.startswith('The implementation and automated experiments are complete'):
            lines[i] = 'The implementation, automated experiments, both Coach C sessions and Xiaomi recorded-file benchmarks are complete to the extent reported in Chapter 5. Live timing remains unverified. The following approaches distinguish the proposed architecture from the software actually evaluated.'
    full = '\n'.join(lines) + '\n'
    write(chapter_path, ch)
    write(full_path, full)
    manifest = json.loads(read('docs/thesis/revision_manifest.json'))
    manifest.update(working_copy_sha256=hashlib.sha256((ROOT / full_path).read_bytes()).hexdigest(), session2_results='complete_with_category_interpretation_limitation', device_latency='recorded_file_complete_live_boundary_unverified', results_tables_renumbered='7–14', findings_finalized='2026-10-12')
    manifest['finalization_evidence'] = ['evaluation/coach_c/session2_gru/coach_c_scores_20261012/scores.json', 'evaluation/coach_c/session2_gru/coach_c_scores_20261012/form_verification.json', 'evaluation/phone_latency/note12_optimized_full_20261010/comparison.json']
    write('docs/thesis/revision_manifest.json', json.dumps(manifest, indent=2) + '\n')
    print('Updated Chapters 5–6, full manuscript and revision trace.')

if __name__ == '__main__':
    main()
