"""Apply the documented consistency pass to the Chapter 4–6 working revision."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'docs/thesis'


def main():
    target = DIR / 'Pascua-Leones_Thesis_Current.md'
    manifest_path = DIR / 'revision_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('consistency_pass'):
        raise SystemExit('Consistency pass already applied; preserve manual edits before rebuilding.')
    original = target.read_text(encoding='utf-8')
    text, tail = original.split('# **BIBLIOGRAPHY** {#bibliography}', 1)
    changes = []

    def paragraph(prefix, replacement):
        nonlocal text
        lines = text.split('\n')
        hits = [i for i, line in enumerate(lines) if line.strip().startswith(prefix)]
        if len(hits) != 1:
            raise ValueError(f'Expected one paragraph: {prefix!r}; found {len(hits)}')
        lines[hits[0]] = replacement
        text = '\n'.join(lines)
        changes.append(prefix)

    paragraph('Pickleball is among the fastest-growing', '''This study develops and evaluates a multimodal computer-vision prototype for pickleball serve classification and rule-based practice feedback. The evaluated classifier is a hybrid of the original five-fold Gated Recurrent Unit (GRU) ensemble and a distance-weighted five-nearest-neighbor (kNN5) classifier. Their class probabilities are averaged equally for Drive, Lob and Topspin predictions. Separate computations provide normalized hip movement and image-plane paddle orientation; a deterministic rule engine generates bounded coaching cues when fused confidence is at least 0.60. Auxiliary ball landing is outside the present coach evaluation. The study distinguishes the 210-clip annotated inventory, 209 quality-accepted retraining clips, the legacy 438-sample classifier population, and a previously inspected 175-clip diagnostic set. Historical original-GRU stratified cross-validation averaged 86.7%, while a separate retrained ensemble achieved 90.9%; these are GRU results, not hybrid validation. On the matched diagnostic set, original GRU, retrained GRU and the evaluated hybrid achieved 51.4%, 46.9% and 49.7% accuracy, respectively. For 30 selected clips, the frozen hybrid produced 16 corrective, one positive, 12 low-confidence and one partial output, giving substantive coaching coverage of 56.7%. Coach C judged 27 clips Bad Form and three Good Form in Session 1. Independent subtype labels were intentionally omitted. Session 2 feedback-validity ratings and physical-phone latency measurements remain pending. The contribution is an implemented and auditable prototype with documented generalization and coverage limitations; reliable coach replacement and attainment of the two-second mobile target are not established.''')
    paragraph('To bridge the gap between elite', '''To investigate accessible serve analysis, this study combines pose-based classification with separate movement and paddle measurements. The evaluated subtype classifier uses GRU and kNN5 probability fusion. The wider multimodal pipeline combines the resulting prediction with available measurements at the decision level, where researcher-authored IF-THEN rules informed by coach commentary select practice cues. This architecture supports inspection of each output and of missing measurements. It does not itself establish improved accuracy or valid coaching; these are separate evaluation questions.''')
    paragraph('This study develops a multimodal pickleball', '''This study develops and evaluates a multimodal pickleball serve-analysis prototype using a GRU–kNN5 hybrid classifier, deterministic hip-movement and image-plane paddle measurements, and coach-informed rules for practice feedback. Ball-placement mapping is an auxiliary capability requiring suitable additional video and is excluded from the current 30-clip evaluation.''')
    paragraph('* Tests the computational distinguishability', '* Examines classification of the recorded Drive, Lob and Topspin categories, while documenting the absence of independent expert subtype labels and the limits of the category reference.')
    paragraph('* Collects and partitions', '* Organizes recordings, participant membership and form annotations, distinguishing the annotated training inventory, legacy model population and previously exposed evaluation material.')
    paragraph('* Develops a multimodal model', '* Integrates the original GRU ensemble and kNN5 through equal probability fusion, then combines the classification with available hip and paddle measurements to generate deterministic, coach-informed feedback.')
    paragraph('* Evaluates the classification performance', '* Reports GRU development cross-validation and separate participant-held-out and matched diagnostic comparisons, including hybrid performance. The original 85% stratified-GRU target is retained with its stated scope; it is not redefined as a measured hybrid result.')
    paragraph('* Validates the contextual appropriateness', '* Evaluates the contextual validity of displayed feedback through Coach C’s two-session review, reporting valid, invalid, unavailable-coaching and missing-response counts separately. Session 2 and physical-device timing remain pending.')
    paragraph('For players, the system provides', 'For players, the prototype is intended to provide automatic practice cues from recorded serves. Current cues concern a limited set of hip-movement and paddle-reference observations. It does not directly measure foot loading, diagnose physical paddle-face opening or closing, judge overall form, or establish reliable replacement of an instructor. Its practical value depends on both feedback validity and the proportion of attempts receiving assessable coaching.')
    paragraph('For coaches, the system serves', 'For coaches, the prototype provides an inspectable implementation of selected coach-informed rules. Coach B’s form commentary informs rule construction, while Coach C reviews the evaluation clips and displayed outputs separately. Session 2 will assess contextual validity; expert approval of every cue cannot be assumed before those responses are collected.')
    paragraph('For pickleball organizations and academies,', 'For pickleball organizations and academies, the project provides a documented prototype and evaluation procedure for studying automatic practice feedback. Use in instruction or certification would require further validation across players and recording conditions. The current study does not establish a standardized biomechanical assessment or certification instrument.')
    paragraph('For future researchers, this study offers', 'For future researchers, the study contributes an auditable comparison of GRU, kNN5 fusion and separate development candidates; a pipeline connecting limited video measurements to deterministic feedback; and records of coaching coverage and expert form judgments. Recorded categories are distinguished from independently assigned subtype labels. The experiments expose participant-transfer and confidence-gating limitations, while physical-device latency and Session 2 feedback validity remain open measurements. Data availability is governed by the retained consent and repository arrangements; a complete public release is not assumed.')
    paragraph('This study focuses strictly on evaluation', 'This study focuses on recorded pickleball serves categorized as Drive, Lob and Topspin. The initial proposal included independent expert subtype labeling and a taxonomy fallback. That labeling question was removed from the actual coach instruments, so the fallback gate was not established as a completed validation step. Classification experiments use their recorded category references; the study cannot claim independent validation of the taxonomy from Coach C’s form judgments.')
    paragraph('To keep the focus on biomechanical', 'The study excludes other strokes and does not act as a serve-legality adjudicator. Recordings cover right-handed participants at the stated venues. Coach A supplies professional demonstrations and a subset of explicitly good-form reference recordings; professional status alone is not a form label. Coach B supplies construction-set form judgments and comments. Coach C supplies independent form judgments in Session 1 and reviews model feedback in Session 2. Neither actual instrument collects separate subtype labels, and Session 2 is output-informed rather than blinded to the model’s advice.')
    paragraph('The system architecture is delimited', 'The implemented mobile proof of concept uses native Android/Kotlin and TFLite assets, alongside a Python recorded-clip runner used for the Coach C batch. The evaluated classifier combines the original five-fold GRU ensemble with distance-weighted kNN5 at equal probability weights. Planned phone measurements cover the Snapdragon 685/6 GB and Dimensity 7300 Pro/8 GB device profiles. Their latency results remain pending; successful model export or desktop execution does not demonstrate the two-second target.')
    paragraph('Relatedly, while the functional requirements', 'The present evaluation uses prepared recordings. Live ingestion, autonomous triggering and automatic capture termination are production requirements rather than demonstrated results of this evaluation. The desktop and Android capture windows differ, so identical preprocessing and end-to-end behavior cannot be assumed. Immediate feedback is a latency objective to be measured after capture, not an established capability of continuous live analysis.')
    paragraph('* **Single-Camera Core Architecture:**', '* **Single-Camera Core Architecture:** The evaluated hip and paddle measurements are image-plane proxies from the primary view. They do not establish physical weight distribution, three-dimensional paddle-face orientation or ball spin. Auxiliary landing is disabled for all 30 clips; synchronized second-camera mobile operation is not validated here.')
    paragraph('* **Single-Expert Ground Truth Constraint:**', '* **Single-Expert Annotation Constraint:** Coach B supplies construction-set form judgments and Coach C supplies evaluation judgments on different clips. This does not support inter-rater reliability. Independent subtype labels were omitted, and recorded category labels must retain their actual provenance. Reference measurements from explicitly good-form Coach A clips do not establish a consensus biomechanical standard.')
    paragraph('* **Annotation Cost Constraint:**', '* **Annotation and Population Constraints:** The supplied recording inventory reports 812 clips, with a 210-clip construction annotation allocation and 30 evaluation clips. Legacy classifiers use a distinct 438-sample population, and 175 beginner clips have diagnostic exposure. These overlapping scopes cannot be summed or described as though every recording outside the annotation allocation was computationally unused. Only three participants contribute to the principal training inventory.')
    paragraph('* **Hardware-Specific Latency Constraints:**', '* **Hardware-Specific Latency Constraints:** Physical-device latency and live behavior remain unmeasured in the available evaluation record. The planned device profiles define measurement targets, not verified minimum operating specifications.')
    paragraph('* **Anatomical Edge Cases:**', '* **Unrepresented Movement and Body Configurations:** The study has not validated performance for participants using prostheses or with limb differences, or for other body configurations and movement conditions absent from its sample. Tracking and coaching reliability for those populations cannot be inferred from the current data.')
    # Retain cited literature; change only its claims about this study.
    old = 'Therefore, this study does not assume these subtypes transfer perfectly'
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if old in line:
            lines[i] = line[:line.index(old)] + 'For the present study, these terms define recorded categories rather than independently validated biomechanical classes. The initially proposed coach-label distinguishability gate and legal-category fallback were not completed under the revised instrument, which collects form judgments and comments without independent subtype labels.'
        if 'a gap this study addresses by incorporating bodyweight shift detection' in line:
            lines[i] = line.replace('a gap this study addresses by incorporating bodyweight shift detection into the core kinematic stream.', 'a motivation for the present hip-movement proxy. This proxy does not directly measure kinetic energy, ground reaction force or weight distribution, and findings from tennis are not treated as its validation.')
        if line.strip().startswith('Expert-grounded rule-based systems represent'):
            start = line.index('This study addresses that gap')
            lines[i] = line[:start] + 'In this study, researchers use Coach B’s form comments to inform a limited set of deterministic rules. The frozen outputs are then presented to Coach C in Session 2 for contextual validity judgments. This pending review does not constitute prior approval of every feedback string, and it is not blinded to the output being evaluated.'
        if line.strip().startswith('BlazePose, introduced'):
            start = line.index('Beyond serve type classification')
            lines[i] = line[:start] + 'In this study, hip-center landmarks also provide a normalized horizontal movement range. This deterministic proxy is separate from the GRU–kNN5 classifier and does not directly establish foot loading or force transfer. Performance for unrepresented body configurations has not been evaluated.'
        if 'bypassing the monocular constraint entirely' in line:
            lines[i] = line.replace('bypassing the monocular constraint entirely rather than attempting to resolve it within a single-camera pipeline.', 'providing a separate view of the court while retaining detection and planar-calibration uncertainty. The auxiliary stream is not exercised in the current coach evaluation.')
        if 'In this study, the GRU operates exclusively' in line:
            start = line.index('In this study, the GRU operates exclusively')
            lines[i] = line[:start] + 'In the evaluated implementation, the original five-fold GRU ensemble supplies one probability vector for subtype classification. A kNN5 branch supplies another from timing features. Hip movement is computed separately and deterministically from pose landmarks.'
    text = '\n'.join(lines)
    marker = '***Late Fusion for Heterogeneous Modalities***'
    assert text.count(marker) == 1
    text = text.replace(marker, '''***Hybrid GRU–kNN5 Classification in This Implementation***

The GRU branch models joint-angle sequences, while the distance-weighted kNN5 branch compares a 32-feature timing representation with its retained training bank. The evaluated system averages the five GRU fold probability vectors, then combines the resulting vector with the kNN5 vector at equal weights. For class c, p_hybrid(c) = 0.5 p_GRU(c) + 0.5 p_kNN5(c). The largest fused probability selects the predicted category; corrective coaching requires that value to be at least 0.60. This score is not assumed to be a calibrated probability of correctness. The matched experiments in Chapter 5 evaluate whether the combination improves performance; improvement is not guaranteed by fusion.

Here, “hybrid classifier” means combining two classifiers that use pose-derived representations. It is distinct from the wider “multimodal pipeline,” which combines classification with separate hip and paddle measurements at the rule layer. It also differs from intermediate-feature hybrid fusion discussed in the literature. Paddle measurements are not inputs to the GRU–kNN5 subtype probability average. The original 85% GRU cross-validation objective remains a GRU-specific development target, while hybrid diagnostic accuracy and coaching coverage are reported separately.

''' + marker)
    paragraph('Signal fusion is the process', 'Signal fusion combines information at the input, feature or decision level (Yang & Gulbahar, 2025; Li et al., 2023). Its benefit depends on the signals and evaluation conditions. This implementation uses probability-level GRU–kNN5 fusion for subtype prediction and a separate rule layer for combining classification with available measurements. The optional landing stream is inactive in the current evaluation.')
    paragraph('Integrating multiple data sources theoretically', 'The reviewed fusion literature distinguishes early input fusion, intermediate representation fusion and late decision fusion for heterogeneous signals (Yang & Gulbahar, 2025; Li et al., 2023). The present study keeps pose classification, hip measurement and paddle measurement separately inspectable and combines their outputs in deterministic rules. This design avoids requiring a paddle measurement for a subtype prediction, but does not guarantee equal signal quality or increased accuracy. A global fused-classification confidence gate can still withhold coaching even when a measurement is available.')
    # Replace outdated study diagrams with editable source diagrams, not altered historical images.
    diagrams = {
        4: 'flowchart TD\n A[Recording inventory and participant splits] --> B[Coach B form comments and training-only development]\n B --> C[Freeze original GRU plus kNN5 and measurement rules]\n C --> D[Private inference on all 30 clips]\n A --> E[Coach C Session 1 without model feedback]\n D --> F[Approved wording frozen before reading responses]\n E --> G[Planned one-day interval]\n F --> H[Session 2 shows frozen outputs]\n G --> H\n H --> I[Score all 30 responses; report coverage separately]',
        5: 'flowchart LR\n A[Primary video] --> B[Pose angles and timing]\n B --> C[Original five-fold GRU ensemble]\n B --> D[Distance-weighted kNN5]\n C --> E[Equal probability fusion]\n D --> E\n A --> F[Normalized hip movement]\n A --> G[Paddle image-plane reference state]\n E --> H[0.60 confidence gate and rule layer]\n F --> H\n G --> H\n H --> I[Subtype plus limited feedback or uncertainty notice]\n J[Auxiliary landing: disabled for this evaluation]',
        6: 'flowchart TD\n A[Prepared video and review ID] --> B[Python recorded-clip runner]\n B --> C[Pose features and original GRU-kNN5 assets]\n B --> D[Hip and paddle measurements]\n C --> E[Fused subtype and confidence]\n E --> F[Gate and deterministic feedback rules]\n D --> F\n F --> G[Frozen JSON reports]\n G --> H[Session 2 Google Form]\n I[Native Kotlin Android proof of concept]\n I -. Separate capture window; phone validation pending .-> C',
        7: 'flowchart TD\n A[Prepared primary-view clip] --> B[Extract usable pose frames]\n B --> C[GRU preprocessing: up to first 128 valid frames]\n B --> D[32 timing features for kNN5]\n C --> E[Average GRU fold probabilities]\n E --> F[50-50 fusion with kNN5 probabilities]\n D --> F\n B --> G[Hip movement proxy]\n A --> H[Paddle detector and contour geometry]\n F --> I{Fused confidence at least 0.60?}\n I -->|No| J[Withhold coaching; retain available results]\n I -->|Yes| K[Evaluate supported measurement rules]\n G --> K\n H --> K\n K --> L[Corrective, positive or partial output]',
    }
    for n, diagram in diagrams.items():
        text, count = re.subn(r'(?:\*\*)?!\[\]\[image'+str(n)+r'\](?:\*\*)?', '```mermaid\n'+diagram+'\n```', text)
        assert count == 1, n
    states = {
        8: '| Display field | Processing state |\n|---|---|\n| Input | Selected recorded clip / review ID |\n| Serve classification | Pending GRU–kNN5 inference |\n| Hip and paddle measurements | Pending or unavailable |\n| Coaching | Pending confidence and rule checks |\n| Ball landing | Disabled for this evaluation |',
        9: '| Display field | Completed state |\n|---|---|\n| Serve classification | Actual fused prediction and confidence |\n| Hip movement | Available measurement and reference comparison |\n| Paddle orientation | Image-plane reference state, if available |\n| Coaching | Frozen cue supported by available measurements |\n| Overall Good/Bad Form | Not a system output |',
        10: '| Condition | Display behavior |\n|---|---|\n| Fused confidence below 0.60 | Mark uncertainty and withhold coaching |\n| Paddle measurement missing | State unavailable; retain other results |\n| Supported hip cue with adequate confidence | May still provide that cue |\n| No supported corrective cue | State limited assessment; do not infer Good Form |',
    }
    for n, state in states.items():
        text, count = re.subn(r'\*\*!\[\]\[image'+str(n)+r'\]\*\*', state, text)
        assert count == 1, n
    paragraph('The single-screen layout places the camera', 'Figure 8 specifies the processing display for a selected recording. Pending fields do not imply that autonomous live triggering or bounce-based capture termination has been demonstrated. These output-state specifications describe the evaluated information; they are not screenshots or evidence of desktop/phone interface parity.')
    paragraph('Figure 6 retains the proposed', 'Figure 6 shows the evaluated recorded-clip architecture and the separate native Android proof of concept. Frozen reports connect inference to the coach instrument. The diagram does not establish equal capture windows or measured real-time phone performance.')
    paragraph('Figure 7 illustrates the intended', 'Figure 7 shows the recorded-clip data flow. Measurement availability is tracked separately from the global classification-confidence gate. No autonomous capture trigger or automatic temporary-file deletion is asserted by this diagram.')
    paragraph('Figure 9 is a proposed results', 'Figure 9 specifies completed output fields. The prediction is the fused subtype, and any positive coaching concerns only assessed measurements. Good/Bad overall form and physical paddle-face opening or closing are not validated system outputs.')
    paragraph('Figure 10 depicts partial delivery.', 'Figure 10 specifies insufficient-confidence and missing-measurement behavior. Available measurements remain accessible, but they do not override the frozen 0.60 classification-confidence gate.')
    text = text.replace('*Bottom-up Approach Model of the System*', '*Bottom-up Approach Model of the System*\n\n```mermaid\nflowchart BT\n A[Pose extraction and detector geometry] --> B[GRU-kNN5 classification and separate measurements]\n B --> C[Confidence gate and rule-based feedback]\n C --> D[Recorded-clip prototype and Android integration]\n D --> E[Expert review and pending phone validation]\n```')
    # Preserve historical schedules, but remove duplicate result-table numbering.
    text = text.replace('**Table 3 (continued)**', '**Table 4 (continued)**')
    chapter_file = DIR / 'CHAPTERS_5_6.md'
    chapters = chapter_file.read_text(encoding='utf-8')
    for n in range(8, 2, -1):
        text = text.replace(f'**Table {n}.', f'**Table {n+4}.')
        chapters = chapters.replace(f'**Table {n}.', f'**Table {n+4}.')
    # Rebuild navigation without obsolete typeset page numbers.
    begin = text.index('# **TABLE OF CONTENTS** {#table-of-contents}')
    end = text.index('# **LIST OF FIGURES** {#list-of-figures}')
    all_body = text + '# **BIBLIOGRAPHY** {#bibliography}' + tail
    headings = re.findall(r'^\s*#{1,3}\s+(.+?)\s+\{#([^}]+)\}\s*$', all_body, re.M)
    toc = '# **TABLE OF CONTENTS** {#table-of-contents}\n\n' + '\n\n'.join('['+name.replace('**','')+'](#'+anchor+')' for name,anchor in headings if anchor != 'table-of-contents') + '\n\n'
    text = text[:begin] + toc + text[end:]
    begin = text.index('# **LIST OF FIGURES** {#list-of-figures}')
    end = text.index('# **CHAPTER 1** {#chapter-1}')
    figure_names = ['The 33 Key Skeletal Points Tracked by BlazePose', 'Homography 3 x 3 Matrix', 'Camera Setup Diagram (proposed auxiliary arrangement)', 'Research Process Diagram', 'Conceptual Framework', 'System Architecture', 'Recorded-Clip Data Flow', 'Output State A — Processing', 'Output State B — Results Available', 'Output State C — Insufficient Data', 'Model Development Life Cycle', 'Bottom-up Approach Model']
    table_names = ['Mechanical Deviation Categories and Implemented Feedback Scope', 'Software Development Tools', 'Proposal Schedule', 'Implementation Schedule', 'Roles and Responsibilities', 'Expenses', 'Evaluation Populations', 'Matched Diagnostic Comparison', 'Augmentation and Representation Study', 'Frozen Output Availability', 'Session 1 Judgments by Feedback Status', 'Pending Session 2 Results']
    lists = '# **LIST OF FIGURES** {#list-of-figures}\n\n' + '\n\n'.join(f'Figure {i}. {name}' for i,name in enumerate(figure_names,1))
    lists += '\n\n# **LIST OF TABLES** {#list-of-tables}\n\n' + '\n\n'.join(f'Table {i}. {name}' for i,name in enumerate(table_names,1)) + '\n\n'
    text = text[:begin] + lists + text[end:]
    text = text.replace('*Wireframe Diagram State A \\- Detecting/Calculating*', '*Output State A — Processing*').replace('*Wireframe Diagram State B \\- Results Visible*', '*Output State B — Results Available*').replace('*Wireframe Diagram State C \\- Insufficient Data*', '*Output State C — Insufficient Data*')
    # Historical correspondence stays verbatim; qualify it outside the preserved letters.
    note = '\nThe appendix letters retain their historical wording, including superseded subtype-label requests. The actual instrument and its amendments are specified in Sections 4.2.1 and 4.10.5. Historical schedules likewise record plans, not verified completion.\n\n'
    text += note
    final = text + '# **BIBLIOGRAPHY** {#bibliography}' + tail
    target.write_text(final, encoding='utf-8')
    chapter_file.write_text(chapters, encoding='utf-8')
    manifest.update(working_copy_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    chapters_1_to_3_preserved=False, consistency_pass=True,
                    consistency_revised_paragraph_prefixes=changes,
                    diagram_replacements=[4,5,6,7,8,9,10,12],
                    results_tables_renumbered='7–12', original_source_unchanged=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    assert final.split('# **BIBLIOGRAPHY** {#bibliography}',1)[1] == tail
    print(f'Aligned {len(changes)} paragraphs, classifier definition, diagrams and navigation; source and historical appendices unchanged.')


if __name__ == '__main__':
    main()
