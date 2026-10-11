"""Record the user-authorized pre-release GRU selection in the current manuscript."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/thesis'


def main():
    p=D/'Pascua-Leones_Thesis_Current.md'
    s=p.read_text(encoding='utf-8')
    body,tail=s.split('# **BIBLIOGRAPHY** {#bibliography}',1)
    replacements={
      'This study develops and evaluates a multimodal computer-vision prototype': 'This study develops and evaluates a multimodal computer-vision prototype for pickleball serve classification and rule-based practice feedback. The selected classifier is the original five-fold Gated Recurrent Unit (GRU) ensemble, with no kNN contribution to its predictions. Separate normalized hip movement and image-plane paddle measurements support deterministic coaching rules at a 0.60 classification-confidence threshold. The earlier GRU–kNN5 hybrid remains a comparison model. Historical original-GRU stratified cross-validation averaged 86.7%, while a separate retrained ensemble achieved 90.9%; on the matched 175-clip diagnostic set, original GRU, hybrid and retrained GRU achieved 51.4%, 49.7% and 46.9%, respectively. Before Session 2 release, the researchers selected original GRU based on that existing comparison. The switch occurred after Session 1 receipt and is disclosed as a protocol amendment, not a preregistered independent test. The revised 30-clip run produced 19 corrective, two positive, eight low-confidence and one partial output, giving substantive coaching coverage of 70.0%. Coach C supplied 27 Bad Form and three Good Form judgments in Session 1; independent subtype labels were omitted. Session 2 feedback-validity ratings and physical-phone latency remain pending. The study demonstrates an implemented prototype with limited generalization and measurement scope, rather than established coach replacement.',
      '* Integrates the original GRU ensemble and kNN5': '* Uses the original five-fold GRU ensemble for subtype classification and combines its output with available hip and paddle measurements for deterministic feedback. Retains GRU–kNN5 as a separately reported comparison.',
      'The GRU branch models joint-angle sequences': 'The selected original GRU classifier averages the class probabilities of its five retained folds. For each class c, p_GRU(c) is the arithmetic mean of those five predictions. The class with maximum probability is displayed; corrective coaching requires that maximum to be at least 0.60. The comparison hybrid averages this GRU vector with a distance-weighted kNN5 vector computed from 32 timing features. That 50/50 combination is not used for the released GRU candidate. In the matched diagnostic comparison, original GRU scored 51.4% and the hybrid 49.7%; the 1.7-point difference does not establish statistically significant or universal superiority.',
      'Here, “hybrid classifier” means': 'The term hybrid classifier refers to the earlier GRU–kNN5 comparison. The selected system remains multimodal because the feedback layer combines GRU classification with distinct hip and paddle measurements. Removing kNN does not remove those measurement streams. Neither hip displacement nor paddle orientation is averaged into GRU subtype probabilities. GRU development accuracy, diagnostic accuracy, coaching coverage and expert-rated feedback validity remain separate measures.',
      '* Processing Mechanism: The original five-fold GRU ensemble': '* Processing Mechanism: The original five-fold GRU ensemble processes joint-angle sequences derived from pose landmarks. Its mean class probabilities determine the selected subtype. Hip-center displacement is computed separately and normalized by body height. The kNN5 fusion branch is retained for comparisons but excluded from the selected Session 2 configuration.',
      '* System Output (Serve Type):': '* System Output (Serve Type): The maximum-probability Drive, Lob or Topspin label from the original GRU ensemble, with its probability vector and confidence retained. Confidence is not assumed to be calibrated correctness.',
      'Figure 5 distinguishes probability-level': 'Figure 5 distinguishes GRU subtype classification from measurement integration. Hip and paddle observations inform feedback rather than subtype probabilities. Auxiliary landing remains disabled for this evaluation.',
      '* **Process:** Pose-derived angles': '* **Process:** Pose-derived angles feed the original five-fold GRU ensemble. Its mean probabilities supply the subtype and the 0.60 confidence gate. Hip movement and image-plane paddle orientation feed the deterministic rule layer. The auxiliary landing branch is inactive for this evaluation.',
      '3. The evaluated hybrid retains': '3. The selected original GRU retains the 0.60 threshold, now applied to the maximum ensemble probability without kNN averaging. The switch was authorized before Session 2 release on the basis of the earlier diagnostic comparison. No model retraining or threshold fitting used Coach C’s comments.',
      'The decision layer first applies': 'The decision layer applies the unchanged 0.60 gate to original-GRU ensemble probabilities. Supported measurement rules then supply corrective cues, limited positive cues or a partial notice. Below the gate, coaching is withheld. No per-measurement override was introduced.',
      'The 12 withheld outputs reflect': 'The earlier hybrid run withheld coaching on 12 clips and supplied substantive coaching on 17. Recomputing original GRU reduced low-confidence outputs to eight and increased substantive coaching to 21, using the same 30 clips, measurements and threshold. This is a coverage comparison, not a determination that the newly accepted predictions or suggestions are correct. Selection occurred after the researchers had seen Session 1, so it is disclosed as a retrospective pre-release amendment. The selection rationale is the previously available 175-clip diagnostic comparison, not fitting to the coach’s form comments.',
      'All 16 corrective outputs occurred': 'With the selected GRU configuration, 18 of 19 corrective outputs occur on clips Coach C judged Bad Form; the remaining corrective output is on C008, judged Good Form. C019 and C022, also judged Good Form, receive positive feedback on the limited assessed measurements. These associations do not establish that individual cues are valid or that the model classifies whole-form quality. Session 2 must judge the displayed suggestions independently of this descriptive comparison.',
      'The study developed a prototype for classifying': 'The study developed a recorded-video prototype using the original five-fold GRU ensemble, a 0.60 confidence gate, separate hip/paddle measurements and deterministic practice cues. The earlier GRU–kNN5 hybrid remains an archived comparison. Auxiliary landing is excluded from the 30-clip evaluation. Original GRU was selected after Session 1 receipt but before Session 2 release, without retraining or adapting cues to Coach C’s comments.',
      'All 30 selected clips produced classifications': 'All 30 selected clips produced classifications and hip measurements; 26 produced paddle measurements. The selected original GRU configuration yields 19 corrective, two positive, eight low-confidence and one partial output. Substantive coaching coverage is 21/30 (70.0%), compared with 17/30 (56.7%) for the archived hybrid. This change measures availability rather than validated coaching quality.',
    }
    lines=body.split('\n')
    for prefix,new in replacements.items():
        indices=[i for i,line in enumerate(lines) if line.strip().startswith(prefix)]
        assert len(indices)==1,(prefix,len(indices))
        lines[indices[0]]=new
    body='\n'.join(lines)
    pairs={
      'The evaluated subtype classifier uses GRU and kNN5 probability fusion.':'The selected subtype classifier uses the original five-fold GRU ensemble without kNN fusion.',
      'using a GRU–kNN5 hybrid classifier':'using the original five-fold GRU classifier',
      'The evaluated classifier combines the original five-fold GRU ensemble with distance-weighted kNN5 at equal probability weights.':'The selected classifier averages the five original GRU folds without kNN weighting. The Android default and the recorded-clip default both select this ensemble; capture-window parity is not assumed.',
      'This implementation uses probability-level GRU–kNN5 fusion for subtype prediction':'The comparison implementation uses probability-level GRU–kNN5 fusion; the selected implementation uses GRU alone for subtype prediction',
      'In the evaluated implementation, the original five-fold GRU ensemble supplies one probability vector for subtype classification. A kNN5 branch supplies another from timing features.':'In the selected implementation, the original five-fold GRU ensemble supplies the subtype probability vector. The archived comparison hybrid adds a kNN5 branch from timing features.',
      '***Hybrid GRU–kNN5 Classification in This Implementation***':'***Original GRU Selection and the Hybrid Comparison***',
      'separate from the GRU–kNN5 classifier':'separate from the GRU classifier',
      'using Android-compatible legacy GRU/kNN assets':'using Android-compatible legacy assets; the selected original-GRU revision recomputes classifications from the preserved pose caches',
      'the original GRU ensemble and kNN5 hybrid, alongside':'the original five-fold GRU ensemble, alongside',
      'The original hybrid remains the selected system for the Coach C evaluation, but that selection does not demonstrate that the hybrid is the most accurate classifier.':'The original GRU ensemble was selected before Session 2 release on the basis of this existing comparison. The previous hybrid outputs remain archived; no claim of statistically established superiority follows from the 1.7-point diagnostic difference.',
      'That policy was not adopted for the evaluated legacy hybrid.':'That policy was not adopted for the selected original GRU or the archived hybrid.',
      'The four feedback-status rows are mutually exclusive and total 30. The measurement-availability rows overlap and must not be added to those status counts. Corrective or positive feedback was available for 17/30 clips, or 56.7%. Corrective-only coverage was 16/30, or 53.3%.':'The four feedback-status rows are mutually exclusive and total 30. Measurement-availability rows overlap. The selected GRU supplies corrective or positive feedback on 21/30 clips (70.0%), including corrective feedback on 19/30 (63.3%).',
      '| Corrective feedback | 16 | 53.3% |':'| Corrective feedback | 19 | 63.3% |',
      '| Positive feedback on assessed measurements | 1 | 3.3% |':'| Positive feedback on assessed measurements | 2 | 6.7% |',
      '| Low-confidence coaching withheld | 12 | 40.0% |':'| Low-confidence coaching withheld | 8 | 26.7% |',
      '| Bad Form | 16 | 1 | 9 | 1 | 27 |':'| Bad Form | 18 | 0 | 8 | 1 | 27 |',
      '| Good Form | 0 | 0 | 3 | 0 | 3 |':'| Good Form | 1 | 2 | 0 | 0 | 3 |',
      '| Total | 16 | 1 | 12 | 1 | 30 |':'| Total | 19 | 2 | 8 | 1 | 30 |',
      'It does not establish that the selected hybrid meets the same CV target, that feedback is valid on all observed mechanics, or that the system can replace an expert coach. The global gate withholds coaching on 40.0% of the selected clips.':'It does not establish feedback validity on all observed mechanics or reliable coach replacement. The selected GRU gate withholds coaching on 26.7% of the clips.',
      "the selected hybrid's external reliability":"the selected original GRU's external reliability",
      'the evaluated hybrid.':'the comparison hybrid.',
      'the implemented hybrid, actual instrument':'the selected original GRU and separate measurement streams, actual instrument',
      'a fused subtype prediction':'an original-GRU subtype prediction',
      'the fused subtype':'the original-GRU subtype',
      'fused classification output':'GRU classification output',
      'fused-classification confidence gate':'GRU confidence gate',
      'fused-confidence gate':'classification-confidence gate',
      'global fused-confidence gate':'global classification-confidence gate',
      'Pending GRU–kNN5 inference':'Pending original GRU ensemble inference',
      'Actual fused prediction and confidence':'Actual GRU prediction and confidence',
      'Fused confidence below 0.60':'GRU confidence below 0.60',
      'The approved wording revision retained each clip\'s predictions, measurement values, triggered rules and feedback status.':'The earlier wording-only revision retained the hybrid predictions and triggers. The subsequent GRU selection changed classification and potentially triggered feedback; it reused the same measurements and approved message templates.',
      'The current Coach C set retains all 30 selected clips, including four with unavailable paddle measurements and 12 below the classification-confidence gate.':'The current Coach C set retains all 30 selected clips, including four with unavailable paddle measurements and eight below the selected GRU confidence gate.',
    }
    for a,b in pairs.items(): body=body.replace(a,b)
    # Diagrams explicitly show the new selected path and the timing amendment.
    diagrams=re.findall(r'```mermaid\n.*?\n```',body,re.S)
    assert len(diagrams)==5
    graphs=[
      'flowchart TD\n A[Training-only development and existing diagnostic comparison] --> B[Earlier hybrid outputs archived]\n C[Coach C Session 1 received] --> D[User selects original GRU before Session 2 release]\n A --> D\n B --> E[Reuse frozen pose and measurement caches]\n D --> F[Recompute all 30 GRU predictions; unchanged gate and cue templates]\n E --> F\n F --> G[Freeze new version and create separate form]\n G --> H[Session 2 after planned interval; ratings pending]',
      'flowchart LR\n A[Primary video] --> B[Pose-derived joint angles]\n B --> C[Original five-fold GRU ensemble]\n C --> D[Mean class probabilities]\n A --> E[Normalized hip movement]\n A --> F[Paddle image-plane reference state]\n D --> G[0.60 gate and deterministic rules]\n E --> G\n F --> G\n G --> H[Subtype plus limited feedback or notice]\n I[Auxiliary landing disabled]',
      'flowchart TD\n A[Preserved pose cache and clip ID] --> B[Original GRU ensemble]\n B --> C[Subtype and confidence]\n D[Unchanged hip and paddle measurements] --> E[Gate and deterministic feedback]\n C --> E\n E --> F[New frozen reports]\n F --> G[New Session 2 form]\n H[Android original-GRU default; phone validation pending]',
      'flowchart TD\n A[Pose-derived angles] --> B[First 128-frame GRU window and moment matching]\n B --> C[Average five fold probabilities]\n C --> D{Confidence at least 0.60?}\n D -->|No| E[Uncertainty notice; retain measurements]\n D -->|Yes| F[Evaluate supported rules]\n G[Hip and paddle reference comparisons] --> F\n F --> H[Corrective, positive or partial output]',
      'flowchart BT\n A[Pose and detector geometry] --> B[Original GRU classification and separate measurements]\n B --> C[Confidence gate and rule-based feedback]\n C --> D[Recorded-clip prototype and Android default]\n D --> E[Expert review and pending phone validation]'
    ]
    for old,new in zip(diagrams,graphs): body=body.replace(old,'```mermaid\n'+new+'\n```')
    amendment='''The selected Session 2 configuration was amended after Session 1 receipt but before outputs were sent to Coach C. Researchers had already inspected Session 1 judgments. Original GRU was selected using the existing matched diagnostic comparison, not by fitting model weights, thresholds or cue text to those judgments. All 30 clips remain; classifications were recomputed from preserved poses and hip/paddle/contact measurements were reused unchanged. The earlier hybrid outputs remain archived. This chronology is disclosed rather than described as a model frozen before Session 1. Android and desktop now default to original GRU, with phone capture and latency still requiring validation.'''
    anchor='## **4.3 Research Procedure** {#4.3-research-procedure}'
    body=body.replace(anchor,anchor+'\n\n'+amendment)
    # Keep the isolated chapter deliverable synchronized.
    chapters=body[body.index('# **CHAPTER 5** {#chapter-5}'):]
    ending='\nThe appendix letters retain their historical wording'
    chapters=chapters.split(ending,1)[0].rstrip()+'\n'
    (D/'CHAPTERS_5_6.md').write_text(chapters,encoding='utf-8')
    p.write_text(body+'# **BIBLIOGRAPHY** {#bibliography}'+tail,encoding='utf-8')
    m=D/'revision_manifest.json'; meta=json.loads(m.read_text(encoding='utf-8'))
    meta.update(working_copy_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                selected_model='original_gru_ensemble',pre_session2_model_amendment=amendment,
                model_and_feedback_changed=True)
    m.write_text(json.dumps(meta,indent=2),encoding='utf-8')


if __name__=='__main__': main()
