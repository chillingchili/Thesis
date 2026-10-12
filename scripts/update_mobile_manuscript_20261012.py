from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
def read(p): return (R/p).read_text(encoding='utf-8')
def write(p,s): (R/p).write_text(s,encoding='utf-8')
def main():
 cpath='docs/thesis/CHAPTERS_5_6.md';old=read(cpath);c=old
 assert 'Post-review mobile replay' not in c, 'Already applied; edit the current revision.'
 section='''### Post-review mobile replay and automatic feedback refinement

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

'''
 c=c.replace('**Table 14. Objective attainment and evidence limits**',section+'**Table 14. Objective attainment and evidence limits**')
 c=c.replace('| Substantive feedback availability | 21/30 clips (70.0%); availability is distinct from validity |','| Substantive feedback availability | Frozen expert-review outputs: 21/30 (70.0%); post-review phone outputs: 18/30 (60.0%), not independently rated |')
 c=c.replace('| Live feedback within two seconds after capture | Not verified by the recorded-file benchmark; no claim of target attainment |','| Live feedback within two seconds after capture | Not established: replay UI updates were fast after detection, but one occurred 3.126 s after file end and three clips had no completed output |')
 c=c.replace('## **6.3 Conclusions**', 'The post-review phone refinement increased corrective-feedback availability from 2/30 to 18/30 on reused diagnostic recordings. Eleven outputs used guarded, classification-independent mechanics rules. Three clips still produced no completed output and one result was late relative to file end. These updated outputs have not received independent expert validation and do not inherit the earlier 90.0% submitted-validity result.\n\n## **6.3 Conclusions**')
 c=c.replace('The thesis objectives were partially attained.', 'The mobile follow-up shows that acquisition and feedback policy materially affect output availability even when GRU weights remain fixed. It supports feasibility of limited automatic practice cues, while exposing unresolved capture, paddle-measurement, classification and latency failures.\n\nThe thesis objectives were partially attained.')
 c=c.replace('Second, instrument the real live capture-end-to-displayed-feedback path on the target phone.', 'Second, preserve the improved automatic configuration as a documented development version and prioritize complete capture boundaries, paddle evidence and matched phone/desktop input preparation. Instrument the real live capture-end-to-displayed-feedback path on the target phone.')
 full=read('docs/thesis/Pascua-Leones_Thesis_Current.md');assert old.strip() in full;full=full.replace(old.strip(),c.strip(),1)
 abstract='The study demonstrates a limited prototype and completed descriptive review, with partially attained objectives;'
 full=full.replace(abstract,'A post-review live-path phone replay on the same 30 clips increased corrective-feedback availability from 2/30 to 18/30 using GPU acquisition, time-normalized capture and guarded general cues. Eleven outputs bypassed uncertain subtype classification only for rules applicable to all serve types. These new outputs were not expert-rated; three clips produced no completed output and one result appeared 3.126 seconds after file end. '+abstract,1)
 full=full.replace('Changes to the initial proposal are disclosed explicitly.', 'Changes to the initial proposal are disclosed explicitly. A later mobile development branch, reported in Section 5.8, changes automatic acquisition and permits guarded general-mechanics cues despite uncertain subtype classification. It was developed after the expert review and does not alter the frozen configuration or inherit its ratings.',1)
 write(cpath,c);write('docs/thesis/Pascua-Leones_Thesis_Current.md',full)
 with (R/'docs/thesis/README.md').open('a',encoding='utf-8') as f:f.write('\n\n## Post-review phone update\n\nThe current manuscript and Chapters 5–6 now include the 30-clip live-path development comparison: corrective output coverage 2/30 CPU, 5/30 GPU, 18/30 GPU with automatic trigger/general-cue changes. Eleven final cues were classification-independent. These outputs were not rated by Coach C. Three clips had no completed output; C005 was 3.126 s after file end. No claim of consistent two-second delivery or coach replacement is made. Evidence: `evaluation/coach_c/mobile_gpu_rate_20261012/`.\n')
 print('Updated current manuscript and matching Chapters 5–6.')
if __name__=='__main__': main()
