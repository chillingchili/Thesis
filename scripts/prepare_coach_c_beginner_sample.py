"""Preserve the random 24 beginner selections; user supplies six unused Coach A clips."""
import json
import shutil
import subprocess
from collections import Counter
from zipfile import ZipFile,ZIP_STORED

from prepare_coach_c_evaluation import ROOT,SEED,csv_write,technical_check
from serve_study import CLASSES,sha,write_json

DEST=ROOT/'outputs/coach_c_evaluation_v2'
INITIAL=ROOT/'outputs/coach_c_evaluation_v1/selection.json'


def main():
    # Keep the first random draw, rather than shopping for another beginner sample.
    initial=json.loads(INITIAL.read_text())
    selected=[r for r in initial['selection'] if r['subject'] in ['Beginner3','Beginner4']]
    slots=[dict(review_id=r['review_id'],subject='CoachA',recording_category=r['recording_category'],
                video=None,video_sha256=None,status='user supplies an additional unused clip')
           for r in initial['selection'] if r['subject']=='CoachA']
    assert len(selected)==24 and len(slots)==6
    assert Counter(r['subject'] for r in selected)=={'Beginner3':12,'Beginner4':12}
    assert Counter(r['recording_category'] for r in selected)==dict.fromkeys(CLASSES,8)
    assert Counter(r['recording_category'] for r in slots)==dict.fromkeys(CLASSES,2)
    for subject in ['Beginner3','Beginner4']:
        assert Counter(r['recording_category'] for r in selected if r['subject']==subject)==dict.fromkeys(CLASSES,4)
    ids={r['review_id'] for r in selected+slots};assert ids=={f'C{i:03d}' for i in range(1,31)}
    frozen=dict(version='coach_c_evaluation_v2',seed=SEED,source_draw_sha256=sha(INITIAL),
        method='preserve the 24 beginner clips from the first random stratified draw; replace six local-coach selections with user-supplied slots',
        original_coach_selection_withdrawn=True,user_instruction='Beginner 3/4 random; user supplies Coach A clips outside root',
        beginner_selection=selected,pending_coach_slots=slots,
        beginner_class_labels='recording-category sampling strata, not Coach C ground truth',
        untouched_final_test=False,prior_beginner_diagnostic_exposure=True,
        model_selection_sha256=sha(ROOT/'models/serve_refinement_v4/frozen_selection.json'))
    DEST.mkdir(parents=True,exist_ok=True);manifest=DEST/'selection.json'
    if manifest.exists():assert json.loads(manifest.read_text())==frozen,'Frozen beginner selection changed'
    else:write_json(manifest,frozen)
    packet=DEST/'session1_staging';videos=packet/'videos';videos.mkdir(parents=True,exist_ok=True)
    key=[];checks=[]
    for row in selected:
        source=ROOT/row['video'];assert sha(source)==row['video_sha256']
        metadata=technical_check(source)
        dest=videos/(row['review_id']+'.mp4')
        if not dest.exists():shutil.copyfile(source,dest)
        assert sha(dest)==row['video_sha256']
        tags=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format_tags:stream_tags','-of','json',str(dest)],text=True))
        for obj in [tags.get('format',{})]+tags.get('streams',[]):
            for value in obj.get('tags',{}).values():
                if any(word in str(value).lower() for word in ['drive','lob','topspin','coacha','beginner']):
                    raise ValueError('Original identity/category in video metadata: '+row['review_id'])
        checks.append(dict(review_id=row['review_id'],**metadata,metadata_identity_or_category_absent=True))
        key.append(dict(review_id=row['review_id'],clip_id=row['clip_id'],participant=row['subject'],
            recording_category=row['recording_category'],source_video=row['video'],source_sha256=row['video_sha256'],
            new_gru_training_overlap=row['new_gru_training_overlap'],legacy_gru_training_overlap=row['legacy_gru_training_overlap'],
            legacy_knn_training_overlap=row['legacy_knn_training_overlap'],prior_diagnostic_overlap=row['prior_diagnostic_overlap'],
            prior_model_prediction_exists=row['prior_model_prediction_exists']))
        print('PACKAGED',row['review_id'],flush=True)
    csv_write(DEST/'researcher_key.csv',key);csv_write(DEST/'technical_checks.csv',checks)
    intake=DEST/'coach_clip_slots.csv'
    if not intake.exists():csv_write(intake,[dict(review_id=r['review_id'],participant='CoachA',
        recording_category=r['recording_category'],source_video='',source_sha256='',
        renamed_packet_filename=r['review_id']+'.mp4',training_overlap_checked='',notes='') for r in slots])
    responses=packet/'annotations.csv'
    if not responses.exists():csv_write(responses,[dict(review_id=f'C{i:03d}',serve_type='',form_judgment='',notes='',reviewer='',review_date='') for i in range(1,31)])
    (packet/'README.md').write_text('''# Independent video annotation — Session 1

The complete review set contains videos C001–C030. If any video is missing,
ask the researchers to complete the set before starting the annotation session.
Open each video in `videos/` and complete its matching row in `annotations.csv`.
Judge the visibly executed serve using your independent coaching judgment.

- serve_type: drive, lob, or topspin; use unjudgeable if the video is insufficient.
- form_judgment: good or bad; use unjudgeable if the video is insufficient.
- notes: optional observations or reasons for an unjudgeable result.
- reviewer and review_date: your name and review date.

Leave unfinished rows blank. Return completed annotations to the researchers
before the later feedback review. No original labels or system outputs are supplied.
''',encoding='utf-8')
    zip_path=DEST/'beginner_24_staging.zip'
    with ZipFile(zip_path,'w',ZIP_STORED) as z:
        for path in sorted(packet.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(packet).as_posix())
    report=dict(selected_beginner_clips=24,pending_coach_clips=6,complete_30_video_packet=False,
        beginner_participant_counts=dict(Counter(r['participant'] for r in key)),
        beginner_recording_category_counts=dict(Counter(r['recording_category'] for r in key)),
        new_gru_training_overlap=sum(r['new_gru_training_overlap'] for r in key),
        legacy_gru_training_overlap=sum(r['legacy_gru_training_overlap'] for r in key),
        legacy_knn_training_overlap=sum(r['legacy_knn_training_overlap'] for r in key),
        prior_diagnostic_overlap=sum(r['prior_diagnostic_overlap'] for r in key),
        seed=SEED,selection_sha256=sha(manifest),packet_sha256=sha(zip_path),
        blind_packet_verified=False,model_outputs_generated=False,coach_c_annotations_pending=True,
        final_claim='retrospective independent coach review; beginner clips were previously used in diagnostic development')
    # Verify copied videos, count, blind names and absence of the researcher key.
    with ZipFile(zip_path) as z:
        names=set(z.namelist());video_names={n for n in names if n.endswith('.mp4')}
        assert video_names=={'videos/'+r['review_id']+'.mp4' for r in selected}
        assert len(video_names)==24 and len(names)==26
        assert 'researcher_key.csv' not in names and 'coach_clip_slots.csv' not in names
        assert z.testzip() is None
    report['blind_packet_verified']=True;write_json(DEST/'status.json',report)
    (DEST/'RESEARCHER_README.md').write_text(f'''# Random beginner sample for Coach C

The requested beginner selection is complete: 12 Beginner 3 clips and 12
Beginner 4 clips, with 4 per recording category from each participant. This
preserves the first random draw (seed {SEED}); no model output or coach result
was used to select or replace a beginner clip.

The six local Coach A selections in the initial, incomplete v1 preparation
are withdrawn. Use this v2 folder only. You will supply two additional unused
Coach A recordings per category. `coach_clip_slots.csv` lists their anonymous
filenames and sampling categories; keep this file with the researchers.

`beginner_24_staging.zip` contains 24 anonymized videos, a blank 30-row annotation
sheet and Session 1 instructions. It is a staging packet, not a complete 30-video
Coach C delivery. Add your six Coach A videos under the indicated Cxxx filenames
before sharing the completed set. Original clip identity and recording category
must not be shared with Coach C. Clip hashes/training overlap must be checked
before the final set is frozen. Coach C may classify the executed serves
differently, so 10 per intended recording category does not guarantee 10 per
independently observed subtype.

All 24 beginner clips are absent from both the legacy and current GRU/kNN
training sets. However, all 24 were used in earlier diagnostic experiments.
This is independent expert review after development, not an untouched final
test sealed before experimentation. These limits remain even when your six
unused Coach A clips are added. Do not tune models or thresholds on Coach C's
final judgments if reporting this as the frozen pipeline's evaluation.

Complete Session 1 independent subtype/form annotation before running the
frozen primary pipeline on these videos. Session 2 then contains each video
and its feedback string, without Session 1 annotations or internal outputs.
No model inference or feedback was generated during this sampling operation.

Sequential decoding verified every selected clip without relying on unreliable
seeking to its last frame. Source files are unchanged; copied video bytes match
the frozen SHA-256. This establishes file integrity, not serve or form correctness.
''',encoding='utf-8')
    (ROOT/'outputs/coach_c_evaluation_v1/SUPERSEDED.md').write_text('''Superseded by ../coach_c_evaluation_v2 after the user elected to supply six
additional Coach A clips. This was an incomplete initial preparation; do not
use its local-coach selection or partial Session 1 files for Coach C delivery.
The 24 beginner selections were preserved in v2 without redrawing.
''',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
