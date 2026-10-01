"""Freeze the thesis's random 30-clip Coach C sample and blinded Session 1 packet.

Selection uses recording categories, never predictions. Existing development
exposure is retained and disclosed; this cannot retroactively create a sealed test.
No model is run and no coach response is overwritten.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from collections import Counter
from zipfile import ZipFile, ZIP_STORED

import cv2
import numpy as np
from serve_study import ROOT, CLASSES, sha, write_json
from coach_annotations import WORKBOOK

DEST = ROOT/'outputs/coach_c_evaluation_v1'
SEED = 20261001
QUOTAS = {'CoachA':2, 'Beginner3':4, 'Beginner4':4}
PAPER = ROOT/'docs/Pascua-Leones_Thesis.md'


def csv_write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def manifest_rows(path, subject):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return [dict(clip_id=Path(r['clip_path'].replace('\\','/')).stem,
                     video=r['clip_path'].replace('\\','/'),
                     recording_category=r['serve_type'].lower(),subject=subject)
                for r in csv.DictReader(handle)]


def legacy_ids():
    folder=ROOT/'data/training/keypoints_angles'
    with (folder/'manifest.csv').open(encoding='utf-8-sig',newline='') as handle:
        ids={Path(r['clip_path'].replace('\\','/')).stem for r in csv.DictReader(handle)}
    return {cid for cid in ids if (folder/f'{cid}.npy').exists()}


def previous_prediction_ids():
    ids=set()
    for path in [ROOT/'outputs/pose_audit/cached_predictions.csv',ROOT/'outputs/pose_audit/fresh_predictions.csv',
                 ROOT/'outputs/hybrid_audit/predictions.csv']:
        if path.exists():
            with path.open(encoding='utf-8-sig',newline='') as handle:
                ids.update(r['clip_id'] for r in csv.DictReader(handle))
    return ids


def pool():
    current=json.loads((ROOT/'data/serve_v2/dataset.json').read_text())
    coach_b={r['clip_id'] for r in current['clips'] if r['split']=='train'}
    training_hashes={r['video_sha256'] for r in current['clips'] if r['split']=='train'}
    old_gru=legacy_ids()
    old_knn=set(map(str,np.load(ROOT/'features/timing_train.npz',allow_pickle=True)['clip_ids']))
    diagnostics={r['clip_id'] for r in current['clips'] if r['split']=='diagnostic_test'}
    inspected=previous_prediction_ids() | diagnostics
    rows=[]
    for category in CLASSES:
        for p in sorted((ROOT/'data/training/raw/coach'/category).glob('*.mp4')):
            if p.stem not in coach_b:
                rows.append(dict(clip_id=p.stem,video=p.relative_to(ROOT).as_posix(),recording_category=category,subject='CoachA'))
    rows+=manifest_rows(ROOT/'data/testing/manifest.csv','Beginner3')
    rows+=manifest_rows(ROOT/'data/testing/manifest_beg4.csv','Beginner4')
    seen=set();seen_hash=set();eligible=[];excluded=[]
    for row in sorted(rows,key=lambda r:(r['subject'],r['recording_category'],r['clip_id'])):
        path=ROOT/row['video']
        if not path.is_file():raise FileNotFoundError(path)
        row['video_sha256']=sha(path)
        if row['clip_id'] in coach_b or row['video_sha256'] in training_hashes:
            excluded.append(dict(**row,reason='Coach B training ID or identical video bytes'));continue
        if row['clip_id'] in seen or row['video_sha256'] in seen_hash:
            excluded.append(dict(**row,reason='duplicate ID or identical video bytes'));continue
        seen.add(row['clip_id']);seen_hash.add(row['video_sha256'])
        row.update(coach_b_training_overlap=False,new_gru_training_overlap=False,
                   legacy_gru_training_overlap=row['clip_id'] in old_gru,
                   legacy_knn_training_overlap=row['clip_id'] in old_knn,
                   prior_diagnostic_overlap=row['clip_id'] in diagnostics,
                   prior_model_prediction_exists=row['clip_id'] in inspected)
        eligible.append(row)
    return eligible,excluded


def draw(rows):
    """Uniform sampling without replacement within each paper participant/type stratum."""
    rng=np.random.default_rng(SEED);selected=[]
    for category in CLASSES:
        for subject,n in QUOTAS.items():
            cell=[r for r in rows if r['recording_category']==category and r['subject']==subject]
            if len(cell)<n:raise ValueError(f'Not enough {subject}/{category}: {len(cell)} < {n}')
            for index in rng.choice(len(cell),size=n,replace=False):selected.append(dict(cell[int(index)]))
    rng.shuffle(selected)
    for i,row in enumerate(selected,1):row['review_id']=f'C{i:03d}'
    assert len(selected)==30 and len({r['clip_id'] for r in selected})==30
    assert Counter(r['recording_category'] for r in selected)==dict.fromkeys(CLASSES,10)
    for category in CLASSES:
        assert Counter(r['subject'] for r in selected if r['recording_category']==category)==QUOTAS
    return selected


def freeze():
    DEST.mkdir(parents=True,exist_ok=True)
    rows,excluded=pool();selection=draw(rows)
    protocol=dict(seed=SEED,rng='numpy.default_rng PCG64',numpy_version=np.__version__,
        method='stratified random without replacement, then shuffle all 30 review IDs',
        quotas_per_recording_category=QUOTAS,recording_categories=CLASSES,
        labels='recording categories are sampling strata, not independently observed Coach C labels',
        coach_b_annotations_sha256=sha(WORKBOOK),paper_sha256=sha(PAPER),
        source_dataset_sha256=sha(ROOT/'data/serve_v2/dataset.json'),
        legacy_timing_bank_sha256=sha(ROOT/'features/timing_train.npz'),
        legacy_gru_manifest_sha256=sha(ROOT/'data/training/keypoints_angles/manifest.csv'),
        prior_predictions_hashes={str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'outputs/pose_audit/cached_predictions.csv',ROOT/'outputs/pose_audit/fresh_predictions.csv',ROOT/'outputs/hybrid_audit/predictions.csv'] if p.exists()},
        protocol_status='retrospective independent coach review; not an untouched sealed evaluation',
        permitted_use='new GRU training-disjoint, but development exposure must be disclosed; legacy-overlap cases cannot count as legacy held-out accuracy')
    payload=dict(version='coach_c_evaluation_v1',protocol=protocol,eligible_pool=rows,
                 excluded_duplicate_or_training_sources=excluded,selection=selection)
    target=DEST/'selection.json'
    if target.exists():
        previous=json.loads(target.read_text())
        if previous != payload:raise ValueError('Frozen Coach C selection changed: do not redraw or overwrite it')
        return previous
    write_json(target,payload)
    return payload


def technical_check(path):
    cap=cv2.VideoCapture(str(path))
    try:
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or count<2 or fps<=0:raise ValueError(f'Unreadable selected clip: {path}')
        # Frame-index seeking can fail on otherwise valid MP4s. Decode in order
        # to verify every frame without rejecting a clip because seeking failed.
        decoded_count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            decoded_count+=1
        if decoded_count<2:raise ValueError(f'Cannot decode selected clip: {path}')
        if abs(decoded_count-count)>max(2,int(count*.01)):
            raise ValueError(f'Unexpected decoded/advertised frame counts: {path} ({decoded_count}/{count})')
        return dict(frame_count=count,fps=fps,width=width,height=height,duration_seconds=count/fps,
                    decoded_frame_count=decoded_count,sequential_decode_complete=True,full_human_quality_review=False)
    finally:cap.release()


def main():
    frozen=freeze();packet=DEST/'session1';videos=packet/'videos';videos.mkdir(parents=True,exist_ok=True)
    key=[];responses=[];checks=[]
    for row in frozen['selection']:
        source=ROOT/row['video'];assert sha(source)==row['video_sha256']
        metadata=technical_check(source)
        dest=videos/(row['review_id']+'.mp4')
        if not dest.exists():shutil.copyfile(source,dest)
        assert sha(dest)==row['video_sha256'],'Copied review video differs from frozen source'
        # Check that the container does not advertise an original serve label/name.
        tags=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format_tags:stream_tags','-of','json',str(dest)],text=True))
        sensitive=[]
        for obj in [tags.get('format',{})]+tags.get('streams',[]):
            for k,v in obj.get('tags',{}).items():
                if any(word in str(v).lower() for word in ['drive','lob','topspin','coacha','beginner']):sensitive.append((k,v))
        if sensitive:raise ValueError(f'Original label/identity in video metadata for {row["review_id"]}: strip metadata before sharing')
        key.append(dict(review_id=row['review_id'],clip_id=row['clip_id'],source_video=row['video'],
                        recording_category=row['recording_category'],subject=row['subject'],source_sha256=row['video_sha256'],
                        **{k:row[k] for k in ['coach_b_training_overlap','new_gru_training_overlap','legacy_gru_training_overlap','legacy_knn_training_overlap','prior_diagnostic_overlap','prior_model_prediction_exists']}))
        checks.append(dict(review_id=row['review_id'],**metadata,video_metadata_labels_absent=True))
        responses.append(dict(review_id=row['review_id'],serve_type='',form_judgment='',notes='',reviewer='',review_date=''))
        print('PACKAGED',row['review_id'],flush=True)
    csv_write(DEST/'researcher_key.csv',key);csv_write(DEST/'technical_checks.csv',checks)
    response=packet/'annotations.csv'
    if not response.exists():csv_write(response,responses)
    # Researchers retain labels, identities, model comparisons and sampling quotas.
    instructions='''# Independent video annotation — Session 1

Open each video in `videos/` and complete its matching row in `annotations.csv`.
Judge the serve that is visibly executed, using your independent coaching judgment.

- serve_type: drive, lob, or topspin. If the video does not support a judgment,
  enter unjudgeable and explain in notes.
- form_judgment: good or bad. If not judgeable, enter unjudgeable and explain.
- notes: optional observations or reasons that the video is not judgeable.
- reviewer and review_date: your name and the date of review.

Use the video identifier (C001–C030) to match rows. Leave unfinished rows blank.
Return the completed annotation file to the researchers before the later feedback
review session. This packet contains no system predictions or feedback.
'''
    (packet/'README.md').write_text(instructions,encoding='utf-8')
    zip_path=DEST/'coach_c_session1.zip'
    with ZipFile(zip_path,'w',ZIP_STORED) as z:
        for path in sorted(packet.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(packet).as_posix())
    report=dict(selected_clips=30,recording_category_counts=dict(Counter(r['recording_category'] for r in key)),
        participant_counts=dict(Counter(r['subject'] for r in key)),eligible_pool_size=len(frozen['eligible_pool']),
        legacy_gru_training_overlap=sum(r['legacy_gru_training_overlap'] for r in key),
        legacy_knn_training_overlap=sum(r['legacy_knn_training_overlap'] for r in key),
        new_gru_training_overlap=sum(r['new_gru_training_overlap'] for r in key),
        coach_b_annotation_overlap=sum(r['coach_b_training_overlap'] for r in key),
        prior_diagnostic_overlap=sum(r['prior_diagnostic_overlap'] for r in key),
        prior_model_prediction_exists=sum(r['prior_model_prediction_exists'] for r in key),
        source_sha256_duplicates=False,seed=SEED,selection_sha256=sha(DEST/'selection.json'),
        session1_zip_sha256=sha(zip_path),coach_c_annotations_pending=True,session2_feedback_generated=False,
        untouched_final_test=False,limitations='sampled after development; legacy training and prior diagnostic exposures disclosed; visual detector training overlap not established')
    write_json(DEST/'status.json',report)
    text=f'''# Coach C random 30-clip sample

Selected with a frozen seed ({SEED}) and no model-confidence or correctness filtering.
There are 10 clips per recording category (Drive, Lob, Topspin), with 2 Coach A,
4 Beginner 3 and 4 Beginner 4 clips per category, matching thesis sections 4.1
and 4.2.1(C). Coach C's independent labels may change the final class balance.

Share only `coach_c_session1.zip` for Session 1. It contains anonymized C001–C030
videos, a blank annotation CSV and instructions. Keep `researcher_key.csv`,
`selection.json`, `technical_checks.csv` and this report with the researchers.
No model inference or feedback was generated in this sampling operation.

## Exposure and thesis limitation

- Coach B training/annotation overlap: {report['coach_b_annotation_overlap']}/30.
- New GRU training overlap: {report['new_gru_training_overlap']}/30.
- Legacy GRU/kNN training overlap: {report['legacy_gru_training_overlap']}/{report['legacy_knn_training_overlap']} of 30.
- Previously used diagnostic clips: {report['prior_diagnostic_overlap']}/30.
- Clips with saved prior model predictions: {report['prior_model_prediction_exists']}/30.

This is a retrospective independent coach review, not a test set sealed before
development. Coach A clips in legacy training cannot count as held-out performance
for those legacy models. Existing Beginner 3/4 development exposure must be
disclosed even though none is a new-model training example. These limitations
cannot be removed by assigning new review IDs. Visual detector training exposure
has not been established by this classifier-data audit.

Session 1 independently records observed serve type and form. Collect those
annotations before running the frozen primary pipeline on the same 30 clips.
Session 2 contains only each video and its system feedback string, with no
Session 1 labels, original categories, internal predictions or rule definitions.
Use a separate researcher comparison file for additional model predictions.

## Integrity and reproduction

All selected videos are byte-identical copies of their frozen source files.
Endpoints decode, metadata was checked for original labels/identities, and no
duplicates or Coach B training-video hashes were selected. Technical checks do
not establish correct serve execution, good form, or pose tracking quality.

`python scripts/prepare_coach_c_evaluation.py` reproduces the same selection and
refuses changed inventory/provenance. It preserves an existing annotation CSV.
'''
    (DEST/'RESEARCHER_README.md').write_text(text,encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
