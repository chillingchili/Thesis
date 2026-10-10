"""Private, frozen 30-clip hybrid inference and Session 2 packet preparation.

Run only after the researcher authorizes the timing/model configuration.
Does not read Coach C responses, modify Session 1, or release any packet.
Resume an interrupted run with --run-dir pointing to its existing directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import threading
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVAL = ROOT / 'outputs/coach_c_evaluation_v2'
OPTIONS = dict(start=0, end=None, model='hybrid', landing='off', moment_matching=True)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze():
    with (EVAL / 'researcher_key.csv').open(encoding='utf-8-sig', newline='') as f:
        key = list(csv.DictReader(f))
    expected = [f'C{i:03d}' for i in range(1, 31)]
    assert [r['review_id'] for r in key] == expected, 'Expected frozen C001–C030'
    clips = {}
    for r in key:
        p = EVAL / 'session1_staging/videos' / (r['review_id'] + '.mp4')
        assert sha(p) == r['source_sha256'], f'Clip changed: {p.name}'
        clips[r['review_id']] = sha(p)
    assets = ROOT / 'android/app/src/main/assets'
    names = ['pose_landmarker_lite.task', 'manifest.json', 'gru_single.tflite',
             'knn_meta.json', 'knn_train.bin', 'shift_config.json',
             'paddle_config.json', 'feedback_rules.json']
    names += [f'gru_fold{i}.tflite' for i in range(1, 6)]
    paths = [assets / n for n in names]
    paths += [ROOT / 'runs/detect/paddle_ft/weights/best.pt']
    paths += sorted((ROOT / 'desktop').glob('*.py'))
    paths += sorted((ROOT / 'scripts').glob('*.py'))
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    protected = [EVAL / n for n in ['selection.json', 'researcher_key.csv',
                 'technical_checks.csv', 'coach_c_session1.zip']]
    protected += [EVAL / 'session1_staging/annotations.csv']
    versions = {}
    for name in ['numpy', 'tensorflow', 'mediapipe', 'ultralytics', 'opencv-python']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    return dict(options=OPTIONS, clip_sha256=clips, frozen_files=hashes,
                session1_protected={p.relative_to(ROOT).as_posix(): sha(p) for p in protected},
                package_versions=versions)


def assert_unchanged(record):
    for group in ['frozen_files', 'session1_protected']:
        for name, digest in record[group].items():
            assert sha(ROOT / name) == digest, f'Frozen/protected file changed: {name}'
    for rid, digest in record['clip_sha256'].items():
        assert sha(EVAL / 'session1_staging/videos' / (rid + '.mp4')) == digest


def build_packet(run, reports):
    packet = run / 'session2_packet_HOLD'
    (packet / 'videos').mkdir(parents=True, exist_ok=True)
    responses, cards, summary_rows = [], [], []
    for rid, report in reports.items():
        feedback = report['feedback']
        message = '\n'.join(feedback['messages'] + feedback['notifications'])
        assert message.strip(), f'No system feedback for {rid}'
        shutil.copy2(EVAL / 'session1_staging/videos' / (rid + '.mp4'), packet / 'videos' / (rid + '.mp4'))
        responses.append(dict(review_id=rid, feedback_string=message, appropriateness='',
                              notes='', reviewer='', review_date=''))
        summary_rows.append(dict(review_id=rid, serve_status=report['serve']['status'],
            predicted_serve=report['serve'].get('label', ''), confidence=report['serve'].get('confidence', ''),
            shift_status=report['shift']['status'], paddle_status=report['paddle']['status'],
            feedback_status=feedback['status'], feedback_string=message, seconds=report['seconds']))
        cards.append(f'''<section data-id="{rid}"><h2>{rid}</h2>
<video controls preload="none" src="videos/{rid}.mp4"></video>
<h3>System feedback</h3><p class="feedback">{html.escape(message)}</p>
<fieldset><legend>Is this feedback appropriate for the mechanics visible in this clip?</legend>
<label><input type="radio" name="{rid}" value="appropriate"> Appropriate</label>
<label><input type="radio" name="{rid}" value="inappropriate"> Inappropriate</label>
<label><input type="radio" name="{rid}" value="unjudgeable"> Unable to judge</label></fieldset>
<label>Notes (optional)<textarea aria-label="Notes for {rid}"></textarea></label></section>''')
    write_csv(packet / 'feedback_review.csv', responses)
    write_csv(run / 'researcher_outputs.csv', summary_rows)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coach C — Session 2</title><style>
body{font:17px/1.6 system-ui,sans-serif;color:#202020;max-width:860px;margin:40px auto;padding:0 22px}
section{border-top:1px solid #bbb;padding:28px 0}video{width:100%;max-height:500px;background:#111}
.feedback{white-space:pre-wrap}fieldset{border:1px solid #ccc}label{display:block;margin:10px 0}
textarea{display:block;width:95%;min-height:75px}input,button,textarea{font:inherit}button{padding:10px 18px}
</style><h1>Session 2: feedback review</h1>
<p>Watch each video and judge the system feedback shown. Evaluate the message against the mechanics you observe.
Do not consult Session 1 answers. A withheld-feedback or missing-measurement message is also a system output to review.
Use “Unable to judge” if the clip or message cannot be assessed; explain in the notes.</p>
<p>Answers stay on this page until you download them. Download before closing the browser; unfinished rows remain blank.</p>
<label>Reviewer <input id="reviewer"></label><label>Review date <input id="date" type="date"></label>
<button onclick="save()">Download responses CSV</button><p id="count" aria-live="polite"></p>
''' + '\n'.join(cards) + '''<button onclick="save()">Download responses CSV</button>
<script>
const q=v=>'"'+String(v).replaceAll('"','""')+'"';
function save(){const rows=[['review_id','appropriateness','notes','reviewer','review_date']];let done=0;
for(const s of document.querySelectorAll('section')){const a=s.querySelector('input:checked');if(a)done++;
rows.push([s.dataset.id,a?a.value:'',s.querySelector('textarea').value,document.getElementById('reviewer').value,document.getElementById('date').value]);}
const blob=new Blob(['\\ufeff'+rows.map(r=>r.map(q).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='session2_responses.csv';a.click();
setTimeout(()=>URL.revokeObjectURL(a.href),1000);document.getElementById('count').textContent=done+' of 30 judgments completed.';}
</script></html>'''
    (packet / 'index.html').write_text(page, encoding='utf-8')
    (packet / 'README.md').write_text('''# Coach C — Session 2

Open index.html in a browser to watch C001–C030 with each system feedback string.
Record appropriate/inappropriate, or unable to judge with a reason. Download
your responses before closing the page. Alternatively complete feedback_review.csv.
Do not consult Session 1 annotations or other model outputs. Return the completed
response file to the researchers. No original categories or predictions are supplied.
''', encoding='utf-8')
    zip_path = run / 'PRIVATE_DO_NOT_SEND_session2.zip'
    with ZipFile(zip_path, 'w', ZIP_STORED) as z:
        for p in sorted(packet.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(packet).as_posix())
    with ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == {'index.html', 'README.md', 'feedback_review.csv'} | {
            f'videos/C{i:03d}.mp4' for i in range(1, 31)}
    for rid in reports:
        assert sha(packet / 'videos' / (rid + '.mp4')) == sha(EVAL / 'session1_staging/videos' / (rid + '.mp4'))
    return summary_rows, zip_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path)
    args = ap.parse_args()
    run = args.run_dir or EVAL / ('private_session2_hybrid_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir(parents=True, exist_ok=True)
    manifest = run / 'freeze.json'
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding='utf-8'))
        assert record['options'] == OPTIONS
        assert_unchanged(record)
    else:
        record = freeze()
        record.update(created_utc=datetime.now(timezone.utc).isoformat(),
            timing_deviation='User authorized private inference during ongoing Session 1; outputs withheld until Session 1 submission.',
            session1_labels_used=False, release_status='HOLD_UNTIL_SESSION1_SUBMITTED',
            limitations=['24 beginner clips previously used in diagnostics.',
                'Recorded-clip capture window differs from Android live motion gating.',
                'No second-camera videos supplied; auxiliary landing disabled.',
                'Paddle direction remains unvalidated; frozen rules restrict directional advice.'])
        write_json(manifest, record)
    print('PRIVATE_RUN_DIR', run, flush=True)
    from desktop.pipeline import analyze
    reports = {}
    for number, rid in enumerate(record['clip_sha256'], 1):
        assert_unchanged(record)
        out = run / 'model_outputs' / rid
        out.mkdir(parents=True, exist_ok=True)
        report_file = out / 'report.json'
        if report_file.exists():
            report = json.loads(report_file.read_text(encoding='utf-8'))
            assert report['options'] == OPTIONS and report['input_sha256'] == record['clip_sha256'][rid]
        else:
            print(f'BEGIN {number}/30 {rid}', flush=True)
            report = analyze(EVAL / 'session1_staging/videos' / (rid + '.mp4'), None,
                             out, dict(OPTIONS), lambda *_: None, threading.Event())
            report.update(review_id=rid, input_sha256=record['clip_sha256'][rid],
                          freeze_sha256=sha(manifest))
            write_json(report_file, report)
        reports[rid] = report
        print(f"DONE {number}/30 {rid} serve={report['serve']['status']} paddle={report['paddle']['status']} feedback={report['feedback']['status']}", flush=True)
    assert_unchanged(record)
    rows, zip_path = build_packet(run, reports)
    write_json(run / 'summary.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        clips=30, release_status='HOLD_UNTIL_SESSION1_SUBMITTED',
        serve_status_counts=dict(Counter(r['serve_status'] for r in rows)),
        shift_status_counts=dict(Counter(r['shift_status'] for r in rows)),
        paddle_status_counts=dict(Counter(r['paddle_status'] for r in rows)),
        feedback_status_counts=dict(Counter(r['feedback_status'] for r in rows)),
        accuracy=None, coach_c_appropriateness=None, session1_preserved=True,
        packet_sha256=sha(zip_path), freeze_sha256=sha(manifest)))
    (run / 'RESEARCHER_README.md').write_text('''# PRIVATE — Session 2 prepared early

Do not share PRIVATE_DO_NOT_SEND_session2.zip or session2_packet_HOLD until
Coach C has submitted Session 1 annotations. Record that submission separately.
The user authorized early private inference as a timing change to the thesis.
The pipeline and assets are frozen in freeze.json; no coach labels were used.
Reports and researcher_outputs.csv are researcher-only. The packet contains
only anonymous videos, feedback strings, review fields and instructions.
No accuracy or feedback-appropriateness result exists before the coach replies.
Do not tune models/thresholds against these clips or select favorable outputs.
''', encoding='utf-8')
    print('COMPLETE', zip_path, flush=True)


if __name__ == '__main__':
    main()
