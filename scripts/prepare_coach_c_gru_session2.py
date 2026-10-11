"""Recompute original GRU from cached poses; preserve hybrid evidence and all 30 clips.

Authorized before Session 2 release, after Session 1 receipt. Does not read coach
responses or fit parameters. Cached hip/paddle measurements are reused unchanged.
"""
from pathlib import Path
import sys
import copy
import hashlib
import json
import csv
import html
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop.pipeline import classify, read_config
from desktop.feedback import evaluate


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import numpy as np
    base = ROOT / 'evaluation/coach_c'
    old = ROOT / 'outputs/coach_c_evaluation_v2/private_session2_hybrid_20261010T045855Z/model_outputs'
    out = base / 'session2_gru'
    out.mkdir(exist_ok=False)
    (out / 'reports').mkdir()
    evidence = json.loads((base / 'provenance.json').read_text(encoding='utf-8'))
    assert all(sha(base / x['path']) == x['sha256'] for x in evidence['files'])
    rows, cards, hashes = [], [], {}
    rules = read_config('feedback_rules.json')
    for i in range(1,31):
        rid = f'C{i:03d}'
        source = base / 'baseline/reports' / f'{rid}.json'
        poses = old / rid / 'keypoints.npy'
        original = json.loads(source.read_text(encoding='utf-8'))
        assert sha(base / 'videos' / f'{rid}.mp4') == original['input_sha256']
        report = copy.deepcopy(original)
        report['serve'] = classify(np.load(poses, allow_pickle=False), 'ensemble', True)
        report['options']['model'] = 'ensemble'
        report['feedback'] = evaluate(rules, report['serve']['label'], report['serve']['confidence'],
                                      report['shift'].get('sufficient'), report['paddle'].get('state'))
        report['seconds'] = None
        report['files'] = []
        report['assets']['feedback_rules.json'] = sha(ROOT / 'android/app/src/main/assets/feedback_rules.json')
        report['recomputation'] = dict(method='Original GRU inference on preserved pose cache; hip/paddle/contact reused',
            keypoints_sha256=sha(poses), source_report_sha256=sha(source),
            original_hybrid_seconds=original['seconds'], end_to_end_latency_measured=False)
        assert all(report[k] == original[k] for k in ['shift','paddle','contact','landing','input_sha256'])
        (out / 'reports' / f'{rid}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        message = '\n'.join(report['feedback']['messages'] + report['feedback']['notifications'])
        label = report['serve']['label'].title()
        uncertain = ' (low confidence)' if report['serve']['confidence'] < .60 else ''
        display = f'Serve classification: {label}{uncertain} | Feedback: {message}'
        rows.append(dict(review_id=rid, predicted_serve=label, confidence=report['serve']['confidence'],
                         feedback_status=report['feedback']['status'], feedback_string=message,
                         hybrid_status=original['feedback']['status'], hybrid_prediction=original['serve']['label']))
        cards.append(f'<section><h2>{rid}</h2><video controls preload="none" src="../videos/{rid}.mp4"></video><p>{html.escape(display)}</p></section>')
        hashes[rid] = dict(video=original['input_sha256'], poses=sha(poses), source_report=sha(source),
                           report=sha(out / 'reports' / f'{rid}.json'))
        print(rid, report['serve']['label'], round(report['serve']['confidence'],4), report['feedback']['status'],flush=True)
    counts = dict(Counter(r['feedback_status'] for r in rows))
    substantive = counts.get('CORRECTIVE',0)+counts.get('POSITIVE',0)
    summary = dict(model='original GRU five-fold ensemble', confidence_threshold=.60, clips=30,
                   feedback_status_counts=counts, substantive_coaching_count=substantive,
                   substantive_coaching_coverage_pct=100*substantive/30,
                   session2_sent=False, session2_ratings_received=False,
                   rationale='User-selected before Session 2 release based on prior matched diagnostic comparison: original GRU 51.4%, hybrid 49.7%. No claim of statistically established superiority.',
                   timing='Switch made after Session 1 receipt; earlier judgments already known to researchers. This generator does not read or fit to them.',
                   measurement_reuse='Same frozen hip/paddle/contact values; same cues and 0.60 threshold; classifier changed only.',
                   independent_classification_accuracy=None)
    (out / 'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out / 'feedback_strings.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coach C Session 2 — original GRU</title><style>body{max-width:850px;margin:40px auto;padding:0 20px;font:17px/1.6 system-ui}section{border-top:1px solid #ccc;padding:24px 0}video{width:100%;max-height:500px}p{white-space:pre-wrap}</style><h1>Session 2: system feedback</h1><p>Review each clip and its feedback using the evaluation form. Do not consult Session 1 answers.</p>'+''.join(cards)+'</html>'
    (out / 'index.html').write_text(page,encoding='utf-8')
    assetdir=ROOT/'android/app/src/main/assets'
    tracked=[assetdir/'manifest.json',assetdir/'feedback_rules.json',assetdir/'shift_config.json',assetdir/'paddle_config.json']
    tracked += [assetdir/f'gru_fold{i}.tflite' for i in range(1,6)]
    tracked += [Path(__file__),ROOT/'desktop/pipeline.py',ROOT/'desktop/feedback.py',ROOT/'android/app/src/main/java/com/thesis/pickleballserve/MainActivity.kt',ROOT/'android/app/src/main/java/com/thesis/pickleballserve/FeedbackEngine.kt']
    (out/'freeze.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
       model='ensemble', wording='coach-c-actionable-wording-v2', threshold=.60, clips=hashes,
       files={p.relative_to(ROOT).as_posix():sha(p) for p in tracked},
       coach_responses_read_by_generator=False, researcher_prior_session1_exposure=True),indent=2),encoding='utf-8')
    assert all(sha(base/x['path'])==x['sha256'] for x in evidence['files'])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
