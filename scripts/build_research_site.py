"""Export the existing research artifacts into a self-contained static showcase.

    No model fitting or inference during export. Only saved, non-blinded samples ship.
Run from anywhere: python scripts/build_research_site.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'research-site'
ASSETS = SITE / 'assets'
CLASSES = ['drive', 'lob', 'topspin']
SOURCES: dict[str, dict] = {}


def read(name):
    p = ROOT / name
    SOURCES[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    return json.loads(p.read_text(encoding='utf-8'))


def metric(d):
    result = {k: d[k] for k in ['accuracy', 'macro_f1', 'n', 'correct'] if k in d}
    result['accuracy'] = d.get('accuracy_all_clips_quality_failures_wrong', d['accuracy'])
    # v2's older confusion/F1 fields predate counting quality failures as unavailable.
    if 'accuracy_all_clips_quality_failures_wrong' in d:
        result.pop('macro_f1', None)
    cm = d.get('confusion_with_unavailable')
    if cm:
        result['confusion'] = cm[:3]
        result['recall'] = [row[i] / sum(row) if sum(row) else 0 for i, row in enumerate(cm[:3])]
        result['precision'] = [cm[i][i] / sum(row[i] for row in cm[:3]) if sum(row[i] for row in cm[:3]) else 0 for i in range(3)]
    result.update({k: d[k] for k in ['class_recall', 'log_loss_available', 'ece_available'] if k in d})
    return result


def copy_source(name, dest=None):
    p = ROOT / name
    dest = dest or p.name
    target = ASSETS / 'sources' / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, target)
    SOURCES[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'download': 'assets/sources/' + dest}
    return 'assets/sources/' + dest


def video_asset(source, stem):
    output = ASSETS / 'samples' / (stem + '.mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source),
                        '-vf', 'scale=960:-2', '-an', '-c:v', 'libx264', '-crf', '25',
                        '-preset', 'fast', '-movflags', '+faststart', str(output)], check=True)
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * .54))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'Cannot read sample {source}')
    frame = cv2.resize(frame, (1280, round(frame.shape[0] * 1280 / frame.shape[1])))
    poster = ASSETS / 'samples' / (stem + '.webp')
    cv2.imwrite(str(poster), frame, [cv2.IMWRITE_WEBP_QUALITY, 86])
    return {'video': f'assets/samples/{stem}.mp4', 'poster': f'assets/samples/{stem}.webp'}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    v2cv = read('models/serve_v2_fixed/cv_results.json')
    v2eval = read('models/serve_v2_fixed/evaluation.json')['results']['diagnostic_test']
    legacy = read('outputs/serve_study_v3/hybrid/results.json')['first128_recorded_window']
    for key, title in [('gru', 'Original GRU ensemble'), ('knn', 'kNN5 · timing features'), ('hybrid', 'Original GRU + kNN5')]:
        rows.append({'id': 'legacy_' + key, 'group': 'mobile', 'name': title, 'mode': 'all',
                     'detail': 'Bundled Android assets. First 128 valid frames, fresh Lite poses. Recorded-window evaluation; no physical phone test.',
                     'diagnostic': metric(legacy[key])})
    for key, title in [('v2_ensemble', 'Retrained GRU · five-fold ensemble'), ('v2_single', 'Retrained GRU · single fit')]:
        row = {'id': key, 'group': 'v2', 'name': title, 'mode': 'gru', 'diagnostic': metric(v2eval[key]['all']),
               'detail': '209 eligible annotated training clips. Complete-serve resampling and feature augmentation. Diagnostic quality failures count as incorrect.'}
        if key == 'v2_ensemble':
            row.update(stratified=metric(v2cv['stratified']['metric']), player=metric(v2cv['participant']['metric']))
        rows.append(row)
    pilot = read('models/serve_v2/cv_results.json')
    rows.append({'id': 'v2_pilot', 'group': 'v2', 'name': 'Retraining pilot · inner early stopping', 'mode': 'gru',
                 'stratified': metric(pilot['stratified']['metric']), 'player': metric(pilot['participant']['metric']),
                 'detail': 'Training-only pilot. Varied stopping epochs; no diagnostic score is reported here.'})
    names = {'angles_control': 'Five angles · control', 'angles_thesis_aug': 'Five angles + video augmentation',
             'skeleton': 'Full skeleton', 'skeleton_motion': 'Full skeleton + motion',
             'lean_control': 'Lean arm / torso · control', 'lean_thesis_aug': 'Lean arm / torso + augmentation',
             'skeleton_thesis_aug': 'Full skeleton + augmentation', 'skeleton_motion_thesis_aug': 'Skeleton / motion + augmentation'}
    v3 = read('models/serve_study_v3/diagnostic_evaluation.json')
    for condition, modes in v3['conditions'].items():
        cv = read(f'models/serve_study_v3/{condition}/cv_results.json')
        for mode in ['gru', 'hybrid']:
            rows.append({'id': f'v3_{condition}_{mode}', 'group': 'v3', 'name': names[condition], 'mode': mode,
                         'stratified': metric(cv['stratified'][mode]), 'player': metric(cv['participant'][mode]),
                         'diagnostic': metric(modes[mode]),
                         'detail': 'Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity.'})
    v4 = read('outputs/serve_refinement_v4/diagnostic_results.json')
    calibration = []
    for condition, modes in v4['conditions'].items():
        cv = read(f'models/serve_refinement_v4/{condition}/cv_results.json')
        for mode, cvmode in [('gru', 'gru'), ('hybrid', 'fixed_hybrid'), ('tuned', 'nested_tuned')]:
            diagmode = {'gru': 'gru', 'hybrid': 'fixed_hybrid', 'tuned': 'tuned_calibrated'}[mode]
            rows.append({'id': f'v4_{condition}_{mode}', 'group': 'v4', 'name': names[condition], 'mode': mode,
                         'player': metric(cv[cvmode]), 'diagnostic': metric(modes[diagmode]),
                         'selected': condition == v4['selected_condition'] and mode == 'tuned',
                         'detail': 'Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled.'})
        calibration.append({'name': names[condition], 'before': cv['fixed_hybrid']['log_loss_available'],
                            'after': cv['nested_tuned']['log_loss_available'],
                            'ece_before': cv['fixed_hybrid']['ece_available'], 'ece_after': cv['nested_tuned']['ece_available'],
                            'accepted_accuracy': cv['nested_selected_accuracy']})
    # Preserve the historical report's metric (unweighted mean of the two players).
    # These are reported figures, not new evaluations or comparable fresh-Lite inputs.
    historical = [
        ('Gradient Boosting', 92.9, 57.1, 44.9), ('Random Forest', 91.8, 57.8, 48.9),
        ('SVM · RBF, scaled', 89.9, 56.0, 48.1), ('kNN7 · scaled, Manhattan', 87.9, None, 48.7),
        ('Original GRU · cached Heavy', 86.7, 56.8, 38.0), ('kNN7 · scaled, Euclidean', 85.4, None, None),
        ('kNN5 · scaled', 84.0, None, None), ('Logistic Regression', 83.1, 53.7, 48.5),
        ('kNN5 family · raw timing features', 78.3, None, 56.1),
        ('GRU · strong style augmentation', None, 55.3, 41.5), ('GRU · metric / center loss', None, None, 52.0),
        ('Moment-match TTA + ensemble', None, None, 58.3), ('SVM + moment TTA', None, None, 57.2),
        ('kNN7 + biomechanical features', None, None, 55.1), ('CORAL · kNN5 / SVM / RF', None, None, 55.7),
        ('Weighted kNN5 / kNN7 / RF', None, None, 56.6)]
    for index, (name, strat, player, diagnostic) in enumerate(historical):
        rows.append({'id': f'archive_{index}', 'group': 'archive', 'name': name, 'mode': 'all',
                     **{key: {'accuracy': value / 100} for key, value in [('stratified', strat), ('player', player), ('diagnostic', diagnostic)] if value is not None},
                     'detail': 'Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website.' + (' The raw kNN CV entry and distance-weighted diagnostic variant are family-level results, not a matched ablation.' if name.startswith('kNN5 family') else '') + (' The group-CV value comes from an older group run, not the stratified ensemble checkpoints.' if name.startswith('Original GRU') else '')})
    detectors = []
    for path, title, task in [
        ('runs/detect/ball_yolo26s/results.csv', 'Ball detector', 'Box detection'),
        ('runs/paddle_yolo26s/results.csv', 'Paddle detector · base', 'Box detection'),
        ('runs/detect/paddle_ft/results.csv', 'Paddle detector · fine-tuned', 'Box detection'),
        ('runs/pose/court_yolo26s/results.csv', 'Court landmarks · base', 'Court keypoints'),
        ('runs/pose/court_ft/results.csv', 'Court landmarks · fine-tuned', 'Court keypoints')]:
        values = list(csv.DictReader((ROOT / path).open(encoding='utf-8')))
        last = {k.strip(): float(v) for k, v in values[-1].items()}
        suffix = '(P)' if task == 'Court keypoints' else '(B)'
        detectors.append({'name': title, 'task': task, 'epochs': len(values),
                          'precision': last[f'metrics/precision{suffix}'], 'recall': last[f'metrics/recall{suffix}'],
                          'map50': last[f'metrics/mAP50{suffix}'], 'map5095': last[f'metrics/mAP50-95{suffix}'],
                          'source': copy_source(path, title.lower().replace(' · ', '-').replace(' ', '-') + '.csv')})
    samples = []
    specs = [
        ('drive-01', 'Beginner 2 · Drive 01', 'Drive', 'Coach B annotation · training clip',
         'outputs/desktop/9f424fba1dec4a51a3bbb1af913ff187', 'v2'),
        ('lob-23', 'Beginner 1 · Lob 23', 'Lob', 'Coach B annotation · training clip',
         'outputs/research_site/samples/Beginner1_Lob_023', 'original'),
        ('topspin-22', 'Beginner 1 · Topspin 22', 'Topspin', 'Coach B annotation · training clip',
         'outputs/research_site/samples/Beginner1_Topspin_022', 'original'),
        ('drive-07', 'Beginner 2 · Drive 07', 'Drive', 'Coach B annotation · training clip',
         'outputs/desktop/cf549b29380f4ca7b55b01eb0a7030f4', 'original')]
    for stem, title, truth, label_source, folder, mode in specs:
        report_path = 'outputs/serve_v2_smoke/report.json' if mode == 'v2' else folder + '/report.json'
        report = read(report_path)
        correct = report['serve']['label'].lower() == truth.lower()
        assert correct == (stem != 'drive-07'), f'Unexpected sample outcome: {stem}'
        sample = {'id': stem, 'name': title, 'truth': truth, 'label_source': label_source,
                  'tab_label': truth + ' · correct' if correct else 'Error example',
                  'role': 'correct' if correct else 'limitation',
                  **video_asset(ROOT / folder / 'input.mp4', stem), 'outputs': []}
        record = {'name': 'Retrained v2 GRU · ensemble' if mode == 'v2' else 'Original GRU · ensemble',
                  'report': report, 'download': copy_source(report_path, stem + '-saved-report.json')}
        sample['outputs'].append(record)
        samples.append(sample)
    for name, dest in [
        ('analysis/THESIS_CHAPTERS_5_6_DRAFT.md', 'chapters-5-6-draft.md'),
        ('analysis/THESIS_CHAPTERS_5_6_DRAFT.docx', 'chapters-5-6-draft.docx'),
        ('docs/RESULTS_SUMMARY.md', 'original-experiments.md'),
        ('analysis/POSE_CLASSIFIER_AUDIT.md', 'pose-audit.md'),
        ('analysis/HYBRID_CLASSIFIER_AUDIT.md', 'hybrid-audit.md'),
        ('analysis/GRU_RETRAINING_V2.md', 'gru-retraining.md'),
        ('analysis/FOUR_FOLLOWUP_STUDY.md', 'augmentation-study.md'),
        ('analysis/SERVE_REFINEMENT_V4.md', 'refinement-study.md'),
        ('models/serve_study_v3/diagnostic_evaluation.json', 'v3-diagnostic.json'),
        ('outputs/serve_refinement_v4/diagnostic_results.json', 'v4-diagnostic.json'),
        ('outputs/serve_study_v3/comparison.png', 'augmentation-comparison.png'),
        ('outputs/serve_refinement_v4/comparison.png', 'refinement-comparison.png')]:
        copy_source(name, dest)
    copy_source('docs/Pascua-Leones_Thesis.md', 'thesis.md')
    # Status only. No blind IDs, researcher key or Coach C staging videos are exported.
    status = read('outputs/coach_c_evaluation_v2/status.json')
    data = {'snapshot': '2026-10-01', 'rows': rows, 'detectors': detectors, 'samples': samples,
            'calibration': calibration, 'capture': v4['capture'],
            'coach_status': {k: status[k] for k in ['selected_beginner_clips', 'pending_coach_clips', 'prior_diagnostic_overlap', 'coach_c_annotations_pending']},
            'sources': SOURCES}
    serialized = json.dumps(data, ensure_ascii=True, indent=2)
    (ASSETS / 'research-data.json').write_text(serialized, encoding='utf-8')
    (ASSETS / 'research-data.js').write_text('window.RESEARCH_DATA = ' + serialized + ';\n', encoding='utf-8')
    (ASSETS / 'favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect width="48" height="48" rx="8" fill="#245440"/><g stroke="#e9eee7" stroke-width="2" fill="none"><rect x="12" y="7" width="24" height="34"/><path d="M12 24h24M12 17h24M12 31h24M24 7v10M24 31v10"/></g></svg>', encoding='utf-8')
    print(f'Exported {len(rows)} model/mode rows, {len(detectors)} detector runs and {len(samples)} real samples to {SITE}')


if __name__ == '__main__':
    main()
