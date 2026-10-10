"""Build opt-in coaching development assets without using Coach C outcomes.

Audit cached training measurements, test nested training-only calibration, build
optionally smoke-test real videos. Uses the existing overall-feedback coach form.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from desktop.coaching_development import (
    VERSION, RULES, PENDING_MECHANICS, pose_quality, hip_motion, contact_candidate,
    axial_summary, axial_distance, temperature, fit_temperature, assert_partitions,
    ensure_development_video, analyze_development,
)
from coach_annotations import baseline_ids

OUT = ROOT / 'outputs/coaching_development_v1'
MODELS = ROOT / 'models/coaching_development_v1'
SEED = 20261010


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def protected_evaluation():
    frozen = ROOT / 'outputs/coach_c_evaluation_v2/private_session2_hybrid_20261010T045855Z'
    record = json.loads((frozen / 'freeze.json').read_text())
    protected = dict(record['frozen_files'], **record['session1_protected'])
    for name in ['freeze.json', 'summary.json', 'PRIVATE_DO_NOT_SEND_session2.zip',
                 'session2_packet_HOLD/index.html', 'session2_packet_HOLD/feedback_review.csv']:
        p = frozen / name; protected[p.relative_to(ROOT).as_posix()] = sha(p)
    for rid, digest in record['clip_sha256'].items():
        p = ROOT / 'outputs/coach_c_evaluation_v2/session1_staging/videos' / (rid + '.mp4')
        protected[p.relative_to(ROOT).as_posix()] = digest
    for name, digest in protected.items():
        if sha(ROOT / name) != digest: raise ValueError('Frozen evaluation changed: ' + name)
    with (ROOT / 'outputs/coach_c_evaluation_v2/researcher_key.csv').open(encoding='utf-8-sig') as f:
        key = list(csv.DictReader(f))
    # Read identifiers/hashes only. Neither Session 1 nor Session 2 responses are opened.
    return protected, {r['clip_id'] for r in key}, set(record['clip_sha256'].values())


def training_audit(excluded_ids, excluded_sha):
    import cv2
    inventory = json.loads((ROOT / 'data/serve_v2/dataset.json').read_text())
    rows = [r for r in inventory['clips'] if r['split'] == 'train']
    if len(rows) != 210 or set(r['subject'] for r in rows) != {'CoachA', 'Beginner1', 'Beginner2'}:
        raise ValueError('Annotated training partition changed')
    by_sha = {r['video_sha256']: r for r in rows}
    if len(by_sha) != len(rows): raise ValueError('Duplicate training video hashes')
    if set(r['clip_id'] for r in rows) & excluded_ids: raise ValueError('Evaluation ID in training')
    audits, pose_cache, values = [], {}, []
    eligible = set(baseline_ids())
    for r in rows:
        ensure_development_video(ROOT / r['video'], by_sha, excluded_sha)
        p = ROOT / 'data/serve_v2/poses' / (r['clip_id'] + '.npz')
        with np.load(p) as z:
            provenance = json.loads(str(z['provenance']))
            if provenance['video_sha256'] != r['video_sha256'] or provenance['contract_fingerprint'] != inventory['contract_fingerprint']:
                raise ValueError('Stale pose cache: ' + r['clip_id'])
            points, visibility, times = z['points'], z['visibility'], z['timestamps']
            fps = float(z['fps'])
        cap = cv2.VideoCapture(str(ROOT / r['video']))
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT); cap.release()
        quality = pose_quality(points, visibility)
        hip = hip_motion(points, visibility, times)
        contact = contact_candidate(points, visibility, times, w / h)
        pose_cache[r['clip_id']] = (points, visibility, times, fps, contact)
        if r['clip_id'] in eligible and hip['status'] == 'ok': values.append((r['clip_id'], hip['value']))
        audits.append(dict(clip_id=r['clip_id'], subject=r['subject'], coach_b_label=r['label'],
            pose_accepted=quality['accepted'], pose_valid_fraction=quality['fraction'],
            hip_status=hip['status'], hip_value=hip['value'], hip_reason=hip.get('reason'),
            contact_status=contact['status'], contact_candidate_frame=contact['frame'],
            contact_reason=contact.get('reason'), source_sha256=r['video_sha256'], pose_cache_sha256=sha(p)))
    hip_reference = dict(metric='visible_hip_x_range_body_height_norm', selection='Coach B explicit Good Form CoachA training clips',
        threshold=float(np.percentile([v for _, v in values], 10)) if len(values) >= 10 else None,
        n=len(values), clip_ids=[cid for cid, _ in values], physical_weight_transfer_validated=False)
    references, reference_audit, angles_by_class = {}, [], defaultdict(list)
    for r in rows:
        if r['clip_id'] not in eligible: continue
        p = ROOT / 'outputs/paddle_angles/ft_train' / (r['clip_id'] + '.json')
        if not p.exists(): continue
        cached = json.loads(p.read_text())
        if cached['video'].replace('\\', '/') != r['video']: raise ValueError('Paddle cache source mismatch')
        *_, contact = pose_cache[r['clip_id']]
        if len(cached['angles']) != len(pose_cache[r['clip_id']][0]) or contact['frame'] is None: continue
        c = contact['frame']
        obs = axial_summary(cached['angles'][max(0, c - 5):c + 6])
        reference_audit.append(dict(clip_id=r['clip_id'], label=r['label'], **obs, cache_sha256=sha(p)))
        if obs['status'] == 'ok': angles_by_class[r['label']].append((r['clip_id'], obs['angle']))
    candidate_stats = {}
    for label in ['drive', 'lob', 'topspin']:
        pairs = angles_by_class[label]
        if not pairs:
            candidate_stats[label] = dict(n=0, center_deg=None, radius_deg=None, selective=False)
            continue
        a = np.array([v for _, v in pairs]); center = float(a[np.argmin(axial_distance(a[:, None], a[None, :]).sum(1))])
        radius = float(np.percentile(axial_distance(a, center), 95))
        # Geometry-only development gate: references spanning over half the axial
        # domain are too broad for a selective prompt. Never tuned on evaluation.
        selective = bool(len(a) >= 5 and radius <= 45)
        candidate_stats[label] = dict(n=len(a), center_deg=center, radius_deg=radius, selective=selective)
        if selective:
            references[label] = dict(center_deg=center, radius_deg=radius, n=len(a), clip_ids=[cid for cid, _ in pairs],
                observational_only=True, face_direction_validated=False, contact_validated=False,
                detector_cache_provenance='legacy ft_train cache; weight hash was not recorded')
    assets = dict(version=VERSION, hip_reference=hip_reference, paddle_references=references,
        adapter_source_sha256=sha(ROOT / 'desktop/coaching_development.py'), builder_source_sha256=sha(__file__),
        training_by_sha=by_sha, excluded_video_sha256=sorted(excluded_sha), excluded_clip_ids=sorted(excluded_ids),
        rules=RULES, pending_mechanics=PENDING_MECHANICS, confidence_threshold=.60,
        calibration_applied=False, feedback_validation='pending_expert_review',
        annotation_workbook_sha256=sha(ROOT / 'rule-based/Deviation_Category_Mapping.xlsx'))
    csv_write(OUT / 'training_measurement_audit.csv', audits)
    dump(OUT / 'paddle_reference_audit.json', dict(clip_observations=reference_audit, candidate_stats=candidate_stats,
        selective_references=references, limitations=['Axes are not paddle-face tilt.',
            'Wrist-speed contact is unverified.', 'Old detector caches do not carry detector weight hashes.']))
    dump(MODELS / 'assets.json', assets)
    dump(OUT / 'measurement_summary.json', dict(training_clips=len(rows),
        pose_accepted=sum(r['pose_accepted'] for r in audits),
        hip_status_counts=dict(Counter(r['hip_status'] for r in audits)),
        contact_status_counts=dict(Counter(r['contact_status'] for r in audits)),
        hip_reference=hip_reference, selective_paddle_reference_classes=list(references),
        mechanics_accuracy=None, contact_accuracy=None))
    print('TRAINING_AUDIT', len(rows), 'hip baseline', len(values), 'paddle reference classes', list(references), flush=True)
    return rows, pose_cache, assets


def probability_metrics(p, y, players):
    correct = p.argmax(1) == y; confidence = p.max(1); accepted = confidence >= .6
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1))
    ece = 0.
    for lo in np.arange(0, 1, .1):
        m = (confidence >= lo) & (confidence < lo + .1 if lo < .9 else confidence <= 1)
        if m.any(): ece += float(m.mean() * abs(correct[m].mean() - confidence[m].mean()))
    return dict(n=len(y), accuracy=float(correct.mean()),
        player_balanced_log_loss=float(np.mean([loss[players == s].mean() for s in set(players)])),
        brier=float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))), ece=ece,
        coverage_at_060=float(accepted.mean()),
        accepted_accuracy_at_060=float(correct[accepted].mean()) if accepted.any() else None,
        accepted_wrong=int(np.sum(accepted & ~correct)))


def calibration_audit(excluded_ids):
    from sklearn.neighbors import KNeighborsClassifier
    from extract_timing_features import timing_features
    with np.load(ROOT / 'data/serve_v2/train.npz') as z:
        keep = z['quality_accepted']; ids, y, players, x = [z[k][keep] for k in ['clip_ids', 'y', 'subjects', 'X']]
    folder = ROOT / 'models/serve_refinement_v4/angles_thesis_aug'
    with np.load(folder / 'participant_oof.npz') as z:
        if not np.array_equal(z['clip_ids'], ids) or not np.array_equal(z['y'], y): raise ValueError('OOF membership mismatch')
        gru, knn = z['gru'], z['knn']
    timing = np.array([timing_features(a.reshape(128, 5, 2)) for a in x])
    index = {str(cid): i for i, cid in enumerate(ids)}
    calibrated = np.zeros_like(gru); folds = []
    for fold, held in enumerate(sorted(set(players)), 1):
        tr, va = np.flatnonzero(players != held), np.flatnonzero(players == held)
        assert_partitions(ids[tr].tolist(), ids[va].tolist(), sorted(set(players[tr])), [str(held)], excluded_ids)
        base_folder = ROOT / f'models/serve_study_v3/angles_thesis_aug/participant/fold{fold}'
        base_record = json.loads((base_folder / 'result.json').read_text())
        base_config = base_record['configuration']
        if base_config['train_ids'] != ids[tr].tolist() or base_config['validation_ids'] != ids[va].tolist():
            raise ValueError('Outer model partition mismatch')
        assert_partitions(base_config['train_ids'], base_config['validation_ids'],
                         base_config['train_subjects'], base_config['validation_subjects'], excluded_ids)
        if sha(base_folder / 'model.keras') != base_record['model_sha256']:
            raise ValueError('Outer model hash changed')
        if not np.allclose(gru[va], np.load(base_folder / 'predictions.npy'), atol=1e-8):
            raise ValueError('Saved GRU OOF predictions differ from their source fold')
        outer_knn = KNeighborsClassifier(5, weights='distance').fit(timing[tr], y[tr]).predict_proba(timing[va])
        if not np.allclose(knn[va], outer_knn, atol=1e-8):
            raise ValueError('Saved kNN OOF bank was not matched to the outer training fold')
        ig, ik = np.zeros((len(tr), 3)), np.zeros((len(tr), 3)); local = {int(ix): j for j, ix in enumerate(tr)}
        source_hashes = {}
        for inner, held_inner in enumerate(sorted(set(players[tr])), 1):
            path = folder / f'outer{fold}/inner{inner}'
            result = json.loads((path / 'result.json').read_text()); c = result['configuration']
            assert_partitions(c['train_ids'], c['validation_ids'], c['train_subjects'], c['validation_subjects'], excluded_ids)
            if held in c['train_subjects'] or held in c['validation_subjects']: raise ValueError('Outer player leaked into calibration')
            if sha(path / 'model.keras') != result['model_sha256']: raise ValueError('Inner model hash changed')
            it, iv = np.array([index[cid] for cid in c['train_ids']]), np.array([index[cid] for cid in c['validation_ids']])
            if hashlib.sha256(x.tobytes()).hexdigest() != c['feature_sha256']: raise ValueError('Nested feature contract mismatch')
            positions = [local[int(i)] for i in iv]
            ig[positions] = np.load(path / 'predictions.npy')
            classifier = KNeighborsClassifier(5, weights='distance').fit(timing[it], y[it])
            if not np.array_equal(classifier.classes_, [0, 1, 2]): raise ValueError('kNN class order changed')
            ik[positions] = classifier.predict_proba(timing[iv])
            source_hashes[str(path.relative_to(ROOT))] = sha(path / 'result.json')
        value = fit_temperature((ig + ik) * .5, y[tr], players[tr])
        calibrated[va] = temperature((gru[va] + knn[va]) * .5, value)
        folds.append(dict(held_player=str(held), temperature=value, fit_ids=ids[tr].tolist(),
            outer_validation_ids=ids[va].tolist(), source_result_hashes=source_hashes,
            outer_model_result_sha256=sha(base_folder / 'result.json')))
    raw = (gru + knn) * .5
    if not np.array_equal(raw.argmax(1), calibrated.argmax(1)): raise ValueError('Temperature changed class ranking')
    report = dict(scope='Saved v4 angles+augmentation candidate; not the old Android ensemble',
        fixed_50_50_hybrid=probability_metrics(raw, y, players),
        nested_temperature_only=probability_metrics(calibrated, y, players), folds=folds,
        deploy_to_current_hybrid=False, diagnostic_or_coach_c_labels_used=False,
        confidence_threshold_unchanged=.6, recognition_improved=False,
        reason='Model and preprocessing differ from the current hybrid; calibration cannot be transplanted.',
        current_hybrid_readiness='Requires matched, verifiable nested OOF predictions before calibration.',
        oof_source_sha256=sha(folder / 'participant_oof.npz'), training_array_sha256=sha(ROOT / 'data/serve_v2/train.npz'),
        existing_v4_warning='Previously fitted acceptance thresholds did not transfer between players; final policies disable acceptance.')
    dump(OUT / 'calibration_audit.json', report)
    print('CALIBRATION', json.dumps({k: report[k] for k in ['fixed_50_50_hybrid', 'nested_temperature_only']}), flush=True)
    return report


def select_training_smokes(rows):
    """Stable training-only test sample; no coach form or manual labels."""
    pools = defaultdict(list)
    for r in rows: pools[(r['subject'], r['label'])].append(r)
    rng = random.Random(SEED); selected = []
    for key in sorted(pools): selected.extend(rng.sample(sorted(pools[key], key=lambda r: r['clip_id']), 3))
    rng.shuffle(selected)
    return [dict(review_id=f'D{i:03d}', source_video=r['video'], subject=r['subject'], clip_id=r['clip_id'])
            for i, r in enumerate(selected, 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', type=int, default=0, help='Run the new adapter on N selected training videos')
    args = ap.parse_args()
    if not 0 <= args.smoke <= 27: ap.error('--smoke must be between 0 and 27')
    OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    protected, excluded_ids, excluded_sha = protected_evaluation()
    dump(OUT / 'evaluation_preservation.json', protected)
    rows, pose_cache, assets = training_audit(excluded_ids, excluded_sha)
    calibration = calibration_audit(excluded_ids)
    selected = select_training_smokes(rows)
    smoke = []
    # Cover each training participant before adding further real smoke cases.
    smoke_order, seen = [], set()
    for row in selected:
        if row['subject'] not in seen:
            smoke_order.append(row); seen.add(row['subject'])
    smoke_order += [r for r in selected if r not in smoke_order]
    for row in smoke_order[:args.smoke]:
        report = analyze_development(ROOT / row['source_video'], OUT / 'smoke' / row['review_id'], assets)
        smoke.append(dict(clip_id=row['clip_id'], feedback_status=report['feedback']['status'],
            hip_status=report['hip_motion']['status'], contact_status=report['contact']['status'],
            paddle_status=report['paddle']['status']))
        print('SMOKE', row['clip_id'], report['feedback']['status'], flush=True)
    # Include completed real reports when rerunning audits without repeating inference.
    smoke = []
    for p in sorted((OUT / 'smoke').glob('*/development_report.json')):
        report = json.loads(p.read_text())
        smoke.append(dict(clip_id=report['clip_id'], feedback_status=report['feedback']['status'],
            hip_status=report['hip_motion']['status'], contact_status=report['contact']['status'],
            paddle_status=report['paddle']['status']))
    for name, digest in protected.items():
        if sha(ROOT / name) != digest: raise ValueError('Frozen evaluation changed during development: ' + name)
    dump(OUT / 'completion.json', dict(version=VERSION, original_evaluation_preserved=True,
        training_clips_audited=len(rows), development_test_pool=len(selected),
        additional_coach_measurement_form_required=False, smoke_runs=smoke,
        calibration_deployed=False, contact_and_mechanics_accuracy=None,
        future_feature_validation_requirements=PENDING_MECHANICS,
        report_paths=['training_measurement_audit.csv', 'measurement_summary.json',
                      'calibration_audit.json']))
    print('COMPLETE_DEVELOPMENT', OUT, flush=True)


if __name__ == '__main__': main()
