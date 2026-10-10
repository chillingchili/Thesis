"""Opt-in measurement-bound coaching; the evaluated desktop/Android stays frozen.

Tracking quality is not biomechanical correctness. A wrist-motion peak is not a
verified impact, and a paddle-axis angle is not a physical paddle-face tilt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'coaching_development_v1'
VISIBILITY = .5
MIN_FRACTION = .8
RULES = [
    dict(id='DEV_HIP_MOTION', condition='hip_motion_below_reference',
         text='Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.',
         source_clips=['Beginner1_Lob_016', 'Beginner2_Drive_019', 'Beginner2_Topspin_022'],
         review_status='pending_expert_review', measurement='image-plane hip displacement; not forward transfer or force'),
    dict(id='DEV_PADDLE_AXIS', condition='reviewed_contact_axis_outside_reference',
         text='Your paddle orientation differed from the reference serves. Try a slower practice swing, keep your grip relaxed, and maintain a consistent paddle position through contact.',
         source_clips=['Beginner2_Drive_010', 'Beginner1_Lob_019', 'Beginner2_Topspin_002'],
         review_status='pending_expert_review', measurement='image-plane long axis; not face tilt'),
]
PENDING_MECHANICS = ['paddle_face_open_closed', 'forward_weight_transfer',
                    'wrist_lag', 'toss_control', 'follow_through', 'swing_path']


def _inputs(points, visibility, timestamps):
    p = np.asarray(points, float)
    v = np.asarray(visibility, float)
    t = np.asarray(timestamps, float)
    if p.ndim != 3 or p.shape[1:] != (33, 2) or v.shape != p.shape[:2]:
        raise ValueError('Expected per-frame 33-joint points and visibility')
    if len(p) < 3 or t.shape != (len(p),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid or non-increasing timestamps')
    return p, v, t


def _valid(p, v, joints):
    values = p[:, joints]
    return (np.isfinite(values).all((1, 2)) & np.isfinite(v[:, joints]).all(1)
            & (v[:, joints] >= VISIBILITY).all(1)
            & (values >= 0).all((1, 2)) & (values <= 1).all((1, 2)))


def pose_quality(points, visibility):
    p, v, _ = _inputs(points, visibility, np.arange(len(points), dtype=float))
    valid = _valid(p, v, [11, 12, 14, 16, 20, 23, 24])
    accepted = int(valid.sum()) >= 32 and float(valid.mean()) >= MIN_FRACTION
    return dict(accepted=bool(accepted), valid_frames=int(valid.sum()), fraction=float(valid.mean()),
                reason=None if accepted else 'The serving arm or torso is poorly visible in too many frames.')


def hip_motion(points, visibility, timestamps, threshold=None):
    p, v, t = _inputs(points, visibility, timestamps)
    valid = _valid(p, v, [0, 23, 24, 27, 28])
    result = dict(status='unavailable', metric='visible_hip_x_range_body_height_norm',
                  valid_frames=int(valid.sum()), fraction=float(valid.mean()),
                  below_reference=None, value=None, threshold=threshold,
                  scope='image-plane hip motion, not physical forward weight transfer')
    if valid.sum() < 10 or valid.mean() < MIN_FRACTION:
        return dict(result, reason='Too few visible hips, nose, or ankles for hip-motion assessment.')
    height = float(np.median(np.abs(p[valid, 0, 1] - p[valid][:, [27, 28], 1].mean(1))))
    if height < .05:
        return dict(result, reason='Body-height reference is too small or invalid.')
    hip = p[:, [23, 24], 0].mean(1)
    # Smooth only completely valid adjacent windows; never join visibility gaps.
    smooth = [float(np.mean(hip[i:i + 5])) for i in range(len(hip) - 4) if valid[i:i + 5].all()]
    if len(smooth) < 10:
        return dict(result, reason='Insufficient continuous visible hip motion.')
    value = float((np.percentile(smooth, 95) - np.percentile(smooth, 5)) / height)
    below = bool(value < threshold) if threshold is not None else None
    return dict(result, status='ok', reason=None, value=value, below_reference=below)


def contact_candidate(points, visibility, timestamps, aspect=1.):
    p, v, t = _inputs(points, visibility, timestamps)
    unavailable = dict(status='unavailable', frame=None, validated=False,
                       method='visibility-aware wrist-speed proxy', reason=None)
    valid = _valid(p, v, [14, 16])
    pairs = valid[:-1] & valid[1:]
    if pairs.sum() < 10:
        return dict(unavailable, reason='Too few adjacent visible wrist frames.')
    wrist = p[:, 16] * [float(aspect), 1.]
    speed = np.zeros(len(p) - 1)
    speed[pairs] = np.linalg.norm(np.diff(wrist, axis=0)[pairs], axis=1) / np.diff(t)[pairs]
    filtered = np.zeros_like(speed)
    for i in range(len(speed)):
        lo, hi = max(0, i - 2), min(len(speed), i + 3)
        if pairs[i] and pairs[lo:hi].sum() >= 3:
            filtered[i] = np.median(speed[lo:hi][pairs[lo:hi]])
    peak = int(filtered.argmax()) + 1
    if filtered.max() <= 1e-5:
        return dict(unavailable, reason='No sustained wrist motion peak was observed.')
    if peak < .2 * len(p) or peak >= .95 * len(p):
        return dict(unavailable, reason='Motion peak is at the recording boundary; contact cannot be located reliably.')
    return dict(status='candidate', frame=peak, seconds=float(t[peak]), validated=False,
                method='visibility-aware wrist-speed proxy', speed=float(filtered.max()),
                reason='Swing-motion candidate only; requires contact-frame review.')


def axial_distance(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 90) % 180 - 90)


def axial_summary(angles, min_n=3):
    a = np.asarray(angles, float).reshape(-1)
    a = a[np.isfinite(a)] % 180
    result = dict(status='unavailable', angle=None, measurements=len(a), concentration=0.)
    if len(a) < min_n:
        return dict(result, reason='Too few paddle-axis measurements.')
    concentration = float(abs(np.exp(2j * np.deg2rad(a)).mean()))
    if concentration < .75:
        return dict(result, concentration=concentration, reason='Paddle-axis measurements are inconsistent.')
    # Axial medoid handles 179°/1° without inventing a 90° orientation.
    center = float(a[np.argmin(axial_distance(a[:, None], a[None, :]).sum(1))])
    return dict(status='ok', angle=center, measurements=len(a), concentration=concentration,
                spread_p90=float(np.percentile(axial_distance(a, center), 90)), reason=None)


def evaluate_feedback(serve, quality, shift, paddle):
    notes = []
    if shift.get('status') != 'ok': notes.append('Hip-motion assessment unavailable.')
    if paddle.get('status') != 'ok': notes.append('Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.')
    elif not paddle.get('contact_validated', False):
        notes.append('Paddle was observed near an estimated swing peak; contact and paddle-face direction are unverified.')
    def result(status, messages, ids=()):
        return dict(status=status, messages=messages, notifications=notes, rule_ids=list(ids),
                    version=VERSION, validation_status='pending_expert_review',
                    overall_form_assessed=False, disabled_mechanics=PENDING_MECHANICS)
    if not quality.get('accepted', False):
        instruction = ('Record a complete side-view serve with the serving arm and torso visible throughout.'
                       if quality.get('capture_issue', True) else
                       'The measurement pipeline could not complete; inspect the reported processing error.')
        return result('INSUFFICIENT_TRACKING', [quality.get('reason') or 'Pose tracking is insufficient.', instruction])
    confidence = serve.get('confidence')
    if (serve.get('label', '').lower() not in {'drive', 'lob', 'topspin'} or confidence is None
            or not np.isfinite(confidence) or not 0 <= confidence <= 1):
        return result('INSUFFICIENT_DATA', ['Serve classification is unavailable; no corrective coaching is supported.'])
    if confidence < .60:
        return result('LOW_CONFIDENCE', ['The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through.'])
    messages, ids = [], []
    if shift.get('status') == 'ok' and shift.get('below_reference') is True:
        messages.append(RULES[0]['text']); ids.append(RULES[0]['id'])
    if (paddle.get('status') == 'ok' and paddle.get('contact_validated')
            and paddle.get('outside_reference') is True):
        messages.append(RULES[1]['text']); ids.append(RULES[1]['id'])
    if messages:
        return result('MEASUREMENT_PROMPT', messages, ids)
    if (shift.get('status') == 'ok' and shift.get('below_reference') is False
            and paddle.get('status') == 'ok' and paddle.get('contact_validated')
            and paddle.get('outside_reference') is False):
        return result('REFERENCE_OBSERVATION', ['Your measured hip movement and paddle orientation are within the reference ranges. Repeat the motion and focus on consistency.'])
    return result('PARTIAL', ['No corrective cue is supported by the available measurements. Overall serve form has not been assessed.'])


def temperature(p, value):
    p = np.asarray(p, float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(1), 1, atol=.001) or not np.isfinite(value) or value <= 0):
        raise ValueError('Invalid probabilities or temperature')
    logits = np.log(np.clip(p, 1e-7, 1)) / value
    logits -= logits.max(1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


def assert_partitions(train_ids, validation_ids, train_players, validation_players, excluded_ids):
    if not train_ids or not validation_ids or set(train_ids) & set(validation_ids):
        raise ValueError('Empty partition or clip overlap')
    if set(train_players) & set(validation_players):
        raise ValueError('Training/validation player overlap')
    if (set(train_ids) | set(validation_ids)) & set(excluded_ids):
        raise ValueError('Coach C evaluation clip entered development')


def fit_temperature(p, y, players):
    from scipy.optimize import minimize_scalar
    p, y, players = np.asarray(p, float), np.asarray(y), np.asarray(players)
    temperature(p, 1)
    if len(y) != len(p) or len(players) != len(p) or len(p) < 20 or len(set(players)) < 2:
        raise ValueError('Need at least 20 group-held-out predictions from two players')
    if not np.isin(y, [0, 1, 2]).all(): raise ValueError('Invalid class labels')
    def loss(logt):
        calibrated = temperature(p, np.exp(logt))
        values = -np.log(np.clip(calibrated[np.arange(len(y)), y], 1e-7, 1))
        return float(np.mean([values[players == s].mean() for s in set(players)]))
    fit = minimize_scalar(loss, bounds=(np.log(.25), np.log(10)), method='bounded')
    if not fit.success: raise ValueError('Temperature optimization did not converge')
    return float(np.exp(fit.x))


def ensure_development_video(path, inventory_by_sha, excluded_sha):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest in excluded_sha:
        raise ValueError('Coach C evaluation video is excluded, even if renamed')
    row = inventory_by_sha.get(digest)
    if not row or row['split'] != 'train':
        raise ValueError('Only frozen training videos are permitted for development')
    return row


def check_annotation_selection(existing, selected):
    if existing is None: return
    old = [(r['review_id'], r['clip_id'], r['video_sha256']) for r in existing['clips']]
    new = [(f'D{i:03d}', r['clip_id'], r['video_sha256']) for i, r in enumerate(selected, 1)]
    if old != new:
        raise ValueError('Annotation selection changed; do not remap existing expert responses')


def paddle_observation(video, points, visibility, contact, reference, out):
    if contact.get('frame') is None:
        return dict(status='unavailable', reason=contact['reason'], contact_validated=False)
    import cv2
    from desktop import pipeline
    from extract_paddle_angles import pick_det, expand_box, crop_angle
    model = pipeline._MODELS.get('paddle')
    if model is None:
        from ultralytics import YOLO
        model = YOLO(str(ROOT / 'runs/detect/paddle_ft/weights/best.pt'))
    cap = cv2.VideoCapture(str(video))
    angles = []
    first, last = max(0, contact['frame'] - 5), min(len(points), contact['frame'] + 6)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    try:
        for i in range(first, last):
            ok, frame = cap.read()
            if not ok: break
            if not _valid(points[i:i + 1], visibility[i:i + 1], [14, 16])[0]: continue
            res = model.predict(frame, imgsz=640, conf=.25, verbose=False, device='cpu')[0]
            box, _ = pick_det(res, frame.shape[:2], points, i)
            if box is None: continue
            x1, y1, x2, y2 = expand_box(box, frame.shape[:2])
            angle, _, _, status = crop_angle(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))
            if status == 'ok': angles.append(angle)
            if i == contact['frame']: cv2.imwrite(str(out / 'development_contact_candidate.jpg'), frame)
    finally:
        cap.release()
    result = axial_summary(angles)
    result.update(contact_validated=bool(contact.get('validated')), face_direction_validated=False,
                  measurement='paddle long-axis orientation in image plane', outside_reference=None)
    if result['status'] == 'ok' and reference:
        result.update(reference=reference, distance_deg=float(axial_distance(result['angle'], reference['center_deg'])),
                      outside_reference=bool(axial_distance(result['angle'], reference['center_deg']) > reference['radius_deg']))
    return result


def analyze_development(primary, out, assets, reviewed_contact=None):
    """Explicit development entry point; never updates the evaluated model assets."""
    from desktop import pipeline
    primary, out = Path(primary), Path(out)
    record = ensure_development_video(primary, assets['training_by_sha'], set(assets['excluded_video_sha256']))
    out.mkdir(parents=True, exist_ok=True)
    options = dict(start=0, end=None, model='hybrid', landing='off', moment_matching=True)
    base = pipeline.analyze(primary, None, out, options, lambda *_: None, threading.Event())
    if not (out / 'pose_details.npz').exists():
        reason = base['serve'].get('reason', 'Pose extraction unavailable.')
        q = dict(accepted=False, reason=reason, capture_issue=False)
        shift = dict(status='unavailable', reason=reason)
        contact = dict(status='unavailable', frame=None, validated=False, reason=reason)
        paddle = dict(status='unavailable', reason=reason)
    else:
        with np.load(out / 'pose_details.npz') as d:
            v, t = d['visibility'], d['timestamps']
        points = np.load(out / 'keypoints.npy')
        q = pose_quality(points, v)
        shift = hip_motion(points, v, t, assets['hip_reference']['threshold'])
        contact = contact_candidate(points, v, t, base['video']['width'] / base['video']['height'])
        if reviewed_contact is not None:
            if (reviewed_contact.get('video_sha256') != record['video_sha256']
                    or not reviewed_contact.get('reviewer') or not reviewed_contact.get('review_date')
                    or reviewed_contact.get('certainty') != 'clear'):
                raise ValueError('Reviewed contact needs matching source hash and dated reviewer provenance')
            frame = reviewed_contact['frame']
            if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < len(points):
                raise ValueError('Reviewed contact frame outside video or not an integer')
            contact = dict(status='reviewed', frame=frame, validated=True,
                           method='supplied development contact annotation', provenance=reviewed_contact, reason=None)
        reference = assets['paddle_references'].get(base['serve'].get('label', '').lower())
        try:
            paddle = paddle_observation(primary, points, v, contact, reference, out)
        except Exception as exc:
            paddle = dict(status='unavailable', reason=str(exc), contact_validated=bool(contact.get('validated')))
    feedback = evaluate_feedback(base['serve'], q, shift, paddle)
    report = dict(version=VERSION, clip_id=record['clip_id'], video_sha256=record['video_sha256'],
                  adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  classifier=base['serve'], classifier_unchanged=True, calibration_applied=False,
                  quality=q, hip_motion=shift, contact=contact, paddle=paddle, feedback=feedback,
                  baseline_report=base, limitations=['Development wording is pending expert validation.',
                      'Contact proxy is not verified impact; face-direction advice stays disabled.',
                      'Current hybrid has no matched, validated calibration policy.'])
    (out / 'development_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
