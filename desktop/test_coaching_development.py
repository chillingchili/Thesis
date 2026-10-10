import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from desktop.coaching_development import (
    axial_summary, contact_candidate, hip_motion, pose_quality,
    evaluate_feedback, temperature, assert_partitions, ensure_development_video,
    analyze_development, fit_temperature,
    check_annotation_selection,
)


def pose(n=90):
    points = np.full((n, 33, 2), .5)
    points[:, 0, 1] = .1
    points[:, [27, 28], 1] = .9
    return points, np.ones((n, 33)), np.arange(n) / 30


class MeasurementTests(unittest.TestCase):
    def test_hip_visibility_rejects_false_motion_and_does_not_mutate_input(self):
        p, v, t = pose()
        p[:, [23, 24], 0] = np.linspace(.2, .6, len(p))[:, None]
        original = p.copy()
        self.assertEqual('ok', hip_motion(p, v, t, .1)['status'])
        v[:, 23] = 0
        self.assertEqual('unavailable', hip_motion(p, v, t, .1)['status'])
        np.testing.assert_array_equal(p, original)

    def test_missing_height_and_timestamps_fail_explicitly(self):
        p, v, t = pose()
        p[:, 0, 1] = .9
        self.assertEqual('unavailable', hip_motion(p, v, t, .1)['status'])
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            hip_motion(p, v, t[::-1], .1)

    def test_pose_visibility_is_not_class_probability(self):
        p, v, t = pose()
        v[:, 16] = .2
        q = pose_quality(p, v)
        self.assertFalse(q['accepted'])
        self.assertIn('serving arm', q['reason'])

    def test_stationary_wrist_has_no_invented_contact(self):
        p, v, t = pose()
        c = contact_candidate(p, v, t)
        self.assertEqual('unavailable', c['status'])
        self.assertIsNone(c['frame'])

    def test_early_motion_peak_does_not_fall_back_to_seventy_percent(self):
        p, v, t = pose()
        p[:10, 16, 0] = np.linspace(0, 1, 10)
        p[10:, 16, 0] = 1
        c = contact_candidate(p, v, t)
        self.assertEqual('unavailable', c['status'])
        self.assertIsNone(c['frame'])

    def test_contact_candidate_is_never_a_validated_impact(self):
        p, v, t = pose()
        p[30:50, 16, 0] = np.linspace(.5, .9, 20)
        p[50:, 16, 0] = .9
        c = contact_candidate(p, v, t)
        self.assertEqual('candidate', c['status'])
        self.assertFalse(c['validated'])
        self.assertTrue(25 <= c['frame'] <= 55)

    def test_visibility_gaps_do_not_create_speed_peaks(self):
        p, v, t = pose()
        p[45:, 16, 0] = .9
        v[43:48, 16] = 0
        self.assertEqual('unavailable', contact_candidate(p, v, t)['status'])

    def test_axial_wraparound_and_ambiguous_angles(self):
        a = axial_summary([178, 179, 1, 2])
        self.assertEqual('ok', a['status'])
        self.assertLess(min(a['angle'], 180 - a['angle']), 5)
        self.assertEqual('unavailable', axial_summary([0, 45, 90, 135])['status'])
        self.assertEqual('unavailable', axial_summary([10, 11])['status'])


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.serve = dict(label='drive', confidence=.8)
        self.quality = dict(accepted=True, reason=None)
        self.shift = dict(status='ok', below_reference=True)
        self.paddle = dict(status='unavailable', reason='No reviewed contact measurement.')

    def test_weight_prompt_does_not_invent_forward_transfer_or_arm_reliance(self):
        f = evaluate_feedback(self.serve, self.quality, self.shift, self.paddle)
        text = ' '.join(f['messages']).lower()
        self.assertIn('hips showed limited movement', text)
        self.assertNotIn('forward', text)
        self.assertNotIn('relying only', text)
        self.assertIn('Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.', f['notifications'])
        self.assertEqual('pending_expert_review', f['validation_status'])

    def test_low_confidence_does_not_blame_camera_when_tracking_is_good(self):
        f = evaluate_feedback(dict(label='lob', confidence=.4), self.quality, self.shift, self.paddle)
        self.assertEqual('LOW_CONFIDENCE', f['status'])
        self.assertFalse(f['rule_ids'])
        self.assertNotIn('visible', ' '.join(f['messages']).lower())

    def test_actual_tracking_failure_gets_specific_capture_instruction(self):
        f = evaluate_feedback(self.serve, dict(accepted=False, reason='serving arm poorly visible'), self.shift, self.paddle)
        self.assertEqual('INSUFFICIENT_TRACKING', f['status'])
        self.assertIn('visible', ' '.join(f['messages']))

    def test_unvalidated_proxy_cannot_trigger_contact_or_face_diagnosis(self):
        p = dict(status='ok', outside_reference=True, contact_validated=False)
        f = evaluate_feedback(self.serve, self.quality, dict(status='ok', below_reference=False), p)
        self.assertFalse(f['rule_ids'])
        self.assertEqual('PARTIAL', f['status'])
        self.assertNotIn('too open', ' '.join(f['messages']).lower())

    def test_reference_agreement_is_not_good_form(self):
        f = evaluate_feedback(self.serve, self.quality, dict(status='ok', below_reference=False),
                              dict(status='ok', outside_reference=False, contact_validated=True))
        self.assertNotIn('good form', ' '.join(f['messages']).lower())
        self.assertEqual('REFERENCE_OBSERVATION', f['status'])


class BoundaryTests(unittest.TestCase):
    def test_annotation_ids_cannot_be_reassigned_to_other_videos(self):
        existing = {'clips': [dict(review_id='D001', clip_id='original', video_sha256='a')]}
        check_annotation_selection(existing, [dict(clip_id='original', video_sha256='a')])
        with self.assertRaisesRegex(ValueError, 'remap'):
            check_annotation_selection(existing, [dict(clip_id='different', video_sha256='b')])

    def test_temperature_fit_uses_player_balanced_probabilities(self):
        p = np.tile([.98, .01, .01], (30, 1))
        y = np.tile([0, 1, 2], 10)
        groups = np.repeat(['A', 'B'], 15)
        value = fit_temperature(p, y, groups)
        self.assertGreater(value, 1)
        np.testing.assert_array_equal(p.argmax(1), temperature(p, value).argmax(1))

    def test_development_pose_failure_retains_baseline_and_explicit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'train.mp4'; p.write_bytes(b'train')
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            assets = dict(training_by_sha={digest: dict(split='train', clip_id='train', video_sha256=digest)},
                          excluded_video_sha256=[], hip_reference={'threshold': .1}, paddle_references={})
            base = dict(serve={'status': 'unavailable', 'reason': 'pose runtime error'}, video={})
            with patch('desktop.pipeline.analyze', return_value=base):
                result = analyze_development(p, Path(tmp) / 'out', assets)
            self.assertEqual('INSUFFICIENT_TRACKING', result['feedback']['status'])
            self.assertEqual('unavailable', result['hip_motion']['status'])
            self.assertIn('baseline_report', result)
            self.assertNotIn('visible', ' '.join(result['feedback']['messages']))

    def test_temperature_preserves_prediction_and_normalization(self):
        p = np.array([[.7, .2, .1], [.15, .8, .05]])
        calibrated = temperature(p, 2)
        np.testing.assert_array_equal(p.argmax(1), calibrated.argmax(1))
        np.testing.assert_allclose(calibrated.sum(1), 1)
        with self.assertRaises(ValueError): temperature(p, 0)

    def test_fold_and_player_leakage_is_rejected(self):
        assert_partitions(['a'], ['b'], ['A'], ['B'], ['c'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assert_partitions(['a'], ['a'], ['A'], ['B'], [])
        with self.assertRaisesRegex(ValueError, 'player'):
            assert_partitions(['a'], ['b'], ['A'], ['A'], [])
        with self.assertRaisesRegex(ValueError, 'evaluation'):
            assert_partitions(['a'], ['b'], ['A'], ['B'], ['b'])

    def test_renamed_evaluation_video_is_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'training-looking.mp4'; p.write_bytes(b'evaluation-video')
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            inventory = {digest: dict(split='train', clip_id='train-id')}
            with self.assertRaisesRegex(ValueError, 'evaluation'):
                ensure_development_video(p, inventory, {digest})
            self.assertEqual('train-id', ensure_development_video(p, inventory, set())['clip_id'])


if __name__ == '__main__':
    unittest.main()
