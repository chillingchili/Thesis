import unittest
import numpy as np

from serve_sequence import build_sequence, resample_angles


def features(theta):
    theta = np.asarray(theta)
    return np.tile(np.stack([np.sin(theta), np.cos(theta)], axis=-1), (1,5)).astype(np.float32)


class SequenceTest(unittest.TestCase):
    def test_late_motion_and_endpoints_survive(self):
        x = features(np.linspace(0, 2, 240))
        y = resample_angles(x)
        np.testing.assert_allclose(y[0], x[0], atol=1e-6)
        np.testing.assert_allclose(y[-1], x[-1], atol=1e-6)
        self.assertGreater(np.arctan2(y[-1,0], y[-1,1]), 1.9)
        np.testing.assert_allclose(np.linalg.norm(y.reshape(128,5,2),axis=2), 1, atol=1e-6)

    def test_frame_rate_independence(self):
        t30, t60 = np.linspace(0,2,61), np.linspace(0,2,121)
        a, b = resample_angles(features(t30), t30), resample_angles(features(t60), t60)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_wraparound_and_missing_frames(self):
        pair = features(np.radians([179, -179]))
        middle = resample_angles(pair, seq_len=3)[1]
        self.assertLess(middle[1], -.999)
        x = features(np.linspace(0,1,128)); x[40:50] = 0
        y = resample_angles(x)
        np.testing.assert_array_equal(y[40:50], 0)
        self.assertTrue(np.any(y[39]))

    def test_low_visibility_is_masked_and_rejected(self):
        points = np.random.default_rng(4).random((128,33,2)).astype(np.float32)
        visibility = np.ones((128,33), np.float32)
        visibility[:40,16] = .1
        window, info = build_sequence(points, visibility)
        np.testing.assert_array_equal(window[:40], 0)
        self.assertFalse(info["quality_accepted"])
        self.assertEqual(info["truncated_frames"], 0)

    def test_bad_time_axis_rejected(self):
        with self.assertRaises(ValueError):
            resample_angles(features([0,1]), [0,0])


class AugmentationTest(unittest.TestCase):
    def test_padding_does_not_turn_into_fake_pose(self):
        from train_gru import SequenceAugment
        layer = SequenceAugment(rate_min=1., rate_max=1., max_shift=0, frame_drop=0)
        x = np.zeros((1,128,10), np.float32)
        x[0,:40] = features(np.linspace(0,1,40))
        y = layer(x, training=True).numpy()
        np.testing.assert_array_equal(y[0,40:], 0)
        np.testing.assert_array_equal(layer(np.zeros_like(x), training=True).numpy(), 0)

    def test_dropped_frames_stay_masked(self):
        from train_gru import SequenceAugment
        layer = SequenceAugment(rate_min=1., rate_max=1., max_shift=0, frame_drop=1)
        x = features(np.linspace(0,1,128))[None]
        np.testing.assert_array_equal(layer(x, training=True).numpy(), 0)


if __name__ == "__main__":
    unittest.main()
