"""Regression checks for split isolation, augmentation geometry and missing poses."""
import unittest
import numpy as np
from prepare_serve_study import augment_frame, source_indices, variant_parameters
from serve_study import skeleton_features


class StudyTests(unittest.TestCase):
    def test_pixel_transform_reproducible_non_mutating(self):
        frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        original = frame.copy()
        p = variant_parameters("example", 1)
        a = augment_frame(frame, p, np.random.default_rng(7))
        b = augment_frame(frame, p, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(frame, original)
        self.assertFalse(np.array_equal(frame, a))
        self.assertEqual(a.shape, frame.shape)

    def test_temporal_keeps_ends_strictly_increasing(self):
        for count in (33, 128, 235):
            ix = source_indices(count, 1.08)
            self.assertEqual(ix[0], 0)
            self.assertEqual(ix[-1], count - 1)
            self.assertTrue(np.all(np.diff(ix) > 0))
            self.assertLess(len(ix), count)

    def test_skeleton_translation_scale_and_missing_joints(self):
        rng = np.random.default_rng(1)
        points = rng.uniform(.2, .8, (128, 33, 2))
        points[:, 11:13, 1] = .25
        points[:, 23:25, 1] = .65
        visibility = np.ones((128, 33))
        visibility[50, 16] = 0
        times = np.arange(128) / 30
        a, motion = skeleton_features(points, visibility, times, 16/9)
        b, _ = skeleton_features(points * .7 + .1, visibility, times, 16/9)
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertTrue(np.all(a[50, 32:34] == 0))
        self.assertTrue(np.all(motion[50:52, 99+32:99+34] == 0))
        self.assertEqual(a.shape, (128, 99))
        self.assertEqual(motion.shape, (128, 165))
        visibility[:] = 0
        a, b = skeleton_features(points, visibility, times, 1)
        self.assertFalse(a.any())
        self.assertFalse(b.any())


if __name__ == "__main__":
    unittest.main()
