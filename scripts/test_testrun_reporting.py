"""Regression checks for confidence/abstention reporting, without model fitting."""
import contextlib
import io
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

import testrun


class ReportingTests(unittest.TestCase):
    def report(self, probabilities):
        model = Mock()
        model.predict.return_value = np.array(probabilities, dtype=np.float32)
        data = (np.zeros((3, 128, 10)), np.array([0, 1, 2]), ["drive", "lob", "topspin"])
        output = io.StringIO()
        with patch.object(sys, "argv", ["testrun.py", "--keypoints_dir", "unused", "--model", "unused"]), \
                patch.object(testrun, "load_holdout", return_value=data), \
                patch.object(testrun.tf.keras.models, "load_model", return_value=model), \
                contextlib.redirect_stdout(output):
            testrun.main()
        return output.getvalue()

    def test_abstention_is_not_reported_as_a_success(self):
        output = self.report([[.8, .1, .1], [.4, .45, .15], [.1, .2, .7]])
        self.assertIn("1/3 clips abstained (below 0.6 confidence)", output)
        self.assertIn("Accuracy with abstentions counted as wrong: 66.667%", output)
        self.assertIn("Confident wrong predictions: 0/2", output)
        # Confident subset has no lob label; an explicit label list must handle it.
        self.assertIn("Scored only on clips the model was confident about", output)

    def test_all_abstain(self):
        output = self.report([[.34, .33, .33]] * 3)
        self.assertIn("Accuracy with abstentions counted as wrong: 0.000%", output)
        self.assertIn("Confident wrong predictions: 0/0", output)


if __name__ == "__main__":
    unittest.main()
