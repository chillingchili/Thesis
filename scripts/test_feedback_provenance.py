"""Check the actual bundled assets against Coach B annotations, not synthetic labels."""
import json
import unittest
from pathlib import Path
from coach_annotations import ROOT, baseline_ids, read_annotations, provenance


class FeedbackProvenanceTest(unittest.TestCase):
    def test_baselines_use_only_explicit_good_form_training_rows(self):
        eligible = set(baseline_ids())
        for filename in ("shift_config.json", "paddle_config.json"):
            cfg = json.loads((ROOT / "android/app/src/main/assets" / filename).read_text())
            selected = set(cfg["baseline_clip_ids"])
            excluded = set(cfg["excluded_baseline_clip_ids"])
            self.assertTrue(selected)
            self.assertFalse(selected & excluded)
            self.assertEqual(eligible, selected | excluded)
            self.assertEqual(len(selected), cfg["n_coach_clips"])
            self.assertEqual(provenance()["annotation_sha256"], cfg["annotation_sha256"])
            self.assertNotIn("CoachA_Drive_004", selected)  # explicitly Bad Form

    def test_rule_sources_are_annotated_training_bad_form_clips(self):
        rows = {r["Filename"]: r for r in read_annotations()}
        rules = json.loads((ROOT / "android/app/src/main/assets/feedback_rules.json").read_text())
        self.assertEqual("pending_coach_c", rules["validation_status"])
        for rule in rules["rules"]:
            category = "C4_Lower_Body" if rule["id"].startswith("C4") else "C6_Paddle_Face"
            for clip in rule["source_clips"]:
                self.assertEqual("Bad Form", rows[clip]["Form Judgment"])
                self.assertEqual("1", rows[clip][category])
                self.assertTrue(rows[clip]["Deviation Comment"])


if __name__ == "__main__":
    unittest.main()
