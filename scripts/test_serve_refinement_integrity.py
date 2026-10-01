"""Checks the completed nested study, frozen selection and isolated mobile bundle."""
import json
import unittest
import numpy as np
from serve_study import ROOT, OUT, training_data, sha
from serve_refinement import DEST, CONDITIONS


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection=json.loads((DEST/'frozen_selection.json').read_text())
        cls.data=training_data()

    def test_every_frozen_checkpoint_and_protocol_hash(self):
        self.assertFalse(self.selection['diagnostic_used'])
        self.assertEqual(self.selection['protocol_sha256'],sha(DEST/'protocol.json'))
        self.assertEqual(len(self.selection['models']),47)
        for path,h in self.selection['models'].items(): self.assertEqual(sha(DEST/path),h,path)
        old=json.loads((OUT/'frozen_selection.json').read_text())
        for path,h in old['models'].items(): self.assertEqual(sha(OUT/path),h,path)

    def test_nested_weights_never_see_outer_player_or_original(self):
        for c in CONDITIONS:
            r=json.loads((DEST/c/'cv_results.json').read_text())
            for outer in r['folds']:
                held=outer['held_player'];expected=set(self.data['clip_ids'][self.data['subjects']!=held])
                policy=outer['policy']
                self.assertNotIn(held,policy['fit_subjects'])
                self.assertEqual(set(policy['fit_ids']),expected)
                validation=[]
                for inner in outer['inner_memberships']:
                    tr=set(inner['train_ids']);va=set(inner['validation_ids'])
                    self.assertFalse(tr & va);self.assertEqual(tr | va,expected)
                    self.assertNotIn(held,inner['train_subjects']+inner['validation_subjects'])
                    self.assertFalse(set(inner['train_subjects']) & set(inner['validation_subjects']))
                    validation.extend(inner['validation_ids'])
                self.assertEqual(set(validation),expected);self.assertEqual(len(validation),len(expected))

    def test_all_training_and_diagnostic_probabilities_are_valid(self):
        for c in CONDITIONS:
            z=np.load(DEST/c/'participant_oof.npz')
            self.assertEqual(z['clip_ids'].tolist(),self.data['clip_ids'].tolist())
            for key in ['gru','knn','tuned']:
                self.assertTrue(np.isfinite(z[key]).all())
                np.testing.assert_allclose(z[key].sum(1),1.,atol=1e-6)
            t=np.load(DEST/c/'diagnostic_predictions.npz')
            self.assertFalse(set(t['clip_ids']) & set(z['clip_ids']))
            self.assertEqual(len(t['clip_ids']),175)

    def test_research_bundle_matches_selected_model_and_conversion(self):
        manifest=json.loads((DEST/'android_bundle/manifest.json').read_text())
        parity=json.loads((DEST/'android_bundle/parity.json').read_text())
        self.assertTrue(parity['passed']);self.assertEqual(parity['cases'],175)
        self.assertEqual(manifest['condition'],self.selection['selected_condition'])
        self.assertFalse(manifest['moment_matching'])
        for name,key in [('model.tflite','model_sha256'),('knn_train.bin','knn_sha256')]:
            self.assertEqual(sha(DEST/'android_bundle'/name),manifest[key])
            self.assertEqual(sha(ROOT/'android/app/src/main/assets/serve_research'/name),manifest[key])
        old=json.loads((ROOT/'outputs/serve_study_v3/hybrid/configuration.json').read_text())
        for name,h in old['assets'].items(): self.assertEqual(sha(ROOT/'android/app/src/main/assets'/name),h)


if __name__=='__main__':unittest.main()
