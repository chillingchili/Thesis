import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
from desktop.hybrid import Knn5,timing_features,combine
from desktop import pipeline


class HybridTest(unittest.TestCase):
    def test_velocity_starts_at_zero(self):
        angles=np.tile(np.array([2.,2.,2.1,2.5,2.6],np.float32)[:,None],(1,5))
        window=np.stack([np.sin(angles),np.cos(angles)],axis=-1).reshape(5,10)
        feature=timing_features(window,5)
        np.testing.assert_allclose(feature[np.arange(5)*6+2],.75)

    def test_training_extractor_feature_parity_and_padding(self):
        from extract_timing_features import timing_features as offline
        angles=np.random.default_rng(8).normal(size=(75,5)).astype(np.float32)
        seq=np.stack([np.sin(angles),np.cos(angles)],axis=-1)
        padded=np.zeros((128,10),np.float32);padded[:75]=seq.reshape(75,10)
        np.testing.assert_allclose(timing_features(padded,75),offline(seq),atol=1e-6)

    def test_actual_bank_exact_neighbor(self):
        bank=Knn5(pipeline.ASSETS)
        p=bank.predict(bank.X[0])
        self.assertAlmostEqual(1.,float(p.sum()),places=6)
        self.assertEqual(bank.y[0],p.argmax())

    def test_pipeline_uses_all_gru_folds_raw_knn_and_combined_result(self):
        points=np.random.default_rng(7).random((140,33,2)).astype(np.float32)
        points[3]=np.nan
        interpreter=Mock()
        interpreter.get_input_details.return_value=[{'index':0}]
        interpreter.get_output_details.return_value=[{'index':1}]
        interpreter.get_tensor.return_value=np.array([[.8,.1,.1]],np.float32)
        bank=Mock(classes=['drive','lob','topspin'],sha256='bank')
        bank.X=np.zeros((438,32));bank.predict.return_value=np.array([0.,1.,0.],np.float32)
        config=pipeline.read_config('manifest.json')
        with patch.object(pipeline,'_MODELS',{}),patch('desktop.hybrid.Knn5',return_value=bank), \
                patch('tensorflow.lite.Interpreter',return_value=interpreter) as factory:
            result=pipeline.classify(points,'hybrid',True)
        self.assertEqual(5,factory.call_count)
        self.assertEqual('lob',result['label'])
        self.assertAlmostEqual(.55,result['confidence'],places=6)
        valid_points=points[np.isfinite(points).all(axis=(1,2))]
        raw,n=pipeline.build_window(valid_points,config,False)
        np.testing.assert_allclose(bank.predict.call_args.args[0],timing_features(raw[0],n))
        self.assertEqual(438,result['components']['knn_training_samples'])

    def test_mixing_does_not_mutate_gru(self):
        gru=np.array([.8,.1,.1],np.float32)
        np.testing.assert_allclose(combine(gru,[0,1,0]),[.4,.55,.05])
        np.testing.assert_allclose(gru,[.8,.1,.1])


if __name__=='__main__':
    unittest.main()
