import unittest
import numpy as np
from serve_refinement import lean, temperature, fit_policy, apply_policy, probability_metrics


class RefinementTests(unittest.TestCase):
    def test_lean_joint_order_and_masks(self):
        angles=np.ones((2,128,10),np.float32)
        skeleton=np.arange(2*128*99,dtype=np.float32).reshape(2,128,99)
        motion=np.arange(2*128*165,dtype=np.float32).reshape(2,128,165)
        out=lean(angles,skeleton,motion)
        self.assertEqual(out.shape,(2,128,45))
        np.testing.assert_array_equal(out[...,10:12],skeleton[...,22:24])
        np.testing.assert_array_equal(out[...,24],skeleton[...,77])
        np.testing.assert_array_equal(out[...,31:33],motion[...,121:123])

    def test_temperature_preserves_ranking_and_normalizes(self):
        p=np.array([[.99,.005,.005],[0.,1.,0.]])
        q=temperature(p,4.)
        np.testing.assert_array_equal(q.argmax(1),p.argmax(1))
        np.testing.assert_allclose(q.sum(1),1.)
        self.assertLess(q[0,0],p[0,0])

    def test_policy_uses_group_oof_and_can_disable_threshold(self):
        y=np.tile(np.arange(3),20);subjects=np.repeat(['a','b'],30)
        g=np.eye(3)[y]*.98+.02/3
        k=np.roll(g,1,axis=1)
        policy=fit_policy(g,k,y,subjects,np.arange(60).astype(str))
        self.assertEqual(policy['gru_weight'],1.)
        np.testing.assert_array_equal(apply_policy(g,k,policy).argmax(1),y)
        uniform=np.ones((60,3))/3
        policy=fit_policy(uniform,uniform,y,subjects,np.arange(60).astype(str))
        self.assertIsNone(policy['confidence_threshold'])

    def test_quality_rejection_stays_in_accuracy_denominator(self):
        y=np.array([0,1]);p=np.array([[.8,.1,.1],[.1,.8,.1]])
        r=probability_metrics(y,p,np.array(['a','a']),np.array([True,False]),.6)
        self.assertEqual(r['accuracy'],.5)
        self.assertEqual(r['selected_coverage'],.5)
        self.assertEqual(r['selected_accuracy'],1.)


if __name__=='__main__': unittest.main()
