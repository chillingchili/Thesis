"""Meaningful leakage, Android-window and human-review regression checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from train_serve_study import fold_membership
from benchmark_serve_hybrid import padded, gate_replay, moment_match
from serve_study import training_data, training_rows, metrics, ROOT, OUT, DATA, PROTOCOL, sha
import score_coach_review


class IntegrityTests(unittest.TestCase):
    def test_original_and_player_split_isolation(self):
        d=training_data();ids=d["clip_ids"];subjects=d["subjects"]
        for tr,va in LeaveOneGroupOut().split(d["X"],d["y"],subjects):
            result=fold_membership(ids,subjects,tr,va,True)
            augmented_parents=np.repeat(ids[tr],2)
            self.assertFalse(set(augmented_parents)&set(result["validation_ids"]))
            self.assertFalse(set(result["train_subjects"])&set(result["validation_subjects"]))
        with self.assertRaises(ValueError):
            fold_membership(ids,subjects,np.array([0,1]),np.array([1,2]))
        with self.assertRaises(ValueError):
            fold_membership(ids,subjects,np.array([0]),np.array([1]),True)

    def test_rolling_buffer_keeps_end_and_pads_only_tail(self):
        a=np.arange(150*10,dtype=np.float32).reshape(150,10)
        w,n=padded(a)
        self.assertEqual(n,128);np.testing.assert_array_equal(w,a[-128:])
        w,n=padded(a[:20]);self.assertEqual(n,20)
        np.testing.assert_array_equal(w[:20],a[:20]);self.assertFalse(w[20:].any())

    def test_gate_requires_completed_motion_and_minimum_frames(self):
        angles=np.zeros((120,5),np.float32)
        angles[20:75]=np.sin(np.arange(55)[:,None]*.8)*.7
        x=np.stack([np.sin(angles),np.cos(angles)],axis=-1).reshape(120,10)
        events,active=gate_replay(x,np.arange(120)/30)
        self.assertTrue(any(e["available"] for e in events));self.assertFalse(active)
        events,active=gate_replay(x[:50],np.arange(50)/30)
        self.assertFalse(events);self.assertTrue(active)
        still=np.tile(x[0],(120,1))
        events,active=gate_replay(still,np.arange(120)/30)
        self.assertFalse(events);self.assertFalse(active)

    def test_moment_matching_does_not_populate_padding(self):
        rng=np.random.default_rng(7)
        x=np.zeros((128,10),np.float32);x[:32]=rng.normal(size=(32,10))
        m=dict(moment_matching=True,feature_mean=[.3]*10,feature_std=[.4]*10)
        w=moment_match(x,32,m)
        self.assertFalse(w[32:].any())
        np.testing.assert_allclose(w[:32].mean(0),.3,atol=1e-6)
        np.testing.assert_allclose(w[:32].std(0),.4,atol=1e-6)

    def test_unavailable_predictions_are_not_counted_correct(self):
        r=metrics(np.array([0,1]),np.array([[1,0,0],[0,1,0]],float),available=[False,True])
        self.assertEqual(r["correct"],1);self.assertEqual(r["accuracy"],.5)
        self.assertEqual(r["available"],1)

    def test_coach_blanks_disagreements_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)
            (dest/"researcher_key.csv").write_text("review_id,clip_id,original_label\nR001,training_example,drive\n",encoding="utf-8")
            row=dict(review_id="R001",reviewer="",review_date="",observed_subtype="",subtype_certainty="",
                     subtype_visible_from_video="",pose_fidelity="",wrong_person="",wrist_or_hand_tracking_error="",occlusion_or_blur="")
            path=dest/"responses.csv"
            def save(rows):
                with path.open("w",newline="",encoding="utf-8") as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
            with patch.object(score_coach_review,"DEST",dest):
                save([row]);r=score_coach_review.score(path)
                self.assertIsNone(r["sample_agreement"]);self.assertEqual(r["pending_subtype_judgments"],1)
                row.update(reviewer="Synthetic test reviewer",review_date="2026-10-01",observed_subtype="lob")
                save([row]);r=score_coach_review.score(path)
                self.assertEqual(r["sample_agreement"],0);self.assertEqual(len(r["disagreements"]),1)
                self.assertFalse(r["labels_changed"])
                save([row,row])
                with self.assertRaises(ValueError):score_coach_review.score(path)

    def test_saved_completed_folds_have_equal_budget_and_disjoint_ids(self):
        for path in OUT.glob("*/*/fold*/result.json"):
            r=json.loads(path.read_text());c=r["configuration"]
            self.assertFalse(set(c["train_ids"])&set(c["validation_ids"]))
            self.assertEqual(sum(r["augmentation_choices"]),len(c["train_ids"])*100)
            self.assertEqual(len(r["history"]),100)
            self.assertFalse(r["outer_validation_passed_to_fit"])

    def test_completed_augmentation_corpus_has_frozen_training_provenance(self):
        marker=DATA/"extraction_complete.json"
        if not marker.exists():
            self.skipTest("Full augmented corpus is still extracting")
        complete=json.loads(marker.read_text())
        self.assertEqual(complete["variants"],418)
        self.assertEqual(complete["protocol_sha256"],sha(DATA/"protocol.json"))
        self.assertEqual(complete["skeleton_sha256"],sha(DATA/"skeleton_train.npz"))
        protocol_hash=sha(DATA/"protocol.json")
        extractor_hash=sha(ROOT/"scripts/prepare_serve_study.py")
        accepted=0
        for row in training_rows():
            self.assertEqual(row["split"],"train")
            for v in (1,2):
                with np.load(DATA/"augmented"/f"{row['clip_id']}__a{v}.npz") as saved:
                    provenance=json.loads(str(saved["provenance"]))
                    self.assertEqual(provenance["original_id"],row["clip_id"])
                    self.assertEqual(provenance["video_sha256"],row["video_sha256"])
                    self.assertEqual(provenance["protocol_sha256"],protocol_hash)
                    self.assertEqual(provenance["extractor_sha256"],extractor_hash)
                    self.assertEqual(saved["X"].shape,(128,10))
                    self.assertTrue(np.isfinite(saved["X"]).all())
                    self.assertTrue(np.all(np.diff(saved["source_indices"])>0))
                    self.assertEqual(saved["source_indices"][0],0)
                    pairs=saved["X"].reshape(128,5,2)
                    present=np.any(saved["X"]!=0,axis=1)
                    np.testing.assert_allclose(np.linalg.norm(pairs[present],axis=2),1,atol=1e-5)
                    accepted+=int(json.loads(str(saved["quality"]))["quality_accepted"])
        self.assertEqual(accepted,complete["accepted"])

    def test_frozen_selection_contains_all_expected_models(self):
        target=OUT/"frozen_selection.json"
        if not target.exists():
            self.skipTest("All candidate models have not yet been frozen")
        frozen=json.loads(target.read_text())
        expected={f"{c}/{protocol}/fold{i}/model.keras" for c in PROTOCOL["conditions"]
                  for protocol,n in [("participant",3),("stratified",5)] for i in range(1,n+1)}
        self.assertEqual({p.replace("\\","/") for p in frozen["models"]},expected)
        self.assertFalse(frozen["diagnostic_test_used"])
        for path,digest in frozen["models"].items():
            self.assertEqual(sha(OUT/path),digest)


if __name__=="__main__":
    unittest.main()
