import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

import numpy as np
from fastapi.testclient import TestClient

from desktop import app as server
from desktop.feedback import evaluate
from desktop.pipeline import ASSETS, build_window, analyze, classify, Cancelled


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        self.rules = json.loads((ASSETS / "feedback_rules.json").read_text())

    def test_confidence_and_partial_signal_contract(self):
        for confidence in [None, float("nan"), float("inf"), -1, 1.1, 0.59]:
            r = evaluate(self.rules, "drive", confidence, False, "OUTSIDE_BASELINE")
            self.assertFalse(r["rule_ids"])
        r = evaluate(self.rules, "drive", .6, False, None)
        self.assertEqual(["C4_WEIGHT_TRANSFER"], r["rule_ids"])
        self.assertIn("Paddle orientation unavailable.", r["notifications"])

    def test_two_cues_and_optional_landing(self):
        for subtype in ["drive", "lob", "topspin"]:
            r = evaluate(self.rules, subtype, .9, False, "OUTSIDE_BASELINE")
            self.assertEqual(["C4_WEIGHT_TRANSFER", "C6_PADDLE_REVIEW"], r["rule_ids"])
        self.assertEqual("POSITIVE", evaluate(self.rules, "drive", .9, True, "OPTIMAL")["status"])
        for zone in [None, "Fault-Long", "Fault-Wide", "Fault-Short", "invalid"]:
            self.assertEqual("PARTIAL", evaluate(self.rules, "drive", .9, True, "OPTIMAL", True, zone)["status"])

    def test_directional_rules_remain_subtype_specific(self):
        self.assertEqual(["C6_TOO_CLOSED"], evaluate(self.rules, "topspin", .9, True, "TOO_CLOSED")["rule_ids"])
        self.assertFalse(evaluate(self.rules, "drive", .9, True, "TOO_CLOSED")["rule_ids"])


class WindowTest(unittest.TestCase):
    def test_candidate_contract_and_model_cache_are_separate(self):
        from serve_sequence import contract, fingerprint, build_sequence
        from desktop import pipeline
        points = np.random.default_rng(9).random((200,33,2)).astype(np.float32)
        visibility = np.ones((200,33), np.float32)
        timestamps = np.arange(200)/60
        interpreter = Mock()
        interpreter.get_input_details.return_value = [{"index":0}]
        interpreter.get_output_details.return_value = [{"index":1}]
        interpreter.get_tensor.return_value = np.array([[.8,.1,.1]], np.float32)
        manifest = dict(classes=["drive","lob","topspin"], ensemble=["gru_fold1.tflite"],
                        single="gru_single.tflite", preprocessing=contract(), contract_fingerprint=fingerprint(),
                        cv_target_met=False)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/"manifest.json").write_text(json.dumps(manifest))
            with patch.object(pipeline,"CANDIDATE_ASSETS",directory), patch.object(pipeline,"_MODELS",{}), \
                    patch("tensorflow.lite.Interpreter",return_value=interpreter):
                result = classify(points,"v2_ensemble",True,visibility,timestamps)
                expected,_ = build_sequence(points,visibility,timestamps)
                np.testing.assert_array_equal(interpreter.set_tensor.call_args.args[1][0],expected)
                self.assertEqual(0,result["truncated_frames"])
                self.assertFalse(result["moment_matching"])
                self.assertFalse(result["feedback_eligible"])
                self.assertIn(str(directory/"gru_fold1.tflite"),pipeline._MODELS)
            manifest["contract_fingerprint"]="wrong"
            (directory/"manifest.json").write_text(json.dumps(manifest))
            with patch.object(pipeline,"CANDIDATE_ASSETS",directory), self.assertRaisesRegex(ValueError,"does not match"):
                classify(points,"v2_single",False,visibility,timestamps)

    def test_candidate_below_cv_target_withholds_coaching(self):
        points = np.random.default_rng(3).random((90,33,2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp, \
                patch("desktop.pipeline.video_info",return_value=dict(fps=30,frames=90,duration=3,width=640,height=360)), \
                patch("desktop.pipeline.extract_pose",return_value=points), \
                patch("desktop.pipeline.classify",return_value=dict(status="ok",label="drive",confidence=.99,truncated_frames=0,feedback_eligible=False)), \
                patch("desktop.pipeline.paddle_analysis",return_value=dict(status="ok",state="OPTIMAL")), \
                patch("desktop.pipeline.encode_video"):
            r=analyze(Path("primary.mp4"),None,Path(tmp),dict(start=0,end=3,model="single",moment_matching=False,landing="off"),lambda *a:None,threading.Event())
            self.assertEqual("drive",r["serve"]["label"])
            self.assertFalse(r["feedback"]["rule_ids"])
            self.assertTrue(any("85%" in warning for warning in r["warnings"]))

    def test_contact_bound_applies_only_to_same_view_landing(self):
        points = np.random.default_rng(3).random((90,33,2)).astype(np.float32)
        for mode, expected in [("primary",55),("secondary",None)]:
            with tempfile.TemporaryDirectory() as tmp, \
                 patch("desktop.pipeline.video_info", return_value=dict(fps=30,frames=90,duration=3,width=640,height=360)), \
                 patch("desktop.pipeline.extract_pose", return_value=points), \
                 patch("desktop.pipeline.classify", return_value=dict(status="ok",label="drive",confidence=.9,truncated_frames=0)), \
                 patch("desktop.pipeline.paddle_analysis", return_value=dict(status="ok",state="OPTIMAL")), \
                 patch("desktop.pipeline.encode_video"), \
                 patch("trim_after_contact.estimate_contact_frame", return_value=55), \
                 patch("balltrack_pipeline.process_video", return_value=dict(landing=None,court_rate=0)) as process:
                analyze(Path("primary.mp4"),Path("secondary.mp4"),Path(tmp),dict(start=0,end=3,model="single",moment_matching=True,landing=mode),lambda *a:None,threading.Event())
                self.assertEqual(expected,process.call_args.kwargs["min_bounce_frame"])

    def test_padding_missing_frames_and_long_clip_truncation(self):
        config = json.loads((ASSETS / "manifest.json").read_text())
        points = np.random.default_rng(8).random((150, 33, 2)).astype(np.float32)
        points[5] = np.nan
        window, count = build_window(points, config)
        self.assertEqual((1, 128, 10), window.shape)
        self.assertEqual(127, count)
        np.testing.assert_array_equal(window[0, 5], 0)
        short, count = build_window(points[:20], config)
        np.testing.assert_array_equal(short[0, 20:], 0)
        self.assertTrue(np.isfinite(window).all())

    def test_pose_failure_retains_explicit_stream_statuses(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("desktop.pipeline.video_info", return_value=dict(fps=30,frames=90,duration=3,width=640,height=360)), \
             patch("desktop.pipeline.extract_pose", side_effect=RuntimeError("pose unavailable")), \
             patch("desktop.pipeline.encode_video"):
            r = analyze(Path("missing.mp4"), None, Path(tmp), dict(start=0,end=3,model="single",moment_matching=True,landing="off"), lambda *a: None, threading.Event())
            self.assertEqual("unavailable", r["serve"]["status"])
            self.assertEqual("unavailable", r["paddle"]["status"])
            self.assertEqual("INSUFFICIENT_DATA", r["feedback"]["status"])

    def test_cancel_is_not_converted_into_a_detection_failure(self):
        cancel = threading.Event(); cancel.set()
        with tempfile.TemporaryDirectory() as tmp, patch("desktop.pipeline.video_info", return_value=dict(fps=30,frames=90,duration=3)):
            with self.assertRaises(Cancelled):
                analyze(Path("missing.mp4"), None, Path(tmp), dict(start=0,end=3), lambda *a:None, cancel)


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output_patch = patch.object(server, "OUTPUTS", Path(self.directory.name)); self.output_patch.start()
        self.submit = patch.object(server.executor, "submit"); self.submit.start()
        server.jobs.clear()
        self.client = TestClient(server.app)

    def tearDown(self):
        self.client.close(); self.submit.stop(); self.output_patch.stop(); self.directory.cleanup(); server.jobs.clear()

    def test_upload_status_queue_and_cancel(self):
        response = self.client.post("/api/jobs", files={"video":("serve.mp4", b"video", "video/mp4")})
        self.assertEqual(202, response.status_code)
        job = response.json()["id"]
        self.assertEqual("queued", self.client.get(f"/api/jobs/{job}").json()["status"])
        self.assertEqual(409, self.client.post("/api/jobs", files={"video":("second.mp4",b"video")}).status_code)
        self.assertEqual(200, self.client.post(f"/api/jobs/{job}/cancel").status_code)
        self.assertTrue(server.jobs[job]["cancel"].is_set())

    def test_bad_files_ranges_and_missing_second_camera(self):
        for filename, data, options in [("bad.txt",b"bad",{}),("empty.mp4",b"",{}),("ok.mp4",b"x",{"start":"nan"}),
                                        ("ok.mp4",b"x",{"start":"2","end":"1"}),("ok.mp4",b"x",{"landing":"secondary"})]:
            response=self.client.post("/api/jobs",files={"video":(filename,data)},data=options)
            self.assertEqual(400,response.status_code,response.text)

    def test_local_origin_and_artifact_boundaries(self):
        self.assertEqual(403, self.client.post("/api/jobs",headers={"origin":"https://example.com"}).status_code)
        self.assertEqual(404, self.client.get("/api/jobs/not-a-job/files/report.json").status_code)
        self.assertEqual(404, self.client.get("/api/jobs/"+'0'*32+"/files/input.mp4").status_code)


if __name__ == "__main__":
    unittest.main()
