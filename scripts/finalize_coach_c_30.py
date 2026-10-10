"""Complete the v2 30-clip freeze after user-supplied Coach A H264 replacements.

Preserves the 24 frozen beginner selections untouched. Verifies the six coach
packet files (OpenCV sequential decode, metadata-leak scan, training-hash
audit), then extends researcher_key.csv / technical_checks.csv to 30 rows,
fills coach_clip_slots.csv, marks selection.json complete, rebuilds the final
Session 1 zip and status.json. Refuses to run if any check fails.
"""
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "coach_c_evaluation_v2"
PACKET = V2 / "session1_staging"
VIDEOS = PACKET / "videos"
SUPPLIED = V2 / "supplied_originals"

COACH_SLOTS = {  # review_id -> recording_category, confirmed by user
    "C001": "lob", "C005": "lob", "C008": "drive",
    "C019": "topspin", "C022": "topspin", "C026": "drive",
}
WORDS = ["drive", "lob", "topspin", "coacha", "beginner"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def tech_check(path):
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        if not cap.isOpened() or count < 2 or fps <= 0 or n < 2:
            raise ValueError(f"Unreadable clip: {path}")
        if abs(n - count) > max(2, int(count * 0.01)):
            raise ValueError(f"Advertised/decoded mismatch {path}: {n}/{count}")
        return dict(frame_count=count, fps=fps, width=w, height=h,
                    duration_seconds=count / fps, decoded_frame_count=n,
                    sequential_decode_complete=True, full_human_quality_review=False)
    finally:
        cap.release()


def metadata_clean(path):
    tags = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries",
         "format_tags:stream_tags", "-of", "json", str(path)], text=True))
    for obj in [tags.get("format", {})] + tags.get("streams", []):
        for v in obj.get("tags", {}).values():
            if any(w in str(v).lower() for w in WORDS):
                raise ValueError(f"Identity/category leak in {path}: {v}")


def main():
    sel = json.loads((V2 / "selection.json").read_text())
    assert len(sel["beginner_selection"]) == 24, "beginner freeze changed"
    ds = json.loads((ROOT / "data/serve_v2/dataset.json").read_text())
    train_hashes = {r["video_sha256"] for r in ds["clips"] if r["split"] == "train"}
    diag_ids = {r["clip_id"] for r in ds["clips"] if r["split"] == "diagnostic_test"}
    pred_ids = set()
    for p in [ROOT / "outputs/pose_audit/cached_predictions.csv",
              ROOT / "outputs/pose_audit/fresh_predictions.csv",
              ROOT / "outputs/hybrid_audit/predictions.csv"]:
        if p.exists():
            pred_ids.update(r["clip_id"] for r in csv_read(p))
    knn_ids = set(map(str, np.load(ROOT / "features/timing_train.npz",
                                   allow_pickle=True)["clip_ids"]))
    with (ROOT / "data/training/keypoints_angles/manifest.csv").open(
            encoding="utf-8-sig", newline="") as f:
        gru_ids = {Path(r["clip_path"].replace("\\", "/")).stem
                   for r in csv.DictReader(f)}

    # Original AV1 provenance for the two Resolve re-exports.
    orig_av1 = {}
    for rid in ["C001", "C005"]:
        p = SUPPLIED / f"{rid}_original_av1.mp4"
        assert p.is_file(), f"missing AV1 original backup {p}"
        orig_av1[rid] = sha(p)

    coach_rows, check_rows, slot_rows = [], [], []
    for rid, cat in sorted(COACH_SLOTS.items()):
        p = VIDEOS / f"{rid}.mp4"
        assert p.is_file(), f"missing packet video {p}"
        h = sha(p)
        assert h not in train_hashes, f"{rid} collides with training bytes"
        assert rid not in gru_ids and rid not in knn_ids
        assert rid not in diag_ids and rid not in pred_ids
        m = tech_check(p)
        metadata_clean(p)
        rel = p.relative_to(ROOT).as_posix()
        coach_rows.append(dict(review_id=rid, clip_id=rid, participant="CoachA",
                               recording_category=cat, source_video=rel,
                               source_sha256=h, new_gru_training_overlap=False,
                               legacy_gru_training_overlap=False,
                               legacy_knn_training_overlap=False,
                               prior_diagnostic_overlap=False,
                               prior_model_prediction_exists=False))
        check_rows.append(dict(review_id=rid, **m,
                               metadata_identity_or_category_absent=True))
        note = ("H264 re-export of Resolve AV1 original "
                f"(av1 sha {orig_av1[rid][:12]}… backed up)"
                if rid in orig_av1 else "user-supplied H264, verified fresh")
        slot_rows.append(dict(review_id=rid, participant="CoachA",
                              recording_category=cat, source_video=rel,
                              source_sha256=h,
                              renamed_packet_filename=f"{rid}.mp4",
                              training_overlap_checked="True", notes=note))
        print("VERIFIED", rid, cat, m["frame_count"], "frames", h[:12], flush=True)

    key = csv_read(V2 / "researcher_key.csv")
    assert len(key) == 24, "researcher key changed under us"
    key = sorted(key + coach_rows, key=lambda r: r["review_id"])
    assert [r["review_id"] for r in key] == [f"C{i:03d}" for i in range(1, 31)]
    csv_write(V2 / "researcher_key.csv", key)

    checks = csv_read(V2 / "technical_checks.csv")
    assert len(checks) == 24
    csv_write(V2 / "technical_checks.csv",
              sorted(checks + check_rows, key=lambda r: r["review_id"]))

    csv_write(V2 / "coach_clip_slots.csv", sorted(
        slot_rows, key=lambda r: r["review_id"]))

    sel["completed_coach_selection"] = [
        dict(review_id=r["review_id"], subject="CoachA",
             recording_category=r["recording_category"],
             packet_video=r["source_video"], video_sha256=r["source_sha256"],
             new_gru_training_overlap=False, legacy_gru_training_overlap=False,
             legacy_knn_training_overlap=False, prior_diagnostic_overlap=False,
             prior_model_prediction_exists=False,
             original_av1_sha256=orig_av1.get(r["review_id"]))
        for r in coach_rows]
    sel["completed_30"] = True
    sel["completion_note"] = (
        "Six user-supplied Coach A clips verified fresh (no training-hash, "
        "legacy, diagnostic or prior-prediction overlap) and frozen. C001/C005 "
        "are H264 re-exports of Resolve AV1 originals whose hashes are kept "
        "in supplied_originals/ and recorded per clip. 24 beginner rows "
        "untouched from the first random draw.")
    (V2 / "selection.json").write_text(json.dumps(sel, indent=2))

    # Annotations sheet must already cover C001-C030 and stay blank.
    ann = csv_read(PACKET / "annotations.csv")
    assert [r["review_id"] for r in ann] == [f"C{i:03d}" for i in range(1, 31)]
    assert all(not r["serve_type"] and not r["form_judgment"] for r in ann), \
        "annotations already filled - refusing to rebuild packet over responses"

    zip_path = V2 / "coach_c_session1.zip"
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_STORED) as z:
        for path in sorted(PACKET.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(PACKET).as_posix())
    with ZipFile(zip_path) as z:
        names = set(z.namelist())
        assert {n for n in names if n.endswith(".mp4")} == \
            {f"videos/C{i:03d}.mp4" for i in range(1, 31)}, names
        assert "researcher_key.csv" not in names
        assert "coach_clip_slots.csv" not in names
        assert z.testzip() is None

    cats = Counter(r["recording_category"] for r in key)
    parts = Counter(r["participant"] for r in key)
    report = dict(selected_beginner_clips=24, pending_coach_clips=0,
                  selected_clips=30, complete_30_video_packet=True,
                  participant_counts=dict(parts),
                  recording_category_counts=dict(cats),
                  new_gru_training_overlap=0, legacy_gru_training_overlap=0,
                  legacy_knn_training_overlap=0,
                  prior_diagnostic_overlap=sum(
                      1 for r in key if str(r["prior_diagnostic_overlap"]) == "True"),
                  seed=sel["seed"], selection_sha256=sha(V2 / "selection.json"),
                  session1_zip_sha256=sha(zip_path),
                  blind_packet_verified=True, model_outputs_generated=False,
                  coach_c_annotations_pending=True,
                  final_claim=("complete 30-clip retrospective independent coach "
                               "review; 24 beginner clips had prior diagnostic "
                               "exposure, 6 coach clips verified fresh"))
    (V2 / "status.json").write_text(json.dumps(report, indent=2))
    (V2 / "RESEARCHER_README.md").write_text(
        """# Coach C 30-clip evaluation set (complete)

30 videos frozen: 24 beginner clips preserved from the first random draw
(seed 20261001; 12 Beginner 3 + 12 Beginner 4; 4 per recording category each)
plus 6 user-supplied Coach A clips (2 per recording category: C001/C005 lob,
C008/C026 drive, C019/C022 topspin).

Intended quota is 10 per recording category (4 Beginner 3 + 4 Beginner 4 + 2
Coach A). Coach C's independent labels may change the observed class balance.

Verification per coach clip: OpenCV sequential decode of every frame,
container-metadata scan (no serve/identity strings), SHA-256 audit against
training bytes (210), legacy GRU/kNN banks, diagnostic IDs and saved prior
predictions. C001/C005 arrived as Resolve AV1 exports undecodable by the
pipeline; the packet holds their H264 re-exports while the AV1 originals are
kept in supplied_originals/ with hashes in selection.json.

Share only `coach_c_session1.zip` for Session 1 (30 anonymized videos, blank
30-row annotations.csv, instructions). Keep researcher_key.csv,
selection.json, technical_checks.csv, coach_clip_slots.csv with researchers.

This is a retrospective independent coach review, not a sealed pre-development
test: all 24 beginner clips appeared in earlier diagnostics. Do not tune
models or thresholds on Coach C's judgments. Run Session 1 annotation before
any frozen-pipeline inference; Session 2 shows video + feedback string only.
""", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
