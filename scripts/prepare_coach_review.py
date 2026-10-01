"""Training-only diagnostic audit and blinded human review packet.

Automated flags select cases; they are NOT proof of a wrong label or pose.
Source labels remain frozen. Never send this packet automatically.
"""
import csv
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import cv2
import numpy as np
from coach_annotations import read_annotations, WORKBOOK
from serve_study import ROOT, SEED, CLASSES, sha, write_json

DEST = ROOT / "outputs/serve_study_v3/coach_review"
CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(16,20),(11,23),(12,24),
               (23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32)]


def csv_write(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns or list(rows[0])); w.writeheader(); w.writerows(rows)


def pose_flags(saved):
    p, vis = saved["points"], saved["visibility"]
    required = [11, 12, 14, 16, 20, 23, 24]
    valid = np.isfinite(p).all(2) & (vis >= .5)
    torso = np.linalg.norm((p[:,11]+p[:,12]-p[:,23]-p[:,24])*.5, axis=1)
    scale = np.nanmedian(torso[valid[:,[11,12,23,24]].all(1)])
    delta = np.linalg.norm(np.diff(p[:,16], axis=0), axis=1) / max(float(scale), 1e-6)
    adjacent = valid[1:,16] & valid[:-1,16]
    jump = float(np.max(delta[adjacent])) if adjacent.any() else None
    return dict(required_low_visibility_fraction=float(1-valid[:,required].all(1).mean()),
                wrist_low_visibility_fraction=float(1-valid[:,16].mean()),
                max_adjacent_wrist_displacement_torso_units=jump)


def strip(video, saved, output, review_id):
    p, v = saved["points"], saved["visibility"]
    indices = np.linspace(0, len(p)-1, 6).round().astype(int)
    cap = cv2.VideoCapture(str(video))
    tiles = []
    decoded = {}
    for i in range(int(indices[-1]) + 1):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode review frame {video}:{i}")
        if i in indices:
            decoded[i] = frame
    for index in indices:
        frame = decoded[int(index)]
        h,w = frame.shape[:2]
        tw, th = 480, round(h*480/w)
        raw = cv2.resize(frame,(tw,th))
        overlay = raw.copy()
        xy = p[index] * [tw,th]
        for a,b in CONNECTIONS:
            if not np.isfinite(xy[[a,b]]).all():
                continue
            color = (90,220,30) if min(v[index,a],v[index,b]) >= .5 else (35,70,240)
            cv2.line(overlay,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),color,2,cv2.LINE_AA)
        for j in [11,12,14,16,20,23,24]:
            if np.isfinite(xy[j]).all():
                point = tuple(np.rint(xy[j]).astype(int))
                cv2.circle(overlay,point,3,(0,220,255),-1)
                cv2.putText(overlay,str(j),point,cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1,cv2.LINE_AA)
        band = np.full((30,tw,3),245,np.uint8)
        cv2.putText(band,f"{review_id} | frame {index} | {float(saved['timestamps'][index]):.2f}s",(8,21),cv2.FONT_HERSHEY_SIMPLEX,.47,(35,35,35),1,cv2.LINE_AA)
        tiles.append(np.vstack([band,raw,overlay]))
    cap.release()
    canvas = np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])])
    output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(output),canvas)


def main():
    annotations = {Path(a["Filename"]).stem:a for a in read_annotations()}
    frozen = json.loads((ROOT/"data/serve_v2/dataset.json").read_text())
    rows = [r for r in frozen["clips"] if r["split"] == "train"]
    quality = {r["clip_id"]:r for r in json.loads((ROOT/"data/serve_v2/train_quality.json").read_text())}
    oof = np.load(ROOT/"models/serve_v2_fixed/participant_oof.npz")
    probabilities = dict(zip(oof["clip_ids"],oof["probabilities"]))
    inventory = []
    for row in rows:
        cid = row["clip_id"]
        s = np.load(ROOT/"data/serve_v2/poses"/f"{cid}.npz")
        p = probabilities.get(cid)
        predicted = CLASSES[int(p.argmax())] if p is not None else "unavailable"
        a = annotations[cid]
        inventory.append(dict(clip_id=cid,subject=row["subject"],original_label=row["label"],
             workbook_label=a["Serve Type"].lower(),label_files_agree=a["Serve Type"].lower()==row["label"],
             quality_accepted=quality[cid]["quality_accepted"],oof_prediction=predicted,
             oof_confidence=float(p.max()) if p is not None else None,
             oof_wrong=predicted!=row["label"],**pose_flags(s),
             form=a["Form Judgment"],coach_comment=a.get("Deviation Comment","")))
    # Predeclared balanced diagnostic sampling: two confident OOF errors plus
    # one seeded random remaining clip in each player x subtype cell.
    rng = np.random.default_rng(SEED)
    selected, reasons = set(), {}
    for subject in sorted({r["subject"] for r in inventory}):
        for label in CLASSES:
            cell = [r for r in inventory if r["subject"]==subject and r["original_label"]==label]
            errors = sorted([r for r in cell if r["oof_wrong"] and r["oof_confidence"] is not None],
                            key=lambda r:(-r["oof_confidence"],r["clip_id"]))[:2]
            for r in errors:
                selected.add(r["clip_id"]); reasons[r["clip_id"]]="confident participant-held-out error"
            remaining = [r for r in cell if r["clip_id"] not in selected]
            for index in rng.choice(len(remaining),size=3-len(errors),replace=False):
                r=remaining[index]; selected.add(r["clip_id"]); reasons[r["clip_id"]]="seeded comparison sample"
    for cid in ["CoachA_Drive_018","Beginner2_Drive_001","Beginner2_Drive_002"]:
        if cid not in selected:
            selected.add(cid); reasons[cid]="tracking rejection or previously reported training example"
    order = sorted(selected); rng.shuffle(order)
    by_id = {r["clip_id"]:r for r in rows}
    packet = DEST/"packet"
    researcher, responses = [], []
    for i,cid in enumerate(order,1):
        rid=f"R{i:03d}"
        row=by_id[cid]
        source=ROOT/row["video"]
        if sha(source)!=row["video_sha256"]:
            raise ValueError("Frozen video changed")
        video=packet/"01_raw_video"/f"{rid}.mp4"
        video.parent.mkdir(parents=True,exist_ok=True)
        if not video.exists():
            shutil.copyfile(source,video)
        strip(source,np.load(ROOT/"data/serve_v2/poses"/f"{cid}.npz"),packet/"02_pose_review"/f"{rid}.jpg",rid)
        researcher.append(dict(review_id=rid,clip_id=cid,selection_reason=reasons[cid],original_label=row["label"],
                               source_sha256=row["video_sha256"]))
        responses.append(dict(review_id=rid,reviewer="",review_date="",observed_subtype="",subtype_certainty="",
                              subtype_visible_from_video="",label_notes="",pose_fidelity="",wrong_person="",
                              wrist_or_hand_tracking_error="",occlusion_or_blur="",pose_notes=""))
    csv_write(DEST/"researcher_key.csv",researcher)
    csv_write(DEST/"training_audit.csv",inventory)
    # Never overwrite a sheet containing human work on rerun.
    if not (packet/"responses.csv").exists():
        csv_write(packet/"responses.csv",responses)
    instructions = """# Independent training-clip review

This is a diagnostic training-data audit, NOT the thesis's independent final
evaluation set. Review IDs intentionally hide original labels and predictions.
The packet contains selected difficult examples and comparison clips; agreement
rates must not be generalized to the entire dataset.

1. Open each video in `01_raw_video`. Complete observed_subtype (drive, lob,
   topspin, ambiguous, or unjudgeable), subtype_certainty (high, medium, low),
   subtype_visible_from_video (yes, no, unsure), and label_notes in responses.csv.
   Judge the visible executed serve, not the intended drill. Do not guess spin
   if the video does not show evidence sufficient for your judgment.
2. Save those judgments BEFORE opening `02_pose_review`. The upper image in
   each pair is raw video; the lower image shows the extracted skeleton. Green
   edges have reported visibility >=0.5; red edges have lower visibility.
   Visibility is model output, not proof of anatomical accuracy. Six frames are
   sampled across each clip, not manually annotated impact frames.
3. Complete pose_fidelity (usable, partly_wrong, unusable, unsure), wrong_person,
   wrist_or_hand_tracking_error, and occlusion_or_blur (yes, no, unsure).
   Add affected timestamps and specific anatomical errors in pose_notes.
4. Fill reviewer and review_date. Return responses.csv to the researcher.
   All labels remain unchanged until discrepancies are explicitly adjudicated.

You may leave an item blank if it has not been reviewed. Ambiguous and unjudgeable
are valid answers. No model prediction or original label is included in this packet.
"""
    (packet/"README.md").write_text(instructions,encoding="utf-8")
    summary=dict(training_clips=len(inventory),label_file_disagreements=sum(not r["label_files_agree"] for r in inventory),
                 quality_rejected=sum(not r["quality_accepted"] for r in inventory),review_cases=len(order),
                 human_review_status="pending coach responses",labels_changed=False,
                 annotation_sha256=sha(WORKBOOK),selection="2 most confident OOF errors + 1 random comparison per player/class; plus named regression/quality examples",
                 limitations="flags do not establish annotation or pose correctness; biased review sample cannot estimate dataset-wide error rate")
    write_json(DEST/"status.json",summary)
    with ZipFile(DEST/"coach_review_packet.zip","w",ZIP_STORED) as archive:
        for p in sorted(packet.rglob("*")):
            if p.is_file():
                archive.write(p,p.relative_to(packet).as_posix())
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
