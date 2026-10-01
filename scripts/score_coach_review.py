"""Validate returned human review CSV and summarize agreement without relabeling."""
import argparse
import csv
from pathlib import Path
from collections import Counter
from sklearn.metrics import cohen_kappa_score
from prepare_coach_review import DEST
from serve_study import CLASSES, write_json, sha


def score(path):
    with (DEST/"researcher_key.csv").open(encoding="utf-8-sig",newline="") as f:
        key={r["review_id"]:r for r in csv.DictReader(f)}
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    seen=set(); completed=[]; disagreements=[]; pose=Counter(); ambiguous=0
    for r in rows:
        rid=r["review_id"]
        if rid not in key or rid in seen:
            raise ValueError(f"Unknown or duplicate review ID: {rid}")
        seen.add(rid)
        observed=r["observed_subtype"].strip().lower()
        if observed not in CLASSES+["ambiguous","unjudgeable",""]:
            raise ValueError(f"Invalid subtype: {rid}")
        for field,choices in [("subtype_certainty",["high","medium","low"]),
                              ("pose_fidelity",["usable","partly_wrong","unusable","unsure"]),
                              *[(n,["yes","no","unsure"]) for n in ["subtype_visible_from_video","wrong_person","wrist_or_hand_tracking_error","occlusion_or_blur"]]]:
            r[field]=r[field].strip().lower()
            if r[field] not in choices+[""]:
                raise ValueError(f"Invalid {field}: {rid}")
        if observed or r["pose_fidelity"]:
            if not r["reviewer"].strip() or not r["review_date"].strip():
                raise ValueError(f"Reviewer and date required: {rid}")
        if observed in CLASSES:
            completed.append((key[rid]["original_label"],observed))
            if observed!=key[rid]["original_label"]:
                disagreements.append(dict(**key[rid],coach_observed=observed))
        elif observed:
            ambiguous+=1
        if r["pose_fidelity"]:
            pose[r["pose_fidelity"]]+=1
    agreement=sum(a==b for a,b in completed)/len(completed) if completed else None
    kappa=float(cohen_kappa_score(*zip(*completed),labels=CLASSES)) if len({a for a,b in completed}|{b for a,b in completed})>1 else None
    report=dict(response_sha256=sha(path),review_rows=len(rows),expected_cases=len(key),
                completed_subtype_judgments=len(completed),ambiguous_or_unjudgeable=ambiguous,
                pending_subtype_judgments=len(key)-len(completed)-ambiguous,
                sample_agreement=agreement,sample_kappa=kappa,pose_judgments=dict(pose),
                disagreements=disagreements,labels_changed=False,
                status="pending" if len(completed)+ambiguous<len(key) else "subtype review complete; adjudication remains manual",
                limitation="selected diagnostic sample; not final evaluation or dataset-wide agreement")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses",type=Path)
    args=parser.parse_args()
    result=score(args.responses)
    write_json(DEST/"returned_review_summary.json",result)
    print(result)
