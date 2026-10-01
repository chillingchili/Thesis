"""Quantify augmented pose changes and render reproducible visual spot checks."""
import csv
import json
import cv2
import numpy as np
from prepare_serve_study import augment_frame
from prepare_coach_review import CONNECTIONS
from serve_study import ROOT, DATA, training_data, training_rows, write_json


def preview(row, target):
    cid=row["clip_id"]
    original=np.load(ROOT/"data/serve_v2/poses"/f"{cid}.npz")
    augmented=np.load(DATA/"augmented"/f"{cid}__a1.npz")
    params=json.loads(str(augmented["provenance"]))["parameters"]
    # Exact pixel frame from the generated variant, including its noise seed.
    index=len(augmented["points"])//2
    source_index=int(augmented["source_indices"][index])
    cap=cv2.VideoCapture(str(ROOT/row["video"]))
    for i in range(source_index+1):
        ok,frame=cap.read()
        if not ok:raise ValueError("Cannot decode preview")
    cap.release()
    rng=np.random.default_rng(params["seed"]+1)
    rng.integers(1,2147483647,size=index)
    transformed=augment_frame(frame,params,rng)
    tiles=[]
    for title,img,points,vis in [("Original",frame,original["points"][source_index],original["visibility"][source_index]),
                                ("Thesis video augmentation",transformed,augmented["points"][index],augmented["visibility"][index])]:
        h,w=img.shape[:2];tw=700;th=round(h*tw/w)
        raw=cv2.resize(img,(tw,th)); overlay=raw.copy();xy=points*[tw,th]
        for a,b in CONNECTIONS:
            if np.isfinite(xy[[a,b]]).all():
                color=(80,220,40) if min(vis[a],vis[b])>=.5 else (25,60,230)
                cv2.line(overlay,tuple(np.rint(xy[a]).astype(int)),tuple(np.rint(xy[b]).astype(int)),color,2,cv2.LINE_AA)
        band=np.full((55,tw,3),245,np.uint8)
        cv2.putText(band,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),1,cv2.LINE_AA)
        cv2.putText(band,f"{cid} | source frame {source_index}",(12,45),cv2.FONT_HERSHEY_SIMPLEX,.45,(25,25,25),1,cv2.LINE_AA)
        tiles.append(np.vstack([band,raw,overlay]))
    target.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(target),np.hstack(tiles))


def main():
    d=training_data(); rows=[]
    for i,cid in enumerate(d["clip_ids"]):
        for v in (1,2):
            path=DATA/"augmented"/f"{cid}__a{v}.npz"
            if not path.exists():continue
            a=np.load(path);x=a["X"];base=d["X"][i]
            valid=np.any(x!=0,1)&np.any(base!=0,1)
            delta=np.arctan2(x[:,0::2],x[:,1::2])-np.arctan2(base[:,0::2],base[:,1::2])
            delta=np.abs((delta+np.pi)%(2*np.pi)-np.pi)*180/np.pi
            q=json.loads(str(a["quality"]))
            rows.append(dict(clip_id=str(cid),variant=v,quality_accepted=q["quality_accepted"],
                             common_valid_frames=int(valid.sum()),
                             mean_angle_change_degrees=float(delta[valid].mean()) if valid.any() else None,
                             p95_angle_change_degrees=float(np.percentile(delta[valid],95)) if valid.any() else None,
                             identical_to_original=bool(np.array_equal(x,base))))
    dest=ROOT/"outputs/serve_study_v3"
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/"augmentation_effects.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    changes=[r["mean_angle_change_degrees"] for r in rows if r["mean_angle_change_degrees"] is not None]
    summary=dict(extracted_variants=len(rows),expected_variants=418,accepted=sum(r["quality_accepted"] for r in rows),
                 identical_to_original=sum(r["identical_to_original"] for r in rows),
                 median_mean_angle_change_degrees=float(np.median(changes)),
                 note="changes reflect pose-estimation and temporal-resampling sensitivity, not independent new serve mechanics")
    write_json(dest/"augmentation_effects.json",summary)
    for cid in ["CoachA_Drive_001","Beginner1_Drive_001","Beginner2_Drive_001"]:
        if (DATA/"augmented"/f"{cid}__a1.npz").exists():
            row=next(r for r in training_rows() if r["clip_id"]==cid)
            preview(row,dest/"previews"/f"{cid}.jpg")
    print(summary)


if __name__=="__main__":main()
