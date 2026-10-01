"""Evaluate all frozen candidates once; diagnostic labels never select a model."""
import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import csv
import cv2
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from extract_timing_features import timing_features
from serve_study import ROOT, DATA, OUT, PROTOCOL, CLASSES, training_data, skeleton_features, sha, write_json, metrics


def predict(model, x):
    return np.concatenate([model(x[i:i+32], training=False).numpy() for i in range(0, len(x), 32)])


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    frozen=json.loads((OUT/"frozen_selection.json").read_text())
    if frozen["protocol_sha256"]!=sha(DATA/"protocol.json"):
        raise ValueError("Protocol changed")
    for p,h in frozen["models"].items():
        if sha(OUT/p)!=h:
            raise ValueError(f"Frozen model changed: {p}")
    train=training_data()
    training_timing=np.array([timing_features(x.reshape(128,5,2)) for x in train["X"]])
    knn_model=KNeighborsClassifier(n_neighbors=5,weights="distance").fit(training_timing,train["y"])
    np.savez_compressed(OUT/"research_knn_bank.npz",X=training_timing,y=train["y"],clip_ids=train["clip_ids"])
    # Diagnostic arrays are opened only after checking every frozen candidate.
    test=np.load(ROOT/"data/serve_v2/diagnostic_test.npz")
    assert not set(train["clip_ids"]) & set(test["clip_ids"])
    assert not set(train["subjects"]) & set(test["subjects"])
    inventory=json.loads((ROOT/"data/serve_v2/dataset.json").read_text())
    rows={r["clip_id"]:r for r in inventory["clips"] if r["split"]=="diagnostic_test"}
    skeleton,motion=[],[]
    for cid in test["clip_ids"]:
        s=np.load(ROOT/"data/serve_v2/poses"/f"{cid}.npz")
        cap=cv2.VideoCapture(str(ROOT/rows[cid]["video"]))
        aspect=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/cap.get(cv2.CAP_PROP_FRAME_HEIGHT);cap.release()
        a,b=skeleton_features(s["points"],s["visibility"],s["timestamps"],aspect)
        skeleton.append(a);motion.append(b)
    features=dict(angles_control=test["X"],angles_thesis_aug=test["X"],skeleton=np.array(skeleton),skeleton_motion=np.array(motion))
    test_timing=np.array([timing_features(x.reshape(128,5,2)) for x in test["X"]])
    knn=knn_model.predict_proba(test_timing)
    quality=test["quality_accepted"]
    result=dict(selected_condition=frozen["selected_condition"],selection_file_sha256=sha(OUT/"frozen_selection.json"),
                research_knn_bank_sha256=sha(OUT/"research_knn_bank.npz"),evaluator_sha256=sha(__file__),
                test_data_sha256=sha(ROOT/"data/serve_v2/diagnostic_test.npz"),
                test_status="previously inspected diagnostic set; not final Coach C evaluation",
                knn=metrics(test["y"],knn,test["subjects"],quality),conditions={})
    records=[]
    for c in PROTOCOL["conditions"]:
        probabilities=[]
        for fold in range(1,6):
            model=tf.keras.models.load_model(OUT/c/"stratified"/f"fold{fold}"/"model.keras",compile=False)
            probabilities.append(predict(model,features[c]))
        p=np.mean(probabilities,axis=0)
        hybrid=(p+knn)*.5
        result["conditions"][c]=dict(gru=metrics(test["y"],p,test["subjects"],quality),
                                      hybrid=metrics(test["y"],hybrid,test["subjects"],quality))
        np.savez_compressed(OUT/c/"diagnostic_predictions.npz",probabilities=p,knn=knn,hybrid=hybrid,
                            clip_ids=test["clip_ids"],subjects=test["subjects"],y=test["y"],quality_accepted=quality)
        for mode,probs in [("gru",p),("hybrid",hybrid)]:
            for cid,y,prob,q in zip(test["clip_ids"],test["y"],probs,quality):
                records.append(dict(condition=c,mode=mode,clip_id=str(cid),truth=CLASSES[int(y)],
                                    predicted=CLASSES[int(prob.argmax())] if q else "unavailable",
                                    quality_accepted=bool(q),confidence=float(prob.max()),**dict(zip(CLASSES,map(float,prob)))))
        print(c,result["conditions"][c]["gru"]["accuracy"],flush=True)
    write_json(OUT/"diagnostic_evaluation.json",result)
    with (OUT/"diagnostic_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


if __name__=="__main__":
    main()
