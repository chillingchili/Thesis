"""Evaluate existing Android GRU + kNN5 assets; no fitting or tuning."""
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import tensorflow as tf
from train_gru import SequenceAugment
from extract_joint_angles import clip_angles
from desktop.pipeline import ASSETS, build_window
from desktop.hybrid import Knn5, timing_features, combine
from audit_pose_classifier import metrics


def main():
    bank=Knn5(ASSETS)
    source=np.load(ROOT/'features/timing_train.npz',allow_pickle=True)
    np.testing.assert_array_equal(bank.X,source['X'].astype(np.float32))
    np.testing.assert_array_equal(bank.y,[bank.classes.index(str(y)) for y in source['y']])
    with (ROOT/'data/training/keypoints_angles/manifest.csv').open(newline='') as f:
        train_ids={Path(r['clip_path'].replace('\\','/')).stem for r in csv.DictReader(f)}
    assert set(map(str,source['clip_ids'])).issubset(train_ids)
    manifest=json.loads((ASSETS/'manifest.json').read_text())
    datasets={split:np.load(ROOT/f'data/serve_v2/{split}.npz') for split in ['train','diagnostic_test']}
    assert set(map(str,source['clip_ids'])).isdisjoint(map(str,datasets['diagnostic_test']['clip_ids']))
    windows,knn,old_knn,records,fixtures=[],[],[],[],[]
    for split,data in datasets.items():
        for cid,subject,y in zip(data['clip_ids'],data['subjects'],data['y']):
            points=np.load(ROOT/'data/serve_v2/poses'/f'{cid}.npz')['points']
            points=points[np.isfinite(clip_angles(points)).all(axis=(1,2))]
            matched,n=build_window(points,manifest,True)
            raw,_=build_window(points,manifest,False)
            feat=timing_features(raw[0],n)
            p=bank.predict(feat)
            wrong=feat.copy()
            angles=np.arctan2(raw[0,:n,0::2],raw[0,:n,1::2])
            for j in range(5):
                wrong[j*6+2]=np.abs(np.diff(angles[:,j],prepend=np.float32(0))).argmax()/max(n-1,1)
            windows.append(matched[0]);knn.append(p);old_knn.append(bank.predict(wrong))
            records.append(dict(split=split,clip_id=str(cid),subject=str(subject),label=bank.classes[int(y)]))
            if str(cid) in ['Beginner2_Drive_001','Beginner2_Drive_002','Beginner3_Drive_001']:
                fixtures.append(dict(clip_id=str(cid),valid_frames=n,window=raw[0].reshape(-1).tolist(),
                                     features=feat.tolist(),knn_probabilities=p.tolist()))
    x=np.stack(windows)
    probabilities=[]
    for i in range(1,6):
        model=tf.keras.models.load_model(ROOT/f'models/gru_runs_angles_strat5/gru_fold{i}.keras',
                                        custom_objects={'SequenceAugment':SequenceAugment},compile=False)
        probabilities.append(np.concatenate([model(x[j:j+64],training=False).numpy() for j in range(0,len(x),64)]))
    gru=np.mean(probabilities,axis=0)
    modes=dict(gru_ensemble=gru,knn5=np.array(knn),hybrid=combine(gru,np.array(knn)),
               hybrid_old_android_velocity=combine(gru,np.array(old_knn)))
    results={}
    output=[]
    labels=np.array([bank.classes.index(r['label']) for r in records])
    for mode,p in modes.items():
        results[mode]={}
        for group in ['train','diagnostic_test','Beginner3','Beginner4']:
            mask=np.array([r['split']==group or r['subject']==group for r in records])
            results[mode][group]=metrics(labels[mask],p[mask])
        for row,prob in zip(records,p):
            output.append(dict(**row,model=mode,predicted=bank.classes[int(prob.argmax())],confidence=float(prob.max()),
                               **dict(zip(bank.classes,map(float,prob)))))
    out=ROOT/'outputs/hybrid_audit';out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(dict(results=results,knn_bank_sha256=bank.sha256,
              bank_matches_training_features=True,training_samples=len(bank.X),new_training_performed=False,
              test_used_for_fitting=False,protocol='first 128 valid frames; raw kNN features; matched GRU features; 50/50 probabilities'),indent=2))
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    fixture=ROOT/'android/app/src/test/resources/fixtures/hybrid.json'
    fixture.write_text(json.dumps(dict(knn_bank_sha256=bank.sha256,cases=fixtures),indent=2))
    for mode,result in results.items():
        print(mode,result['diagnostic_test'],flush=True)


if __name__=='__main__':
    main()
