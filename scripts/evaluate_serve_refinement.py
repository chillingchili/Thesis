"""Diagnostic evaluation and isolated Android research bundle after selection freezes."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import csv
import json
import struct
import numpy as np
import tensorflow as tf
import cv2
from sklearn.neighbors import KNeighborsClassifier
from extract_timing_features import timing_features
from serve_study import ROOT, DATA, CLASSES, training_data, skeleton_features, sha, write_json
from serve_refinement import DEST, OUTPUT, CONDITIONS, features, apply_policy, probability_metrics
from train_serve_study import predict
from benchmark_serve_hybrid import padded, moment_match, interpreters, infer
from extract_joint_angles import clip_angles
from desktop.hybrid import Knn5, timing_features as legacy_timing


def export_bundle(selected, model, policy, timing, train):
    folder=DEST/'android_bundle';folder.mkdir(parents=True,exist_ok=True)
    converter=tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops=False
    (folder/'model.tflite').write_bytes(converter.convert())
    bank=struct.pack('<ii',len(timing),timing.shape[1])+timing.astype('<f4').tobytes()+train['y'].astype('<i4').tobytes()
    (folder/'knn_train.bin').write_bytes(bank)
    write_json(folder/'knn_meta.json',dict(n_samples=len(timing),n_features=32,k=5,classes=CLASSES))
    manifest=dict(version='serve_refinement_v4',condition=selected,classes=CLASSES,policy=policy,
        seq_len=128,n_features=int(model.input_shape[-1]),moment_matching=False,
        window='complete_selected_serve',pose_mode='VIDEO',visibility_min=.5,min_source_frames=32,
        min_valid_fraction=.8,max_seconds=12,pose_sha256=sha(ROOT/'android/app/src/main/assets/pose_landmarker_lite.task'),
        model_sha256=sha(folder/'model.tflite'),knn_sha256=sha(folder/'knn_train.bin'),
        selection_sha256=sha(DEST/'frozen_selection.json'),status='research replay only; default live assets unchanged')
    write_json(folder/'manifest.json',manifest)
    original=np.load(DATA/'skeleton_train.npz')
    bank_model=KNeighborsClassifier(5,weights='distance').fit(timing,train['y'])
    cases=[]
    for player in sorted(set(train['subjects'])):
        i=int(np.flatnonzero(train['subjects']==player)[0])
        x=features(selected,train['X'][i],original['skeleton'][i],original['skeleton_motion'][i])
        gp=model(x[None],training=False).numpy();kp=bank_model.predict_proba(timing[i:i+1])
        cases.append(dict(clip_id=str(train['clip_ids'][i]),input=x.reshape(-1).tolist(),timing=timing[i].tolist(),
            gru=gp[0].tolist(),knn=kp[0].tolist(),calibrated=apply_policy(gp,kp,policy)[0].tolist()))
    write_json(folder/'native_parity_cases.json',dict(condition=selected,n_features=int(model.input_shape[-1]),
        purpose='training examples for numerical portability only, not accuracy or calibration fitting',cases=cases))
    return folder


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    frozen=json.loads((DEST/'frozen_selection.json').read_text())
    assert frozen['protocol_sha256']==sha(DEST/'protocol.json')
    for name,h in frozen['models'].items(): assert sha(DEST/name)==h
    train=training_data()
    timing=np.array([timing_features(a.reshape(128,5,2)) for a in train['X']])
    bank=KNeighborsClassifier(5,weights='distance').fit(timing,train['y'])
    selected=frozen['selected_condition']
    selected_model=tf.keras.models.load_model(DEST/selected/'full_train/model.keras',compile=False)
    bundle=export_bundle(selected,selected_model,frozen['policies'][selected],timing,train)
    # No diagnostic labels are opened before the above training-only selection/export.
    test=np.load(ROOT/'data/serve_v2/diagnostic_test.npz')
    assert not set(test['clip_ids']) & set(train['clip_ids'])
    inventory=json.loads((ROOT/'data/serve_v2/dataset.json').read_text())
    rows={r['clip_id']:r for r in inventory['clips']}
    skeleton=[];motion=[];raw=[];cache=[]
    for cid in test['clip_ids']:
        z=np.load(ROOT/'data/serve_v2/poses'/f'{cid}.npz')
        cap=cv2.VideoCapture(str(ROOT/rows[cid]['video']));aspect=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/cap.get(cv2.CAP_PROP_FRAME_HEIGHT);cap.release()
        s,m=skeleton_features(z['points'],z['visibility'],z['timestamps'],aspect)
        skeleton.append(s);motion.append(m)
        a=clip_angles(z['points']).reshape(-1,10);raw.append(a[np.isfinite(a).all(1)])
        cache.append(z)
    skeleton=np.array(skeleton);motion=np.array(motion)
    kp=bank.predict_proba(np.array([timing_features(a.reshape(128,5,2)) for a in test['X']]))
    results={};records=[]
    for c in CONDITIONS:
        x=features(c,test['X'],skeleton,motion)
        model=selected_model if c==selected else tf.keras.models.load_model(DEST/c/'full_train/model.keras',compile=False)
        gp=predict(model,x);policy=frozen['policies'][c]
        modes=dict(gru=gp,fixed_hybrid=(gp+kp)*.5,tuned_calibrated=apply_policy(gp,kp,policy))
        results[c]={name:probability_metrics(test['y'],p,test['subjects'],test['quality_accepted'],
                     policy['confidence_threshold'] if name=='tuned_calibrated' else .6) for name,p in modes.items()}
        np.savez_compressed(DEST/c/'diagnostic_predictions.npz',gru=gp,knn=kp,tuned=modes['tuned_calibrated'],
                            y=test['y'],clip_ids=test['clip_ids'],quality_accepted=test['quality_accepted'])
        for name,p in modes.items():
            for cid,y,prob,q in zip(test['clip_ids'],test['y'],p,test['quality_accepted']):
                records.append(dict(condition=c,mode=name,clip_id=str(cid),truth=CLASSES[int(y)],
                    prediction=CLASSES[int(prob.argmax())] if q else 'unavailable',confidence=float(prob.max()),available=bool(q)))
        print('DIAGNOSTIC',c,{k:round(v['accuracy'],4) for k,v in results[c].items()},flush=True)
        if c==selected:
            lite=tf.lite.Interpreter(model_path=str(bundle/'model.tflite'),num_threads=2);lite.allocate_tensors()
            lp=[]
            for w in x:
                lite.set_tensor(lite.get_input_details()[0]['index'],w[None].astype(np.float32));lite.invoke()
                lp.append(lite.get_tensor(lite.get_output_details()[0]['index'])[0].copy())
            lp=np.array(lp);np.testing.assert_allclose(lp,gp,atol=2e-5,rtol=2e-5)
            write_json(bundle/'parity.json',dict(cases=len(x),max_probability_difference=float(np.max(np.abs(lp-gp))),passed=True))
    # Quantify current assets' window mismatch without fitting them or their weights.
    assets=ROOT/'android/app/src/main/assets';manifest=json.loads((assets/'manifest.json').read_text())
    models=interpreters(manifest);old_bank=Knn5(assets);capture={}
    for mode in ['first128','last128','full_timeline']:
        gs=[];ks=[];available=[];loss=[]
        for i,a in enumerate(raw):
            if mode=='full_timeline': w=test['X'][i];n=128;ok=bool(test['quality_accepted'][i])
            else: w,n=padded(a[:128] if mode=='first128' else a);ok=len(a)>=32
            gs.append(infer(models,moment_match(w,n,manifest)) if ok else np.zeros(3))
            ks.append(old_bank.predict(legacy_timing(w,n)) if ok else np.zeros(3));available.append(ok)
            loss.append(max(0,len(a)-128) if mode!='full_timeline' else 0)
        gs=np.array(gs);ks=np.array(ks)
        capture[mode]=dict(gru=probability_metrics(test['y'],gs,test['subjects'],available,.6),
            hybrid=probability_metrics(test['y'],(gs+ks)*.5,test['subjects'],available,.6),
            clips_losing_frames=int(np.sum(np.array(loss)>0)),discarded_pose_frames=int(sum(loss)))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_json(OUTPUT/'diagnostic_results.json',dict(selected_condition=selected,conditions=results,capture=capture,
        selection_sha256=sha(DEST/'frozen_selection.json'),diagnostic_used_for_selection=False,
        physical_device_connected=False,test_status='previously inspected diagnostic clips',
        final_models='single full-training fits; not v3 five-fold ensembles'))
    with (OUTPUT/'diagnostic_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print('EVALUATION AND EXPORT COMPLETE',flush=True)


if __name__=='__main__':main()
