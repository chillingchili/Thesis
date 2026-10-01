"""Nested participant validation for augmented geometry and calibrated fusion."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from serve_study import ROOT, DATA, OUT, SEED, training_data, sha, write_json
from train_serve_study import build_model, predict, arrays
from extract_timing_features import timing_features
from serve_refinement import DEST, CONDITIONS, PROTOCOL, freeze, features, fit_policy, apply_policy, probability_metrics


def prepare(d):
    a, av=arrays('angles_thesis_aug',d)
    s=np.load(DATA/'skeleton_train.npz')
    assert np.array_equal(s['clip_ids'],d['clip_ids'])
    sv=np.repeat(s['skeleton'][:,None],3,axis=1)
    mv=np.repeat(s['skeleton_motion'][:,None],3,axis=1)
    # Derive augmented skeletons from the same cached landmarks and frozen aspect.
    from serve_study import skeleton_features
    inventory=json.loads((ROOT/'data/serve_v2/dataset.json').read_text())
    import cv2
    rows={r['clip_id']:r for r in inventory['clips']}
    for i,cid in enumerate(d['clip_ids']):
        cap=cv2.VideoCapture(str(ROOT/rows[cid]['video']))
        aspect=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/cap.get(cv2.CAP_PROP_FRAME_HEIGHT);cap.release()
        for v in (1,2):
            z=np.load(DATA/'augmented'/f'{cid}__a{v}.npz')
            if json.loads(str(z['quality']))['quality_accepted']:
                sv[i,v],mv[i,v]=skeleton_features(z['points'],z['visibility'],z['timestamps'],aspect)
    return {c:(features(c,a,s['skeleton'],s['skeleton_motion']),
               None if c=='lean_control' else features(c,av,sv,mv)) for c in CONDITIONS}


def fit(c, tag, tr, va, x, variants, d, seed):
    folder=DEST/c/tag; record=folder/'result.json'
    config=dict(condition=c,train_ids=d['clip_ids'][tr].tolist(),validation_ids=d['clip_ids'][va].tolist(),
        train_subjects=sorted(set(d['subjects'][tr])),validation_subjects=sorted(set(d['subjects'][va])),
        seed=seed,protocol_sha256=sha(DEST/'protocol.json'),trainer_sha256=sha(__file__),
        feature_sha256=__import__('hashlib').sha256(x.tobytes()).hexdigest(),
        variants_sha256=__import__('hashlib').sha256(variants.tobytes()).hexdigest() if variants is not None else None)
    assert not set(config['train_ids']) & set(config['validation_ids'])
    assert not set(config['train_subjects']) & set(config['validation_subjects'])
    if record.exists():
        r=json.loads(record.read_text());assert r['configuration']==config
        assert sha(folder/'model.keras')==r['model_sha256']
        return np.load(folder/'predictions.npy'),r
    model=build_model(x.shape[-1],seed)
    order_rng=np.random.default_rng(seed); aug_rng=np.random.default_rng(seed+10000)
    start=time.monotonic();history=[]
    for epoch in range(PROTOCOL['epochs']):
        order=order_rng.permutation(tr)
        choices=aug_rng.integers(0,3,len(order)) if variants is not None else np.zeros(len(order),int)
        model.reset_metrics()
        for i in range(0,len(order),16):
            ix=order[i:i+16];batch=x[ix] if variants is None else variants[ix,choices[i:i+len(ix)]]
            log=model.train_on_batch(batch,d['y'][ix],return_dict=True)
        history.append({k:float(v) for k,v in log.items()})
    p=predict(model,x[va]) if len(va) else np.zeros((0,3))
    folder.mkdir(parents=True,exist_ok=True);model.save(folder/'model.keras');np.save(folder/'predictions.npy',p)
    r=dict(configuration=config,model_sha256=sha(folder/'model.keras'),parameters=model.count_params(),
        elapsed_seconds=time.monotonic()-start,history=history,outer_validation_passed_to_fit=False)
    write_json(record,r);print('FIT',c,tag,'seconds',round(r['elapsed_seconds']),flush=True)
    return p,r


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism();freeze()
    d=training_data();prepared=prepare(d);subjects=sorted(set(d['subjects']))
    timing=np.array([timing_features(a.reshape(128,5,2)) for a in d['X']])
    y=d['y']; reports={}
    for c,(x,variants) in prepared.items():
        gp=np.zeros((len(y),3));kp=np.zeros_like(gp); tuned=np.zeros_like(gp); accepted=np.zeros(len(y),bool);folds=[]
        for fold,held in enumerate(subjects,1):
            tr=np.flatnonzero(d['subjects']!=held);va=np.flatnonzero(d['subjects']==held)
            inner_g=np.zeros((len(tr),3));inner_k=np.zeros_like(inner_g);members=[]
            for inner,held_inner in enumerate(sorted(set(d['subjects'][tr])),1):
                it=tr[d['subjects'][tr]!=held_inner];iv=tr[d['subjects'][tr]==held_inner]
                ip,ir=fit(c,f'outer{fold}/inner{inner}',it,iv,x,variants,d,SEED+1000+fold*10+inner)
                local=np.flatnonzero(d['subjects'][tr]==held_inner)
                inner_g[local]=ip
                inner_k[local]=KNeighborsClassifier(5,weights='distance').fit(timing[it],y[it]).predict_proba(timing[iv])
                members.append(ir['configuration'])
            policy=fit_policy(inner_g,inner_k,y[tr],d['subjects'][tr],d['clip_ids'][tr])
            assert held not in policy['fit_subjects']
            if c=='angles_thesis_aug':
                old=OUT/c/'participant'/f'fold{fold}'
                r=json.loads((old/'result.json').read_text())
                assert r['configuration']['train_ids']==d['clip_ids'][tr].tolist()
                assert r['configuration']['validation_ids']==d['clip_ids'][va].tolist()
                assert sha(old/'model.keras')==r['model_sha256']
                p=np.load(old/'predictions.npy')
            else:
                p,r=fit(c,f'outer{fold}/base',tr,va,x,variants,d,SEED+100+fold)
            gp[va]=p;kp[va]=KNeighborsClassifier(5,weights='distance').fit(timing[tr],y[tr]).predict_proba(timing[va])
            tuned[va]=apply_policy(gp[va],kp[va],policy)
            if policy['confidence_threshold'] is not None: accepted[va]=tuned[va].max(1)>=policy['confidence_threshold']
            folds.append(dict(held_player=held,policy=policy,inner_memberships=members,
                raw=probability_metrics(y[va],gp[va],d['subjects'][va]),
                tuned=probability_metrics(y[va],tuned[va],d['subjects'][va],threshold=policy['confidence_threshold'])))
            print('OUTER',c,held,'weight',policy['gru_weight'],'T',round(policy['temperature'],2),flush=True)
        final_policy=fit_policy(gp,kp,y,d['subjects'],d['clip_ids'])
        report=dict(gru=probability_metrics(y,gp,d['subjects']),knn=probability_metrics(y,kp,d['subjects']),
            fixed_hybrid=probability_metrics(y,(gp+kp)*.5,d['subjects']),
            nested_tuned=probability_metrics(y,tuned,d['subjects']),folds=folds,final_policy=final_policy,
            nested_selected_coverage=float(accepted.mean()),
            nested_selected_accuracy=float(np.mean(tuned[accepted].argmax(1)==y[accepted])) if accepted.any() else None,
            nested_selected_wrong=int(np.sum(accepted & (tuned.argmax(1)!=y))))
        report['mean_player_macro_f1']=float(np.mean([f['tuned']['macro_f1'] for f in folds]))
        np.savez_compressed(DEST/c/'participant_oof.npz',gru=gp,knn=kp,tuned=tuned,y=y,subjects=d['subjects'],clip_ids=d['clip_ids'],accepted=accepted)
        write_json(DEST/c/'cv_results.json',report);reports[c]=report
        fit(c,'full_train',np.arange(len(y)),np.array([],int),x,variants,d,SEED+2000)
    selected=max(CONDITIONS,key=lambda c:(reports[c]['mean_player_macro_f1'],reports[c]['nested_tuned']['mean_subject_accuracy']))
    selection=dict(selected_condition=selected,diagnostic_used=False,
        scores={c:r['mean_player_macro_f1'] for c,r in reports.items()},protocol_sha256=sha(DEST/'protocol.json'),
        models={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.glob('**/model.keras'))},
        policies={c:r['final_policy'] for c,r in reports.items()})
    write_json(DEST/'frozen_selection.json',selection)
    print('FROZEN SELECTED',selected,selection['scores'],flush=True)


if __name__=='__main__': main()
