"""Real training clips and missing-frame cases for Kotlin/Python contract parity."""
import json
import cv2
import numpy as np
from serve_study import ROOT, training_rows, skeleton_features, write_json
from serve_sequence import build_sequence
from extract_joint_angles import clip_angles
from serve_refinement import lean, temperature


def main():
    rows=training_rows();cases=[]
    for subject in ['Beginner1','Beginner2','CoachA']:
        row=next(r for r in rows if r['subject']==subject) if 'subject' in rows[0] else next(r for r in rows if r['clip_id'].startswith(subject+'_'))
        z=np.load(ROOT/'data/serve_v2/poses'/f"{row['clip_id']}.npz")
        cap=cv2.VideoCapture(str(ROOT/row['video']));aspect=cap.get(cv2.CAP_PROP_FRAME_WIDTH)/cap.get(cv2.CAP_PROP_FRAME_HEIGHT);cap.release()
        for missing in ([False,True] if subject=='Beginner1' else [False]):
            points=z['points'].copy();visibility=z['visibility'].copy();times=z['timestamps']
            if missing: points[10:20]=np.nan;visibility[10:20]=0
            angles=clip_angles(points).reshape(-1,10)
            window,q=build_sequence(points,visibility,times)
            s,m=skeleton_features(points,visibility,times,aspect)
            frames=[]
            for p,v,t,a in zip(points,visibility,times,angles):
                frames.append(dict(timestamp_ms=float(t*1000),xy=p.reshape(-1).tolist() if np.isfinite(p).all() else None,
                    visibility=v.tolist(),angles=a.tolist() if np.isfinite(a).all() else None))
            cases.append(dict(id=row['clip_id']+('_missing' if missing else ''),aspect=aspect,frames=frames,
                angles=window.reshape(-1).tolist(),skeleton=s.reshape(-1).tolist(),motion=m.reshape(-1).tolist(),
                lean=lean(window,s,m).reshape(-1).tolist(),accepted=q['quality_accepted']))
    gp=np.array([[.9,.05,.05],[.2,.6,.2]]);kp=np.array([[.1,.7,.2],[.3,.2,.5]])
    policy=dict(gru_weight=.75,temperature=2.3,confidence_threshold=None)
    fixture=dict(cases=cases,policy=policy,policy_gru=gp.tolist(),policy_knn=kp.tolist(),
        policy_expected=temperature(.75*gp+.25*kp,2.3).tolist())
    write_json(ROOT/'android/app/src/test/resources/fixtures/serve_sequence.json',fixture)
    print('Wrote',len(cases),'training-only fixtures')


if __name__=='__main__':main()
