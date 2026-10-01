"""Compare a saved Android replay against Python features and optional raw video.

No fitting, relabeling or threshold tuning. A known subtype can be recorded only
as a diagnostic observation. Missing callbacks/poses retain their timeline slot.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from serve_sequence import ROOT, build_sequence, extract_video
from serve_study import skeleton_features, write_json, sha
from serve_refinement import DEST, features, apply_policy
from extract_timing_features import timing_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--video',type=Path,help='Same original MP4 selected on the phone')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();trace=json.loads(args.trace.read_text(encoding='utf-8'))
    assert trace['version']=='serve_phone_replay_v1' and trace['pose_mode']=='VIDEO'
    rows=trace['frames'];times=np.array([r['timestamp_ms']/1000 for r in rows])
    p=np.array([np.array(r['xy']).reshape(33,2) if r['xy'] is not None else np.full((33,2),np.nan) for r in rows])
    v=np.array([r['visibility'] if r['visibility'] is not None else np.zeros(33) for r in rows])
    window,quality=build_sequence(p,v,times)
    phone_window=np.array(trace['angle_window'])
    difference=float(np.max(np.abs(window-phone_window)))
    np.testing.assert_allclose(window,phone_window,atol=2e-5,rtol=2e-5)
    assert quality['quality_accepted']==trace['quality_accepted']
    report=dict(trace_sha256=sha(args.trace),source_video_sha256=trace['video_sha256'],
        frames=len(rows),quality=quality,angle_contract_parity=True,max_angle_window_difference=difference,
        timestamp_interval_ms=dict(min=float(np.diff(times).min()*1000),median=float(np.median(np.diff(times))*1000),max=float(np.diff(times).max()*1000)),
        phone_predictions=trace['predictions'],physical_phone_trace=True)
    candidate=trace['predictions'].get('research_complete_serve')
    if candidate:
        import tensorflow as tf
        from sklearn.neighbors import KNeighborsClassifier
        from serve_study import training_data
        frozen=json.loads((DEST/'frozen_selection.json').read_text());condition=candidate['condition']
        assert condition==frozen['selected_condition']
        assert candidate['model_sha256']==sha(DEST/'android_bundle/model.tflite')
        assert candidate['selection_sha256']==sha(DEST/'frozen_selection.json')
        s,m=skeleton_features(p,v,times,trace['aspect']);x=features(condition,window,s,m)
        model=tf.lite.Interpreter(model_path=str(DEST/'android_bundle/model.tflite'));model.allocate_tensors()
        model.set_tensor(model.get_input_details()[0]['index'],x[None].astype(np.float32));model.invoke()
        gp=model.get_tensor(model.get_output_details()[0]['index'])
        d=training_data();bank=KNeighborsClassifier(5,weights='distance').fit(
            np.array([timing_features(a.reshape(128,5,2)) for a in d['X']]),d['y'])
        kp=bank.predict_proba(timing_features(window.reshape(128,5,2))[None])
        calibrated=apply_policy(gp,kp,frozen['policies'][condition])[0]
        np.testing.assert_allclose(gp[0],candidate['gru'],atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(kp[0],candidate['knn'],atol=3e-4,rtol=3e-4)
        np.testing.assert_allclose(calibrated,candidate['calibrated_hybrid'],atol=3e-4,rtol=3e-4)
        report['research_inference_parity']=True
    if args.video:
        assert sha(args.video)==trace['video_sha256'],'Phone and Python must replay identical MP4 bytes'
        vp,vv,vt,fps=extract_video(args.video)
        vw,vq=build_sequence(vp,vv,vt)
        report['fresh_python_video']=dict(source_frames=len(vp),fps=float(fps),quality=vq,
            angle_window_mean_absolute_difference=float(np.mean(np.abs(vw-window))),
            angle_window_max_absolute_difference=float(np.max(np.abs(vw-window))),
            note='This includes Android/Python decoder, timestamp and MediaPipe platform differences; not pure resampler error')
    output=args.output or args.trace.with_suffix('.comparison.json')
    write_json(output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
