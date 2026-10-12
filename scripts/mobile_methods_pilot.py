"""Training-only participant-separated mobile-rate/distillation pilot. Never reads Coach C responses."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,hashlib,time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from extract_joint_angles import clip_angles
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/mobile_methods_20261012/training_v2';OUT.mkdir(parents=True,exist_ok=True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
D=np.load(ROOT/'data/serve_v2/train.npz');ok=D['quality_accepted']
ids,y,groups,full=[D[k][ok] for k in ['clip_ids','y','subjects','X']]
assert set(groups)=={'CoachA','Beginner1','Beginner2'}
def mobile(clip,rate):
 d=np.load(ROOT/'data/serve_v2/poses'/f'{clip}.npz');t=d['timestamps'].astype(float);t=t-t[0]
 assert 0 < t[-1] <= 12, 'Cached timestamps must be seconds'
 f=clip_angles(d['points']).reshape(-1,10)
 valid=np.isfinite(f).all(1)&(d['visibility'][:,[11,12,14,16,20,23,24]].min(1)>=.5)
 target=np.arange(0,t[-1]+1e-6,1/rate)
 ix=np.unique(np.searchsorted(t,target).clip(0,len(t)-1));ix=ix[valid[ix]]
 if len(ix)>128: ix=ix[np.linspace(0,len(ix)-1,128).round().astype(int)]
 x=np.zeros((128,10),np.float32);x[:len(ix)]=f[ix]
 return x,len(ix)
rates=[6,15,30]
views={};counts={}
for rate in rates:
 pairs=[mobile(i,rate) for i in ids];views[rate]=np.array([a for a,b in pairs]);counts[rate]=np.array([b for a,b in pairs])
protocol={'seed':20261012,'epochs':30,'training_exposure':'Full-window control repeated to match mobile-view optimizer steps; same initialization seed as mobile teacher0','rates':rates,'training_ids':ids.tolist(),'groups':groups.tolist(),'training_sha256':hashlib.sha256((ROOT/'data/serve_v2/train.npz').read_bytes()).hexdigest(),'policy':'3 outer held-out participants; fixed epoch budget; all rate variants stay with source clip; teacher fit only on outer training participants; student targets 0.5 one-hot + 0.5 mean probabilities of two 32-unit teachers; no Coach C data or diagnostic selection','limitation':'Cached VIDEO-mode poses sampled at mobile rates; not actual live extraction, motion gate, resolution or dropped-frame reproduction. Research pilot, not deployment replacement.'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
def model(units,seed):
 tf.keras.utils.set_random_seed(seed)
 x=tf.keras.Input((128,10));v=tf.keras.layers.Masking()(x);v=tf.keras.layers.GRU(units)(v);v=tf.keras.layers.Dense(3,activation='softmax')(v)
 m=tf.keras.Model(x,v);m.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='categorical_crossentropy');return m

def fit(name,units,x,target,seed,folder):
 p=folder/(name+'.keras')
 if p.exists(): return tf.keras.models.load_model(p)
 print('Training '+str(p.relative_to(OUT)),flush=True)
 m=model(units,seed);m.fit(x,target,epochs=30,batch_size=32,verbose=0);m.save(p);return m

def predict(m,x): return np.concatenate([m(x[j:j+32],training=False).numpy() for j in range(0,len(x),32)])
def metrics(p,truth,n):
 pred=p.argmax(1);usable=n>=32;accepted=usable&(p.max(1)>=.6)
 return {'n':len(truth),'correct':int((pred==truth).sum()),'accuracy':float((pred==truth).mean()),'enough_frames':int(usable.sum()),'accepted':int(accepted.sum()),'accepted_correct':int((accepted&(pred==truth)).sum()),'usable_classification_accuracy':float((pred[usable]==truth[usable]).mean()) if usable.any() else None}
reports=[]
for fold,(tr,va) in enumerate(LeaveOneGroupOut().split(full,y,groups),1):
 folder=OUT/f'fold{fold}';folder.mkdir(exist_ok=True)
 one=np.eye(3,dtype=np.float32)[y[tr]];xm=np.concatenate([views[r][tr] for r in rates]);ym=np.tile(one,(len(rates),1))
 control=fit('full_window_control32_matched',32,np.tile(full[tr],(len(rates),1,1)),ym,200+fold*10,folder)
 teachers=[fit('mobile_teacher'+str(k),32,xm,ym,200+fold*10+k,folder) for k in range(2)]
 soft=sum(predict(m,xm) for m in teachers)/2
 student=fit('distilled16',16,xm,.5*ym+.5*soft,300+fold,folder)
 small=fit('hard_label16',16,xm,ym,400+fold,folder)
 for rate in rates:
  preds={'full_window_control32_matched':predict(control,views[rate][va]),'mobile32':predict(teachers[0],views[rate][va]),'mobile_ensemble2':sum(predict(m,views[rate][va]) for m in teachers)/2,'distilled16':predict(student,views[rate][va]),'hard_label16':predict(small,views[rate][va])}
  for name,p in preds.items():
   reports.append({'fold':fold,'held_out':str(groups[va][0]),'rate':rate,'model':name,**metrics(p,y[va],counts[rate][va])})
   np.savez(folder/f'{name}_{rate}fps.npz',p=p,y=y[va],ids=ids[va],real_frames=counts[rate][va])
 (OUT/'results.json').write_text(json.dumps(reports,indent=2))
 print('Completed held-out '+str(groups[va][0]),flush=True)
# Full-training exports are isolated candidates, never copied to production assets.
folder=OUT/'full_train';folder.mkdir(exist_ok=True)
xm=np.concatenate([views[r] for r in rates]);ym=np.tile(np.eye(3,dtype=np.float32)[y],(len(rates),1))
teachers=[fit('mobile_teacher'+str(k),32,xm,ym,700+k,folder) for k in range(2)]
soft=sum(predict(m,xm) for m in teachers)/2
student=fit('distilled16',16,xm,.5*ym+.5*soft,800,folder)
for name,m in [('mobile32',teachers[0]),('distilled16',student)]:
 @tf.function(input_signature=[tf.TensorSpec([1,128,10],tf.float32)])
 def serve(x): return m(x,training=False)
 from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
 frozen=convert_variables_to_constants_v2(serve.get_concrete_function())
 c=tf.lite.TFLiteConverter.from_concrete_functions([frozen]);c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS];c._experimental_lower_tensor_list_ops=False
 blob=c.convert();(folder/(name+'.tflite')).write_bytes(blob)
 fixture=views[15][0];(folder/'input.json').write_text(json.dumps(fixture.reshape(-1).tolist()))
 print('Exported '+name+' bytes='+str(len(blob)),flush=True)
print('Finished training and distillation pilot.',flush=True)
