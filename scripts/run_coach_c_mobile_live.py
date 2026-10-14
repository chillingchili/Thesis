"""Full frozen Coach C set through installed live app replay, no tuning."""
from pathlib import Path
import csv,hashlib,json,subprocess,time,zipfile,argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--gpu',action='store_true')
parser.add_argument('--clips',default='')
parser.add_argument('--hold-ms',type=int,default=350)
parser.add_argument('--motion-rate',action='store_true')
parser.add_argument('--relaxed-paddle',action='store_true')
parser.add_argument('--fullres-paddle',action='store_true')
parser.add_argument('--cooldown-secs',type=int,default=0)
parser.add_argument('--output',default='evaluation/coach_c/mobile_live_20261012')
ARGS=parser.parse_args()
OUT=ROOT/ARGS.output
ADB=['adb','-s','1fb637d']; PKG='com.thesis.pickleballserve'
def run(args): return subprocess.check_output(ADB+args,text=True,errors='replace').strip()
def logs(pid): return run(['logcat','-d','--pid',pid,'-v','threadtime','LiveLatency:I','ServeClipReplay:I','Feedback:I','FeedbackOutput:I','GruClassifier:I','AndroidRuntime:E','*:S'])
def main():
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'logs').mkdir(exist_ok=True)
 key=list(csv.DictReader((ROOT/'evaluation/coach_c/source/researcher_key.csv').open(encoding='utf-8-sig')))
 assert len(key)==30 and len({r['review_id'] for r in key})==30
 sources=[]
 for r in key:
  path=ROOT/'outputs/coach_c_evaluation_v2/session1_staging/videos'/f"{r['review_id']}.mp4"
  assert hashlib.sha256(path.read_bytes()).hexdigest()==r['source_sha256'],path
  sources.append((r,path))
 apk=ROOT/'android/app/build/outputs/apk/debug/app-debug.apk'
 baseline=json.loads((ROOT/'evaluation/phone_latency/note12_reaudit_20261012/device.json').read_text())
 with zipfile.ZipFile(apk) as z:
  assert all(hashlib.sha256(z.read(f'assets/gru_fold{i}.tflite')).hexdigest()==baseline['assets'][f'assets/gru_fold{i}.tflite'] for i in range(1,6))
 protocol={'selected_clips':ARGS.clips,'low_hold_ms':ARGS.hold_ms,'clips':key,'replays_per_clip':1,'model':'original five-fold GRU','pose':'GPU' if ARGS.gpu else 'CPU','independent_mechanics':ARGS.motion_rate,'independent_mechanics_guard':'32 real poses and 1500ms; shared rules only','motion_rate':ARGS.motion_rate,'pose_priority':True,'end_v2':False,'confidence':0.60,'relaxed_paddle':ARGS.relaxed_paddle,'relaxed_paddle_spacing_ms':150 if ARGS.relaxed_paddle else 350,'fullres_paddle':ARGS.fullres_paddle,'fullres_paddle_scope':'showcase-only contact-window second pass at source resolution; same detector and rules','apk_sha256':hashlib.sha256(apk.read_bytes()).hexdigest(),'source_hashes_verified':True,'original_gru_hashes_verified':True,'scope':'App live-path replay at camera pace; decoding excluded; not physical camera capture or independently annotated serve endpoints','idle':'1500ms prefix and 3000ms tail, synthetic excluded from classification/mechanics','order':'C001..C030; no model tuning; missing retained','battery_before':run(['shell','dumpsys','battery'])}
 protocol_path=OUT/'protocol.json'
 if protocol_path.exists():
  saved=json.loads(protocol_path.read_text())
  assert saved['apk_sha256']==protocol['apk_sha256'] and saved['pose']==protocol['pose'] and saved.get('motion_rate',False)==ARGS.motion_rate and saved.get('low_hold_ms',350)==ARGS.hold_ms and saved.get('relaxed_paddle',False)==ARGS.relaxed_paddle and saved.get('fullres_paddle',False)==ARGS.fullres_paddle, 'Use a new output directory for a different APK.'
 else: protocol_path.write_text(json.dumps(protocol,indent=2))
 print(run(['install','-r',str(apk)]),flush=True)
 for r,path in sources:
  if ARGS.clips and r['review_id'] not in ARGS.clips.split(','): continue
  clip=r['review_id']; target=OUT/'logs'/f'{clip}.log'
  if target.exists() and f'Replay finished clip={clip}.mp4' in target.read_text():
   print('Already finished '+clip,flush=True);continue
  run(['push',str(path),'/data/local/tmp/coach_c_replay.mp4'])
  run(['shell','run-as',PKG,'cp','/data/local/tmp/coach_c_replay.mp4',f'files/latency_input/{clip}.mp4'])
  device_hash=run(['shell','run-as',PKG,'sha256sum',f'files/latency_input/{clip}.mp4']).split()[0]
  assert device_hash==r['source_sha256']
  run(['shell','am','force-stop',PKG])
  run(['shell','am','start','-W','-n',PKG+'/.MainActivity','--es','replay_clip',clip+'.mp4','--ez','replay_loop','false','--ez','research_pose_priority','true','--ez','research_end_v2','false','--ez','research_pose_gpu',str(ARGS.gpu).lower(),'--ez','research_motion_rate',str(ARGS.motion_rate).lower(),'--ez','research_relaxed_paddle',str(ARGS.relaxed_paddle).lower(),'--ez','research_fullres_paddle',str(ARGS.fullres_paddle).lower(),'--el','research_low_hold_ms',str(ARGS.hold_ms)])
  time.sleep(1);pid=run(['shell','pidof',PKG]); deadline=time.monotonic()+240
  print('Running '+clip+' pid='+pid,flush=True)
  while time.monotonic()<deadline:
   time.sleep(3); log=logs(pid)
   if 'Replay finished ' in log or 'Replay failed' in log or 'FATAL EXCEPTION' in log: break
  time.sleep(2);log=logs(pid);target.write_text(log,encoding='utf-8')
  print('Saved '+clip+' finished='+str('Replay finished' in log)+' outputs='+str(log.count('FeedbackOutput:')),flush=True)
  if ARGS.cooldown_secs: print('Cooldown '+str(ARGS.cooldown_secs)+'s',flush=True);time.sleep(ARGS.cooldown_secs)
 run(['shell','am','force-stop',PKG])
 (OUT/'battery_after.txt').write_text(run(['shell','dumpsys','battery']))
 print('Full set complete',flush=True)
if __name__=='__main__': main()
