"""Four predeclared live-path variants, same three inputs and corrected idle-frame replay."""
from pathlib import Path
import subprocess,time,json,hashlib
OUT=Path(__file__).resolve().parents[1]/'evaluation/mobile_methods_20261012/phone'
OUT.mkdir(parents=True,exist_ok=True)
ADB=['adb','-s','1fb637d']
variants=[('gpu_pose',False,False)]
protocol={'variants':variants,'clips':['C001','C002','C007'],'source_replays_per_case':3,'order':'rotate variant order by clip index','idle':'1500ms prefix +3000ms tail; duplicate idle frames excluded from classifier/hip/paddle counts','timing':'UI-update after app gate end; source_end logged separately; not independently marked physical completion','apk_sha256':hashlib.sha256(Path('android/app/build/outputs/apk/debug/app-debug.apk').read_bytes()).hexdigest()}
(OUT/'protocol_gpu.json').write_text(json.dumps(protocol,indent=2))
for ci,clip in enumerate(protocol['clips']):
 for name,priority,end in variants[ci:]+variants[:ci]:
  target=OUT/f'{clip}_{name}.log'
  if target.exists() and 'source_end clip='+clip+'.mp4 round=3' in target.read_text():
   print('Already completed '+target.name,flush=True);continue
  subprocess.run(ADB+['shell','am','force-stop','com.thesis.pickleballserve'],capture_output=True)
  subprocess.run(ADB+['shell','am','start','-W','-n','com.thesis.pickleballserve/.MainActivity','--es','replay_clip',clip+'.mp4','--ez','replay_loop','true','--ez','research_pose_priority',str(priority).lower(),'--ez','research_end_v2',str(end).lower(),'--ez','research_pose_gpu','true'],capture_output=True,check=True)
  time.sleep(2)
  pid=subprocess.check_output(ADB+['shell','pidof','com.thesis.pickleballserve'],text=True).strip()
  print('Running '+clip+' '+name+' pid='+pid,flush=True)
  deadline=time.monotonic()+240;completed=False
  while time.monotonic()<deadline:
   time.sleep(5)
   log=subprocess.check_output(ADB+['logcat','-d','--pid',pid,'-v','threadtime','LiveLatency:I','ServeClipReplay:I','Feedback:I','GruClassifier:I','AndroidRuntime:E','*:S'],text=True,errors='replace')
   if 'source_end clip='+clip+'.mp4 round=3' in log:
    completed=True;time.sleep(4);break
  log=subprocess.check_output(ADB+['logcat','-d','--pid',pid,'-v','threadtime','LiveLatency:I','ServeClipReplay:I','Feedback:I','GruClassifier:I','AndroidRuntime:E','*:S'],text=True,errors='replace')
  target.write_text(log,encoding='utf-8')
  print('Saved '+target.name+' completed='+str(completed),flush=True)
subprocess.run(ADB+['shell','am','force-stop','com.thesis.pickleballserve'],capture_output=True)
print('Phone matrix complete.',flush=True)
