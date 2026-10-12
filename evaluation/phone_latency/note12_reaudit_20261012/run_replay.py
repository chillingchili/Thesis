import subprocess,time,pathlib
out=pathlib.Path('evaluation/phone_latency/note12_reaudit_20261012')
for clip in ['C001','C002','C007']:
 subprocess.run(['adb','-s','1fb637d','shell','am','force-stop','com.thesis.pickleballserve'],capture_output=True)
 stamp=subprocess.check_output(['adb','-s','1fb637d','shell','date','+%m-%d\ %H:%M:%S.000'],text=True).strip()
 subprocess.run(['adb','-s','1fb637d','shell','am','start','-W','-n','com.thesis.pickleballserve/.MainActivity','--es','replay_clip',clip+'.mp4','--ez','replay_loop','true'],capture_output=True,check=True)
 time.sleep(2)
 pid=subprocess.check_output(['adb','-s','1fb637d','shell','pidof','com.thesis.pickleballserve'],text=True).strip()
 print('Started '+clip+' pid='+pid,flush=True)
 # Separate expensive replay-file preparation from timed live-path observation.
 ready=False
 for attempt in range(24):
  time.sleep(10)
  log=subprocess.check_output(['adb','-s','1fb637d','logcat','-d','-v','threadtime','--pid',pid,'ServeClipReplay:I','*:S'],text=True,errors='replace')
  if 'Decoded clip='+clip+'.mp4' in log:
   ready=True
   print(clip+' decoded; starting 60-second observation',flush=True)
   break
  if attempt%3==2: print(clip+' preparation '+str((attempt+1)*10)+'s',flush=True)
 if ready:
  for n in range(2):
   time.sleep(30)
   print(clip+' observation '+str((n+1)*30)+'s',flush=True)
 else: print(clip+' preparation timed out',flush=True)
 log=subprocess.check_output(['adb','-s','1fb637d','logcat','-d','-v','threadtime','--pid',pid,'LiveLatency:I','ServeClipReplay:I','GruClassifier:I','MainActivity:E','Feedback:I','*:S'],text=True,errors='replace')
 (out/(clip+'.log')).write_text(log,encoding='utf-8')
 print('Saved '+clip+' log',flush=True)
subprocess.run(['adb','-s','1fb637d','shell','am','force-stop','com.thesis.pickleballserve'],capture_output=True)
