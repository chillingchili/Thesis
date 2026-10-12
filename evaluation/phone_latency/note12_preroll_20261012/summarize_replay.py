"""Summarize retained installed-app replay logs; never treat missing predictions as successes."""
import hashlib,json,math,re,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
rows=[]
for clip in ['C001','C002','C007']:
 text=(P/(clip+'.log')).read_text(encoding='utf-8')
 current=None
 for line in text.splitlines():
  if 'serve_ended start=' in line:
   fields=dict(re.findall(r'(\w+)=([^\s]+)',line.split('serve_ended ',1)[1]))
   current={'clip':clip,**fields}
   rows.append(current)
  elif current is not None and 'Feedback:' in line:
   current.update(dict(re.findall(r'(status|rules)=([^\s]+)',line)))
  elif current is not None and 'feedback_shown shown=' in line:
   current.update(dict(re.findall(r'(\w+)=([^\s]+)',line.split('feedback_shown ',1)[1])))
 def nums(key,group): return [float(x[key].removesuffix('ms')) for x in group if key in x]
summary={}
for clip in ['C001','C002','C007']:
 group=[x for x in rows if x['clip']==clip]
 pred=[x for x in group if x.get('pred') not in (None,'null')]
 def metric(key):
  v=nums(key,group)
  return {'n':len(v),'median':statistics.median(v) if v else None,'max':max(v) if v else None,'within_2000_ms':sum(x<=2000 for x in v)}
 summary[clip]={'ended_events':len(group),'predictions':len(pred),'confidence_at_least_0_60':sum(float(x['conf'])>=.6 for x in pred),'frames':nums('frames',group),'predicted_labels':[x['pred'] for x in pred],'confidences':nums('conf',pred),'feedback_statuses':[x.get('status') for x in group],'end_to_shown_ms':metric('end_to_shown'),'peak_to_shown_ms':metric('peak_to_shown'),'final_infer_ms':metric('final_infer'),'paddle_dropped':nums('paddle_dropped',group)}
result={'protocol':'Installed original-GRU app. Three looping prerecorded inputs; all gate-end events retained after preparation. Original weights unchanged; timestamped pre-roll and final-only inference. This is live-path replay, not actual camera capture; gate-end and motion-peak timestamps are not independently labeled physical serve completion.','summary':summary,'events':rows,'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('C*.log')}}
(P/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
