"""Score app experiments by source replay, retaining absent and premature results."""
from pathlib import Path
import json,re,statistics
P=Path(__file__).resolve().parents[1]/'evaluation/mobile_methods_20261012'
records=[]
for file in (P/'phone').glob('C*.log'):
 clip,mode=file.stem.split('_',1);text=file.read_text();starts={};ends={};events=[];current=None
 for line in text.splitlines():
  fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
  if 'replay_start ' in line: starts[int(fields['round'])]=int(fields['time'])
  elif 'source_end ' in line: ends[int(fields['round'])]=int(fields['time'])
  elif 'serve_ended ' in line: current=fields;events.append(current)
  elif 'Feedback:' in line and current is not None: current.update({k:v for k,v in fields.items() if k in ['status','paddle']})
  elif 'feedback_shown ' in line and current is not None: current.update(fields)
 for round in range(1,4):
  start=starts.get(round);end=ends.get(round);next_start=starts.get(round+1,10**18)
  ev=[e for e in events if start is not None and start<=int(e['start'])<next_start]
  for e in ev:
   e['source_end_to_ui_ms']=int(e['shown'])-end if 'shown' in e and end is not None else None
  records.append({'clip':clip,'mode':mode,'round':round,'source_start':start,'source_end':end,'events':ev})
summary={}
for mode in sorted(set(r['mode'] for r in records)):
 rows=[r for r in records if r['mode']==mode];ev=[e for r in rows for e in r['events']];pred=[e for e in ev if e.get('pred') not in [None,'null']]
 values=[float(e['end_to_shown'].removesuffix('ms')) for e in ev if 'end_to_shown' in e]
 summary[mode]={'source_replays':len(rows),'source_replays_complete':sum(r['source_end'] is not None for r in rows),'end_events':len(ev),'replays_with_classification':sum(any(e.get('pred') not in [None,'null'] for e in r['events']) for r in rows),'classification_events':len(pred),'substantive_events':sum(e.get('status') in ['CORRECTIVE','POSITIVE'] for e in ev),'paddle_available_events':sum(e.get('paddle') not in [None,'UNAVAILABLE'] for e in ev),'gate_to_ui_median_ms':statistics.median(values) if values else None,'gate_to_ui_max_ms':max(values) if values else None,'events_before_source_end':sum(e.get('source_end_to_ui_ms') is not None and e['source_end_to_ui_ms']<0 for e in ev),'events_0_to_2s_after_source_end':sum(e.get('source_end_to_ui_ms') is not None and 0<=e['source_end_to_ui_ms']<=2000 for e in ev),'confidences':[float(e['conf']) for e in pred]}
(P/'phone_summary.json').write_text(json.dumps({'summary':summary,'replays':records},indent=2))
print(json.dumps(summary,indent=2))
