from pathlib import Path
import csv,json,re,statistics,collections,sys
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'evaluation/coach_c/mobile_live_20261012')
def rows(path): return list(csv.DictReader(path.open(encoding='utf-8-sig')))
def norm(s): return ' '.join(s.lower().split())
def main():
 key=rows(ROOT/'evaluation/coach_c/source/researcher_key.csv'); frozen={r['review_id']:r for r in rows(ROOT/'evaluation/coach_c/session2_gru/feedback_strings.csv')}; records=[]
 for r in key:
  clip=r['review_id'];path=OUT/'logs'/f'{clip}.log';log=path.read_text(encoding='utf-8') if path.exists() else '';events=[];current=None;start=end=None
  for line in log.splitlines():
   f=dict(re.findall(r'(\w+)=([^\s]+)',line))
   if 'replay_start ' in line: start=int(f['time'])
   elif 'source_end ' in line: end=int(f['time'])
   elif 'serve_ended ' in line: current=f;events.append(current)
   elif 'Feedback:' in line and current is not None:
    current.update({k:v for k,v in f.items() if k in ['status','paddle']})
    m=re.search(r'rules=(\[.*?\])',line)
    if m: current['rules']=m.group(1)
   elif 'FeedbackOutput:' in line and current is not None: current['displayed']=json.loads(line.split('FeedbackOutput:',1)[1].strip())
   elif 'feedback_shown ' in line and current is not None: current.update(f)
  for e in events: e['source_end_to_ui_ms']=int(e['shown'])-end if end is not None and 'shown' in e else None
  shown=[e for e in events if 'shown' in e and 'displayed' in e];final=shown[-1] if shown else {};old=frozen[clip]
  pred=final.get('pred');pred=None if pred in [None,'null'] else pred.lower();conf=float(final['conf']) if pred else None
  records.append(dict(clip=clip,recording_category=r['recording_category'],replay_complete='Replay finished' in log,source_start=start,source_end=end,events=events,output_count=len(shown),prediction=pred,confidence=conf,status=final.get('status','NO_OUTPUT' if 'Replay finished' in log else 'NOT_COMPLETED'),feedback=final.get('displayed',{}).get('text',''),frames=int(final['frames']) if 'frames' in final else None,paddle=final.get('paddle'),gate_to_ui_ms=int(final['end_to_shown'].removesuffix('ms')) if 'end_to_shown' in final else None,source_end_to_ui_ms=final.get('source_end_to_ui_ms'),category_agreement=pred==r['recording_category'].lower() if pred else None,confident=bool(pred and conf>=.60),frozen_prediction=old['predicted_serve'],frozen_status=old['feedback_status'],frozen_feedback=old['feedback_string'],exact_feedback_match=bool(final and norm(final.get('displayed',{}).get('text',''))==norm(old['feedback_string']))))
 shown=[r for r in records if r['output_count']];classified=[r for r in records if r['prediction']];confident=[r for r in records if r['confident']];sub=[r for r in records if r['status'] in ['CORRECTIVE','POSITIVE']];values=[r['gate_to_ui_ms'] for r in shown]
 summary=dict(expected_clips=30,replay_complete=sum(r['replay_complete'] for r in records),clips_with_output=len(shown),multiple_outputs=sum(r['output_count']>1 for r in records),status_counts=dict(collections.Counter(r['status'] for r in records)),classification_available=len(classified),recording_category_agreements=sum(r['category_agreement'] is True for r in records),confident_classifications=len(confident),confident_category_agreements=sum(r['category_agreement'] is True for r in confident),substantive_feedback=len(sub),substantive_with_category_agreement=sum(r['category_agreement'] is True for r in sub),paddle_available=sum(r['paddle'] not in [None,'UNAVAILABLE'] for r in records),frozen_substantive=sum(r['frozen_status'] in ['CORRECTIVE','POSITIVE'] for r in records),same_status_as_frozen=sum(r['status']==r['frozen_status'] for r in records),exact_feedback_matches=sum(r['exact_feedback_match'] for r in records),gate_to_ui_median_ms=statistics.median(values) if values else None,gate_to_ui_max_ms=max(values) if values else None,before_source_end=sum(r['source_end_to_ui_ms'] is not None and r['source_end_to_ui_ms']<0 for r in records),within_0_to_2s_after_source_end=sum(r['source_end_to_ui_ms'] is not None and 0<=r['source_end_to_ui_ms']<=2000 for r in records),over_2s_after_source_end=sum(r['source_end_to_ui_ms'] is not None and r['source_end_to_ui_ms']>2000 for r in records))
 summary['all_displayed_events']=sum(r['output_count'] for r in records)
 summary['clips_with_any_substantive_output']=sum(any(e.get('status') in ['CORRECTIVE','POSITIVE'] and 'shown' in e for e in r['events']) for r in records)
 summary['all_event_status_counts']=dict(collections.Counter(e['status'] for r in records for e in r['events'] if 'shown' in e))
 (OUT/'results.json').write_text(json.dumps({'summary':summary,'clips':records},indent=2),encoding='utf-8')
 flat=[{k:v for k,v in r.items() if k!='events'} for r in records]
 with (OUT/'clip_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
