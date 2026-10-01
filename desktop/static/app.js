const $ = id => document.getElementById(id);
let primary = null, secondary = null, objectUrl = null, jobId = null, busy = false;
const show = (id, visible = true) => { $(id).hidden = !visible; };
const text = (id, value) => { $(id).textContent = value; };
const fmt = (v, n = 1) => Number(v).toFixed(n);
const fileUrl = name => `/api/jobs/${jobId}/files/${name}`;

function error(message) { text('error', message); show('error'); }
function clearResults() {
  for (const id of ['serve-value','shift-value','paddle-value']) { text(id, '—'); $(id).className = ''; }
  text('serve-confidence',''); text('paddle-angle',''); text('serve-detail','Drive, lob or topspin');
  text('shift-detail','Forward transfer through the serve'); text('paddle-detail','Measured around estimated contact');
  text('landing-value', $('landing-mode').value === 'off' ? 'Not enabled' : 'Awaiting analysis');
  text('landing-detail','Landing needs visible court lines and the first bounce');
  text('feedback-title','A clearer picture of your serve.');
  $('feedback-content').replaceChildren(Object.assign(document.createElement('p'), {textContent:'Your coaching cues will appear here once the clip has been analyzed.'}));
  for(const id of ['probabilities','shift-meter','warnings','report-actions','technical','inspection','landing-playback']) show(id,false);
  text('technical-json','');
  $('landing-player').pause(); $('landing-player').removeAttribute('src'); $('landing-player').load();
  text('result-state','Awaiting analysis'); show('error',false);
}
function selectVideo(file) {
  if (busy || !file) return;
  if (!/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.name)) return error('Choose a supported video file.');
  if (file.size > 250*1024*1024) return error('Choose a video smaller than 250 MB.');
  if (!file.size) return error('This file is empty.');
  primary = file; clearResults(); jobId = null; show('progress-wrap',false);
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file); $('player').src = objectUrl;
  text('filename',file.name); text('fileduration',`${fmt(file.size/1024/1024)} MB`);
  $('start').value=0; $('end').value=''; show('playback-warning',false);
  show('dropzone',false); show('player-wrap'); show('replace'); $('analyze').disabled=false;
}
$('video-input').addEventListener('change',e=>selectVideo(e.target.files[0]));
$('dropzone').onclick = $('replace').onclick = () => $('video-input').click();
function wireDrop(element, receive) {
  element.addEventListener('dragover',e=>{e.preventDefault(); if(!busy) element.classList.add('dragover');});
  element.addEventListener('dragleave',()=>element.classList.remove('dragover'));
  element.addEventListener('drop',e=>{e.preventDefault();element.classList.remove('dragover'); if(!busy) receive(e.dataTransfer.files[0]);});
}
wireDrop($('dropzone'),selectVideo); wireDrop($('player-wrap'),selectVideo);
const setSecondary = file => { if (file) { secondary=file; text('secondary-drop',file.name); } };
wireDrop($('secondary-drop'),setSecondary);
$('secondary-input').onchange=e=>setSecondary(e.target.files[0]);
document.addEventListener('dragover',e=>e.preventDefault()); document.addEventListener('drop',e=>e.preventDefault());
function invalidateSettings(){
  if(busy)return;
  jobId=null;clearResults();show('progress-wrap',false);
  if(objectUrl)$('player').src=objectUrl;
}
$('landing-mode').onchange=()=>{show('secondary-wrap',$('landing-mode').value==='secondary');invalidateSettings();};
for(const id of ['model','moment','start','end'])$(id).addEventListener('change',invalidateSettings);
$('player').addEventListener('loadedmetadata',()=>{
  if(!jobId && Number.isFinite($('player').duration)) text('fileduration',`${fmt($('player').duration)} sec · ${fmt(primary.size/1024/1024)} MB`);
});
$('player').addEventListener('error',()=>show('playback-warning'));
function setBusy(value){
  busy=value;
  for (const id of ['replace','video-input','secondary-input','start','end','model','landing-mode','moment']) $(id).disabled=value;
  $('analyze').disabled=value||!primary; show('cancel',value); $('cancel').disabled=false;
  text('analyze',value?'Analyzing serve…':'Analyze serve ↗');
}
async function api(url, options) {
  const response=await fetch(url,options);
  const data=await response.json();
  if(!response.ok) throw Error(typeof data.detail==='string'?data.detail:'The request could not be completed.');
  return data;
}
$('analyze').onclick=async()=>{
  if(!primary||busy) return;
  const start=Number($('start').value), end=$('end').value;
  if(!Number.isFinite(start)||start<0||(end!==''&&(!Number.isFinite(Number(end))||Number(end)<=start))) return error('Choose a clip end after its start.');
  if($('landing-mode').value==='secondary'&&!secondary) return error('Add a second-camera clip in analysis settings.');
  clearResults();setBusy(true);$('cancel').disabled=true;show('progress-wrap');$('progress').value=0; text('progress-percent','0%');text('progress-text','Uploading to local analyzer…');text('result-state','Analyzing');
  const form=new FormData();form.append('video',primary);if(secondary)form.append('secondary',secondary);
  for(const [key,value] of Object.entries({start,end,model:$('model').value,landing:$('landing-mode').value,moment_matching:$('moment').checked}))form.append(key,value);
  try{
    jobId=null;
    const created=await api('/api/jobs',{method:'POST',body:form});jobId=created.id;$('cancel').disabled=false;
    await poll();
  }catch(e){error(e.message);text('result-state','Unable to analyze');}
  finally{setBusy(false);}
};
$('cancel').onclick=async()=>{
  if(!jobId)return;
  try{await api(`/api/jobs/${jobId}/cancel`,{method:'POST'});$('cancel').disabled=true;text('progress-text','Cancelling after the current model operation…');}
  catch(e){error(e.message);}
};
async function poll(){
  while(true){
    const job=await api(`/api/jobs/${jobId}`);
    $('progress').value=job.progress;text('progress-percent',`${Math.round(job.progress*100)}%`);text('progress-text',job.message);
    if(job.status==='complete'){render(job.result);return;}
    if(job.status==='failed')throw Error(job.message);
    if(job.status==='cancelled'){text('result-state','Cancelled');return;}
    await new Promise(resolve=>setTimeout(resolve,700));
  }
}
function render(r){
  text('result-state','Analysis complete');
  const s=r.serve,w=r.shift,p=r.paddle,l=r.landing,f=r.feedback;
  text('serve-value',s.label||'Unavailable');text('serve-confidence',s.confidence!=null?`${fmt(s.confidence*100)}%`:'');
  text('serve-detail',s.status==='ok'?(s.feedback_eligible===false?'Research candidate · coaching withheld':s.confidence<.6?'Below 60% confidence · coaching withheld':`${s.model==='hybrid'?'GRU + kNN5 hybrid':s.model==='ensemble'?'Five-fold ensemble':'Single GRU'} · ${s.valid_frames} usable frames`):s.reason||'Classification unavailable');
  if(s.probabilities){$('probabilities').replaceChildren();for(const [name,value] of Object.entries(s.probabilities)){
    const item=document.createElement('div');item.className='probability';
    const label=document.createElement('div');label.append(Object.assign(document.createElement('span'),{textContent:name}),Object.assign(document.createElement('span'),{textContent:`${fmt(value*100,0)}%`}));
    const track=document.createElement('div');track.className='prob-track';const bar=document.createElement('i');bar.style.width=`${value*100}%`;track.append(bar);item.append(label,track);$('probabilities').append(item);
  }show('probabilities');}
  text('shift-value',w.status==='ok'?(w.sufficient?'Sufficient':'Needs more transfer'):'Unavailable');$('shift-value').className=w.status==='ok'?(w.sufficient?'good':'warn'):'';
  text('shift-detail',w.status==='ok'?`Measured ${fmt(w.value,3)} · reference minimum ${fmt(w.threshold,3)}`:w.reason||'Weight shift unavailable');
  if(w.status==='ok'){show('shift-meter');$('shift-meter').firstElementChild.style.width=`${Math.min(100,w.value/w.threshold*70)}%`;}
  const paddleNames={OPTIMAL:'Within reference range',OUTSIDE_BASELINE:'Outside reference range',TOO_OPEN:'Too open',TOO_CLOSED:'Too closed'};
  text('paddle-value',paddleNames[p.state]||'Unavailable');text('paddle-angle',p.angle!=null?`${fmt(p.angle)}°`:'');
  text('paddle-detail',p.status==='ok'?`Reference ${fmt(p.low)}–${fmt(p.high)}° · ${p.measurements} contact-window measurements`:p.reason||'Paddle orientation unavailable');
  $('paddle-value').className=p.status==='ok'?(p.state==='OPTIMAL'?'good':'warn'):'';
  text('landing-value',l.status==='disabled'?'Not enabled':l.result?.zone||'Unavailable');
  text('landing-detail',l.result?`${fmt(l.result.x_ft)} ft across · ${fmt(l.result.y_ft)} ft along the court`:l.reason||'Enable ball landing in analysis settings');
  const titles={CORRECTIVE:'Focus on your next serve.',POSITIVE:'Keep that movement going.',LOW_CONFIDENCE:'Let’s get a clearer read.',INSUFFICIENT_DATA:'More information is needed.',PARTIAL:'A partial picture of your serve.'};
  text('feedback-title',titles[f.status]||'Coaching feedback');$('feedback-content').replaceChildren();
  for(const message of [...f.messages,...f.notifications])$('feedback-content').append(Object.assign(document.createElement('p'),{textContent:message}));
  $('warnings').replaceChildren();for(const message of r.warnings)$('warnings').append(Object.assign(document.createElement('p'),{textContent:message}));show('warnings',r.warnings.length>0);
  if(r.files.includes('preview.mp4')){$('player').src=fileUrl('preview.mp4');show('playback-warning',false);text('fileduration',`${fmt(r.range.end-r.range.start)} sec · selected range`);}
  if(r.files.includes('contact.jpg')){$('contact-image').src=fileUrl('contact.jpg');show('inspection');text('contact-time',`${fmt(r.contact.seconds,2)} sec · frame ${r.contact.frame}`);}
  if(r.files.includes('landing.mp4')){$('landing-player').src=fileUrl('landing.mp4');show('landing-playback');}
  $('download').href=fileUrl('report.json');text('runtime',`${fmt(r.seconds)}s processing`);show('report-actions');
  text('technical-json',JSON.stringify(r,null,2));show('technical');
}
$('model').addEventListener('change',()=>{const candidate=$('model').value.startsWith('v2_');$('moment').disabled=candidate;if(candidate)$('moment').checked=false;});
fetch('/api/health').then(r=>r.json()).then(data=>{for(const option of $('model').options){if(option.value.startsWith('v2_'))option.disabled=!data.candidate_ready;}const missing=Object.entries(data.models).filter(([,ok])=>!ok).map(([name])=>name);text('health',missing.length?`Missing: ${missing.join(', ')}`:'● Models ready');$('health').classList.toggle('bad',missing.length>0);}).catch(()=>{text('health','Analyzer offline');$('health').classList.add('bad');});
