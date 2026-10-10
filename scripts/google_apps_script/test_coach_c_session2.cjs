const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, 'coach_c_session2.gs'), 'utf8');

function setup() {
  const state = {forms: [], props: {FORM_ID_COACH_C_SESSION1: 'original'}, failVideo: false, missing: false, duplicate: false};
  class Item {
    constructor(type) { this.type = type; this.title = ''; this.help = ''; }
    getType() { return this.type; } getTitle() { return this.title; }
    setTitle(x) { this.title = x; return this; }
    setHelpText(x) { this.help = x; return this; } getHelpText() { return this.help; }
    setVideoUrl(x) { if (state.failVideo) { state.failVideo = false; throw Error('Simulated interruption'); } this.url = x; return this; }
    setWidth(x) { return this; } setAlignment(x) { return this; }
    setChoiceValues(x) { this.choices = Array.from(x); return this; }
    getChoices() { return this.choices.map(value => ({getValue: () => value})); }
    setRequired(x) { this.required = x; return this; } isRequired() { return this.required; }
    asSectionHeaderItem() { return this; } asMultipleChoiceItem() { return this; }
  }
  class Form {
    constructor() { this.items = []; this.responses = []; this.id = String(state.forms.length); }
    getId() { return this.id; } getItems() { return this.items.slice(); }
    getResponses() { return this.responses; }
    add(type) { const item = new Item(type); this.items.push(item); return item; }
    addSectionHeaderItem() { return this.add('SECTION_HEADER'); } addPageBreakItem() { return this.add('PAGE_BREAK'); }
    addVideoItem() { return this.add('VIDEO'); } addMultipleChoiceItem() { return this.add('MULTIPLE_CHOICE'); }
    deleteItem(item) { this.items.splice(this.items.indexOf(item), 1); }
    setAcceptingResponses(x) { this.accepting = x; } supportsAdvancedResponderPermissions() { return true; }
    setPublished(x) { this.published = x; } setDescription(x) { this.description = x; }
    setProgressBar(x) {} setShuffleQuestions(x) {} setPublishingSummary(x) {}
    getEditUrl() { return 'edit'; } getPublishedUrl() { return 'respond'; }
  }
  const ctx = vm.createContext({
    Logger: {log() {}},
    FormApp: {ItemType: Object.fromEntries(['SECTION_HEADER','PAGE_BREAK','VIDEO','MULTIPLE_CHOICE'].map(x=>[x,x])), Alignment: {CENTER: 'CENTER'},
      create() { const f = new Form(); state.forms.push(f); return f; }, openById(id) { return state.forms[Number(id)]; }},
    PropertiesService: {getUserProperties: () => ({getProperty: k => state.props[k] ?? null, setProperty(k,v) { state.props[k]=v; }})},
    LockService: {getUserLock: () => ({tryLock: () => true, releaseLock() {}})},
    Utilities: {DigestAlgorithm: {SHA_256: 'sha256'}, Charset: {UTF_8: 'utf8'}, computeDigest: (_, data) => Array.from(crypto.createHash('sha256').update(data).digest())},
    YouTube: {PlaylistItems: {list(_, opts) {
      let ids = Array.from({length: 30}, (_,i) => 'C'+String(i+1).padStart(3,'0'));
      if(state.missing) ids.pop(); if(state.duplicate) ids[29]=ids[0];
      const offset=opts.pageToken ? 15 : 0;
      return {items: ids.slice(offset,offset+15).map(id => ({snippet: {title:id+'.mp4',resourceId:{videoId:'video-'+id}}})), nextPageToken: offset===0 ? 'next' : ''};
    }}}
  });
  vm.runInContext(source, ctx);
  return {state, ctx};
}

let {state,ctx}=setup();
ctx.createCoachCSession2Form(); assert.equal(state.forms[0].items.length,40); assert.equal(state.forms[0].accepting,false);
ctx.createCoachCSession2Form(); assert.equal(state.forms[0].items.length,80);
ctx.createCoachCSession2Form(); assert.equal(state.forms[0].items.length,120); assert.equal(state.forms[0].published,true); assert.equal(state.forms[0].accepting,true);
state.forms[0].responses.push({}); ctx.createCoachCSession2Form(); assert.equal(state.forms[0].items.length,120);
assert.equal(state.props.FORM_ID_COACH_C_SESSION1,'original');
for(let i=0;i<30;i++) {
  assert.equal(state.forms[0].items[i*4+2].help,ctx.c2OutputText_(ctx.C2_FEEDBACK[i]));
  assert.deepEqual(state.forms[0].items[i*4+3].choices,['Valid','Not valid','No coaching feedback to assess']);
  assert.equal(state.forms[0].items[i*4+3].required,true);
}
const root=path.resolve(__dirname,'../..');
const localStatus=path.join(root,'outputs/coach_c_evaluation_v2/status.json');
const reportFolder=fs.existsSync(localStatus)
  ? path.join(root,JSON.parse(fs.readFileSync(localStatus)).active_session2_run,'revised_reports')
  : path.join(root,'evaluation/coach_c/session2/reports');
for(const row of ctx.C2_FEEDBACK) {
  const report=JSON.parse(fs.readFileSync(path.join(reportFolder,row.id+'.json')));
  assert.equal(row.feedback,[...report.feedback.messages,...report.feedback.notifications].join('\n'));
  assert.equal(ctx.C2_CLASSIFICATIONS[row.id].label.toLowerCase(),report.serve.label.toLowerCase());
  assert.equal(ctx.C2_CLASSIFICATIONS[row.id].lowConfidence,report.serve.confidence < .60);
}
console.log('PASS: 3 batches, pagination, exact 30 messages, required options, idempotence, Session 1 isolation');

({state,ctx}=setup()); state.failVideo=true;
assert.throws(()=>ctx.createCoachCSession2Form(),/Simulated interruption/);
assert.equal(state.forms[0].items.length,2); ctx.createCoachCSession2Form();
assert.equal(state.forms[0].items.length,40); assert.equal(state.forms[0].items[0].title,'Clip C001');
console.log('PASS: interrupted clip recovery without duplication');

for(const flag of ['missing','duplicate']) {
  ({state,ctx}=setup()); state[flag]=true;
  assert.throws(()=>ctx.createCoachCSession2Form(),/missing|Duplicate/);
  assert.equal(state.forms.length,0);
}
console.log('PASS: missing and duplicate playlist clips fail before form creation');

({state,ctx}=setup()); ctx.createCoachCSession2Form();
ctx.C2_FEEDBACK[0].feedback='Changed';
assert.throws(()=>ctx.createCoachCSession2Form(),/changed/); assert.equal(state.forms[0].items.length,40);
console.log('PASS: changed wording cannot overwrite an existing form');

({state,ctx}=setup()); ctx.createCoachCSession2Form(); state.forms[0].responses.push({});
assert.throws(()=>ctx.createCoachCSession2Form(),/Responses already exist/); assert.equal(state.forms[0].items.length,40);
console.log('PASS: existing responses protect incomplete form from edits');

({state,ctx}=setup()); ctx.createCoachCSession2Form();
state.forms[0].addSectionHeaderItem().setTitle('Unrelated section');
assert.throws(()=>ctx.createCoachCSession2Form(),/Unexpected partial/); assert.equal(state.forms[0].items.length,41);
console.log('PASS: unrelated trailing content is not deleted');

({state,ctx}=setup());
assert.throws(()=>ctx.renameClipsByServeType(),/Create the Coach C Session 2/);
ctx.createCoachCSession2Form();
// Recreate a partially built form from the prior script, including its signature.
for(let i=0;i<10;i++) {
  const row=ctx.C2_FEEDBACK[i], titles=ctx.c2Titles_(row.id,true);
  state.forms[0].items[i*4].title=titles[0];
  state.forms[0].items[i*4+1].title=titles[1];
  state.forms[0].items[i*4+2].help=row.feedback;
}
state.props[ctx.C2_KEY+'_SIGNATURE']=ctx.c2Signature_(ctx.c2VideoMap_(),true);
ctx.updateCoachCSession2Display();
assert.equal(state.forms[0].items[0].title,'Clip C001');
assert.equal(state.forms[0].items[1].title,'Video - C001');
state.failVideo=true; assert.throws(()=>ctx.createCoachCSession2Form(),/Simulated interruption/);
ctx.createCoachCSession2Form(); ctx.createCoachCSession2Form();
ctx.renameClipsByServeType(); ctx.renameClipsByServeType();
ctx.createCoachCSession2Form(); ctx.publishCoachCSession2Form();
assert.equal(state.forms[0].items.length,120);
assert.equal(state.props.FORM_ID_COACH_C_SESSION1,'original');
for(let i=0;i<30;i++) {
  const row=ctx.C2_FEEDBACK[i];
  assert.equal(state.forms[0].items[i*4].title,'Clip '+row.id);
  assert.equal(state.forms[0].items[i*4+1].title,'Video - '+row.id);
  assert.equal(state.forms[0].items[i*4+2].help,ctx.c2OutputText_(row));
  assert.ok(state.forms[0].items[i*4+2].help.endsWith(' | Feedback: '+row.feedback));
  assert.equal(state.forms[0].items[i*4+3].title,row.id+' - Feedback validity');
}
assert.equal(ctx.C2_FEEDBACK.filter(r=>ctx.c2OutputText_(r).includes('(low confidence)')).length,12);
assert.ok(ctx.C2_FEEDBACK.some(r=>ctx.C2_CLASSIFICATIONS[r.id].label!==ctx.C2_SERVE_TYPES[r.id]));
console.log('PASS: old-form migration, clip-only titles, actual predictions, 12 uncertainty labels, later batches, recovery, unchanged feedback/questions');
