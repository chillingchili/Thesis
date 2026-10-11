// Coach C Session 2. Paste this entire file into Google Apps Script.
// Enable Services > YouTube Data API v3. Run createCoachCSession2Form three times.
// Exact approved wording-v2 feedback is embedded below. No local file access needed.
var C2_PLAYLIST_ID = 'PLdBi5bcaLX50'; // Replace if this is not the full playlist ID.
var C2_BATCH_SIZE = 10;
var C2_KEY = 'COACH_C_SESSION2_ORIGINAL_GRU_V1';
var C2_TITLE = 'Pickleball Feedback Validation - Coach C - Session 2 - Original GRU (30 Clips)';
var C2_CHOICES = ['Valid', 'Not valid', 'No coaching feedback to assess'];
// Legacy recorded categories retained only to recognize and remove old titles.
var C2_SERVE_TYPES = {
  C001: 'Lob', C002: 'Topspin', C003: 'Lob', C004: 'Topspin', C005: 'Lob',
  C006: 'Lob', C007: 'Drive', C008: 'Drive', C009: 'Topspin', C010: 'Drive',
  C011: 'Drive', C012: 'Topspin', C013: 'Topspin', C014: 'Drive', C015: 'Drive',
  C016: 'Lob', C017: 'Topspin', C018: 'Lob', C019: 'Topspin', C020: 'Topspin',
  C021: 'Lob', C022: 'Topspin', C023: 'Drive', C024: 'Lob', C025: 'Lob',
  C026: 'Drive', C027: 'Topspin', C028: 'Lob', C029: 'Drive', C030: 'Drive'
};
// Actual frozen original-GRU predictions, separate from legacy title labels.
var C2_CLASSIFICATIONS = {
  "C001": {
    "label": "Topspin",
    "lowConfidence": true
  },
  "C002": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C003": {
    "label": "Topspin",
    "lowConfidence": true
  },
  "C004": {
    "label": "Drive",
    "lowConfidence": true
  },
  "C005": {
    "label": "Lob",
    "lowConfidence": false
  },
  "C006": {
    "label": "Drive",
    "lowConfidence": false
  },
  "C007": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C008": {
    "label": "Lob",
    "lowConfidence": false
  },
  "C009": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C010": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C011": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C012": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C013": {
    "label": "Topspin",
    "lowConfidence": true
  },
  "C014": {
    "label": "Topspin",
    "lowConfidence": true
  },
  "C015": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C016": {
    "label": "Lob",
    "lowConfidence": false
  },
  "C017": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C018": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C019": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C020": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C021": {
    "label": "Lob",
    "lowConfidence": false
  },
  "C022": {
    "label": "Drive",
    "lowConfidence": false
  },
  "C023": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C024": {
    "label": "Lob",
    "lowConfidence": false
  },
  "C025": {
    "label": "Drive",
    "lowConfidence": true
  },
  "C026": {
    "label": "Topspin",
    "lowConfidence": true
  },
  "C027": {
    "label": "Topspin",
    "lowConfidence": false
  },
  "C028": {
    "label": "Drive",
    "lowConfidence": false
  },
  "C029": {
    "label": "Topspin",
    "lowConfidence": true
  },
  "C030": {
    "label": "Topspin",
    "lowConfidence": false
  }
};
var C2_FEEDBACK = [
  {
    "id": "C001",
    "feedback": "The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through."
  },
  {
    "id": "C002",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C003",
    "feedback": "The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through."
  },
  {
    "id": "C004",
    "feedback": "The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through."
  },
  {
    "id": "C005",
    "feedback": "No supported corrective cue for the available measurements. Overall serve form has not been assessed.\nPaddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing."
  },
  {
    "id": "C006",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C007",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.\nYour paddle orientation differed from the reference serves. Try a slower practice swing, keep your grip relaxed, and maintain a consistent paddle position through contact."
  },
  {
    "id": "C008",
    "feedback": "Your paddle orientation differed from the reference serves. Try a slower practice swing, keep your grip relaxed, and maintain a consistent paddle position through contact."
  },
  {
    "id": "C009",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C010",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C011",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C012",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C013",
    "feedback": "The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through."
  },
  {
    "id": "C014",
    "feedback": "The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through."
  },
  {
    "id": "C015",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C016",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.\nYour paddle orientation differed from the reference serves. Try a slower practice swing, keep your grip relaxed, and maintain a consistent paddle position through contact."
  },
  {
    "id": "C017",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C018",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.\nYour paddle orientation differed from the reference serves. Try a slower practice swing, keep your grip relaxed, and maintain a consistent paddle position through contact."
  },
  {
    "id": "C019",
    "feedback": "Your measured hip movement and paddle orientation are within the reference ranges. Repeat the motion and focus on consistency."
  },
  {
    "id": "C020",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.\nPaddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing."
  },
  {
    "id": "C021",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C022",
    "feedback": "Your measured hip movement and paddle orientation are within the reference ranges. Repeat the motion and focus on consistency."
  },
  {
    "id": "C023",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C024",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C025",
    "feedback": "The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through."
  },
  {
    "id": "C026",
    "feedback": "The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through.\nPaddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing."
  },
  {
    "id": "C027",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.\nPaddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing."
  },
  {
    "id": "C028",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  },
  {
    "id": "C029",
    "feedback": "The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through."
  },
  {
    "id": "C030",
    "feedback": "Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced."
  }
];

function c2VideoMap_() {
  var map = {}, token = '';
  do {
    var options = {playlistId: C2_PLAYLIST_ID, maxResults: 50};
    if (token) options.pageToken = token;
    var response = YouTube.PlaylistItems.list('snippet', options);
    (response.items || []).forEach(function(item) {
      var s = item.snippet || {};
      var matches = (s.title || '').match(/\bC\d{3}\b/gi) || [];
      var ids = matches.map(function(id) { return id.toUpperCase(); }).filter(function(id, i, all) {
        return all.indexOf(id) === i && C2_FEEDBACK.some(function(row) { return row.id === id; });
      });
      if (ids.length > 1) throw new Error('Ambiguous clip IDs in title: ' + s.title);
      if (!ids.length) return;
      var video = (s.resourceId || {}).videoId;
      if (!video) throw new Error('Missing video ID for ' + ids[0]);
      if (map[ids[0]]) throw new Error('Duplicate clip: ' + ids[0]);
      map[ids[0]] = 'https://www.youtube.com/watch?v=' + video;
    });
    token = response.nextPageToken || '';
  } while (token);
  var missing = C2_FEEDBACK.filter(function(row) { return !map[row.id]; });
  if (missing.length) throw new Error('Playlist missing: ' + missing.map(function(row) { return row.id; }).join(', '));
  return map;
}

function verifyCoachCSession2Playlist() {
  var map = c2VideoMap_();
  C2_FEEDBACK.forEach(function(row) { Logger.log(row.id + ' -> ' + map[row.id]); });
  Logger.log('All 30 clip IDs verified. Preview the videos to confirm correct uploads and playback.');
}

function c2Signature_(map, legacy) {
  var payload = JSON.stringify([C2_TITLE, C2_FEEDBACK, C2_CHOICES, map]);
  if (!legacy) payload += JSON.stringify(['classification-in-output-v1', C2_CLASSIFICATIONS]);
  return Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, payload, Utilities.Charset.UTF_8)
    .map(function(b) { return ('0' + ((b + 256) % 256).toString(16)).slice(-2); }).join('');
}

function c2OutputText_(row) {
  var prediction = C2_CLASSIFICATIONS[row.id];
  if (!prediction) throw new Error('Missing frozen classification: ' + row.id);
  return 'Serve classification: ' + prediction.label +
    (prediction.lowConfidence ? ' (low confidence)' : '') + ' | Feedback: ' + row.feedback;
}

function c2Titles_(id, withServeType) {
  var label = id + ' - ' + C2_SERVE_TYPES[id] + ' Serve';
  return [withServeType ? 'Clip ' + label : 'Clip ' + id,
    withServeType ? label : 'Video - ' + id,
    id + ' - Model output', id + ' - Feedback validity'];
}

function c2ValidatePrefix_(form, completed) {
  var items = form.getItems();
  if (items.length < completed * 4) throw new Error('Completed form items were removed. Stop and inspect the form.');
  for (var i = 0; i < completed; i++) {
    var row = C2_FEEDBACK[i], start = i * 4;
    var types = [i === 0 ? FormApp.ItemType.SECTION_HEADER : FormApp.ItemType.PAGE_BREAK,
      FormApp.ItemType.VIDEO, FormApp.ItemType.SECTION_HEADER, FormApp.ItemType.MULTIPLE_CHOICE];
    var titles = c2Titles_(row.id, false), renamedTitles = c2Titles_(row.id, true);
    for (var j = 0; j < 4; j++) {
      if (items[start+j].getType() !== types[j] ||
          (items[start+j].getTitle() !== titles[j] && items[start+j].getTitle() !== renamedTitles[j])) {
        throw new Error('Existing content changed at ' + row.id + '. Stop and inspect the form.');
      }
    }
    var output = items[start+2].asSectionHeaderItem().getHelpText();
    if (output !== row.feedback && output !== c2OutputText_(row)) throw new Error('Feedback changed: ' + row.id);
    var question = items[start+3].asMultipleChoiceItem();
    if (!question.isRequired() || JSON.stringify(question.getChoices().map(function(c) { return c.getValue(); })) !== JSON.stringify(C2_CHOICES)) {
      throw new Error('Question changed: ' + row.id);
    }
  }
}

function createCoachCSession2Form() {
  var lock = LockService.getUserLock();
  if (!lock.tryLock(1000)) throw new Error('Another batch is running. Wait and run again.');
  try {
    if (C2_FEEDBACK.length !== 30 || C2_FEEDBACK.some(function(row, i) {
      return row.id !== 'C' + ('00' + (i+1)).slice(-3) || !row.feedback.trim();
    })) throw new Error('Expected exact C001-C030 feedback data.');
    if (!Number.isInteger(C2_BATCH_SIZE) || C2_BATCH_SIZE < 1) throw new Error('BATCH_SIZE must be a positive integer.');
    var map = c2VideoMap_(); // Validate all clips before creating/modifying a form.
    var signature = c2Signature_(map);
    var props = PropertiesService.getUserProperties();
    var formId = props.getProperty(C2_KEY + '_FORM_ID');
    var form, completed = 0;
    if (formId) {
      if (props.getProperty(C2_KEY + '_SIGNATURE') !== signature) throw new Error('Feedback or display version changed. Run updateCoachCSession2Display() to migrate an existing form.');
      form = FormApp.openById(formId);
      completed = Number(props.getProperty(C2_KEY + '_COMPLETED') || '0');
      if (!Number.isInteger(completed) || completed < 0 || completed > 30) throw new Error('Invalid saved progress.');
      c2ValidatePrefix_(form, completed);
      if (completed === 30) {
        if (form.getItems().length !== 120) throw new Error('Unexpected items in completed form.');
        Logger.log('All 30 clips already built. No content changed.');
        showCoachCSession2Links();
        return;
      }
      if (form.getResponses().length) throw new Error('Responses already exist. Will not modify this incomplete form.');
    } else {
      form = FormApp.create(C2_TITLE);
      form.setAcceptingResponses(false);
      if (form.supportsAdvancedResponderPermissions()) form.setPublished(false);
      props.setProperty(C2_KEY + '_FORM_ID', form.getId());
      props.setProperty(C2_KEY + '_SIGNATURE', signature);
      props.setProperty(C2_KEY + '_COMPLETED', '0');
    }
    form.setAcceptingResponses(false);
    form.setDescription('Watch each clip and read the model output. Judge whether the feedback is appropriate for what you observe. Do not consult your Session 1 answers.\n\n' +
      'Valid: the feedback is appropriate.\nNot valid: the feedback is incorrect, misleading, or inappropriate.\nNo coaching feedback to assess: only an uncertainty/unavailable notice or a statement that no corrective cue is supported.\n\n' +
      'If a correction or positive coaching assessment is present alongside a missing-measurement notice, evaluate the supplied feedback. Do not choose the third option solely because one measurement is missing.');
    form.setProgressBar(true);
    form.setShuffleQuestions(false);
    form.setPublishingSummary(false);
    // Recover only the current incomplete trailing clip. Never delete completed clips.
    var items = form.getItems(), tail = items.slice(completed * 4);
    if (tail.length > 4) throw new Error('Unexpected trailing items; inspect the form before resuming.');
    var nextId = C2_FEEDBACK[completed].id;
    var expectedTitles = c2Titles_(nextId, false), renamedTitles = c2Titles_(nextId, true);
    var expectedTypes = [completed === 0 ? FormApp.ItemType.SECTION_HEADER : FormApp.ItemType.PAGE_BREAK,
      FormApp.ItemType.VIDEO, FormApp.ItemType.SECTION_HEADER, FormApp.ItemType.MULTIPLE_CHOICE];
    tail.forEach(function(item, i) {
      if (item.getType() !== expectedTypes[i] || (item.getTitle() && item.getTitle() !== expectedTitles[i] && item.getTitle() !== renamedTitles[i])) throw new Error('Unexpected partial clip content; no items deleted.');
    });
    for (var t = items.length - 1; t >= completed * 4; t--) form.deleteItem(items[t]);
    var end = Math.min(30, completed + C2_BATCH_SIZE);
    for (var i = completed; i < end; i++) {
      var row = C2_FEEDBACK[i];
      var titles = c2Titles_(row.id, false);
      (i === 0 ? form.addSectionHeaderItem() : form.addPageBreakItem()).setTitle(titles[0]);
      form.addVideoItem().setTitle(titles[1]).setHelpText('Watch the entire clip before evaluating.')
        .setVideoUrl(map[row.id]).setWidth(640).setAlignment(FormApp.Alignment.CENTER);
      form.addSectionHeaderItem().setTitle(row.id + ' - Model output').setHelpText(c2OutputText_(row));
      form.addMultipleChoiceItem().setTitle(row.id + ' - Feedback validity')
        .setHelpText('Is the model feedback valid for this clip?').setChoiceValues(C2_CHOICES).setRequired(true);
      props.setProperty(C2_KEY + '_COMPLETED', String(i + 1));
    }
    c2ValidatePrefix_(form, end);
    Logger.log('Completed ' + end + ' of 30 clips.');
    if (end === 30) {
      if (form.supportsAdvancedResponderPermissions()) form.setPublished(true);
      form.setAcceptingResponses(true);
      Logger.log('Form complete and accepting responses. Preview it before sharing the respondent link.');
      showCoachCSession2Links();
    } else Logger.log('Run createCoachCSession2Form again to continue.');
  } finally { lock.releaseLock(); }
}

function publishCoachCSession2Form() {
  var lock = LockService.getUserLock();
  if (!lock.tryLock(1000)) throw new Error('A batch is running. Try again after it finishes.');
  try {
    var props = PropertiesService.getUserProperties();
    var id = props.getProperty(C2_KEY + '_FORM_ID');
    if (!id || props.getProperty(C2_KEY + '_COMPLETED') !== '30') throw new Error('Build all 30 clips first.');
    if (props.getProperty(C2_KEY + '_SIGNATURE') !== c2Signature_(c2VideoMap_())) throw new Error('Feedback or playlist changed.');
    var form = FormApp.openById(id);
    c2ValidatePrefix_(form, 30);
    if (form.getItems().length !== 120) throw new Error('Unexpected form items.');
    if (form.supportsAdvancedResponderPermissions()) form.setPublished(true);
    form.setAcceptingResponses(true);
    showCoachCSession2Links();
  } finally { lock.releaseLock(); }
}

function showCoachCSession2Links() {
  var id = PropertiesService.getUserProperties().getProperty(C2_KEY + '_FORM_ID');
  if (!id) { Logger.log('Run createCoachCSession2Form first.'); return; }
  var form = FormApp.openById(id);
  Logger.log('Edit URL: ' + form.getEditUrl());
  Logger.log('Respondent URL: ' + form.getPublishedUrl());
}

function updateCoachCSession2Display() {
  var lock = LockService.getUserLock();
  if (!lock.tryLock(1000)) throw new Error('Another batch is running. Wait and run again.');
  try {
    var props = PropertiesService.getUserProperties();
    var formId = props.getProperty(C2_KEY + '_FORM_ID');
    if (!formId) throw new Error('Create the Coach C Session 2 form first.');
    var map = c2VideoMap_(), signature = c2Signature_(map);
    var savedSignature = props.getProperty(C2_KEY + '_SIGNATURE');
    if (savedSignature !== signature && savedSignature !== c2Signature_(map, true)) {
      throw new Error('Feedback or playlist changed; cannot migrate this form.');
    }
    var completed = Number(props.getProperty(C2_KEY + '_COMPLETED') || '0');
    if (!Number.isInteger(completed) || completed < 0 || completed > 30) throw new Error('Invalid saved progress.');
    var form = FormApp.openById(formId);
    c2ValidatePrefix_(form, completed);
    var items = form.getItems();
    props.setProperty(C2_KEY + '_SERVE_TITLES', 'false');
    for (var i = 0; i < completed; i++) {
      var titles = c2Titles_(C2_FEEDBACK[i].id, false);
      items[i * 4].setTitle(titles[0]);
      items[i * 4 + 1].setTitle(titles[1]);
      items[i * 4 + 2].asSectionHeaderItem().setHelpText(c2OutputText_(C2_FEEDBACK[i]));
    }
    c2ValidatePrefix_(form, completed);
    props.setProperty(C2_KEY + '_SIGNATURE', signature);
    Logger.log('Updated ' + completed + ' clips: clip-only titles and classification inside model output.');
    if (completed < 30) Logger.log('Continue createCoachCSession2Form; remaining clips will use this format.');
    showCoachCSession2Links();
  } finally { lock.releaseLock(); }
}

// Compatibility entry point: old menu selections now apply the requested format.
function renameClipsByServeType() {
  updateCoachCSession2Display();
}
