'use strict';

(() => {
  const data = window.RESEARCH_DATA;
  const $ = (selector) => document.querySelector(selector);
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
  const percent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
  const classes = ['Drive', 'Lob', 'Topspin'];
  const state = {study: 'v3', mode: 'gru', metric: 'diagnostic', selected: null, sample: 0, output: 0};
  const studyInfo = {
    v3: 'Matched input experiments / 209 eligible training clips. Diagnostic predictions average five GRU folds. Video variants follow their original clip into training folds; validation uses originals only.',
    v4: 'Nested fusion development / 209 eligible clips. Diagnostic predictions use single final fits, not v3’s five-fold ensembles. Lean control was selected on training-player macro-F1. Final confidence acceptance is disabled.',
    v2: 'Corrected complete-serve pipeline / 209 eligible training clips. The 100-epoch schedule was chosen using training development evidence. Quality failures count as incorrect in diagnostic accuracy; older v2 F1 fields are omitted because they used a different failure convention.',
    mobile: 'Actual bundled Android model assets, evaluated offline on recorded first-128 windows with fresh Lite poses. kNN bank: 438 older training examples. These results do not establish live phone accuracy or processing speed.',
    archive: 'Historical reported scores / September 24, 2026. Older cached inputs and exploratory protocols. Diagnostic scores are an unweighted mean across the two players, not pooled 175-clip accuracy. Approximate values and missing figures are retained as reported; no models were rerun for this page.'
  };
  const metricNames = {diagnostic: 'Diagnostic accuracy', player: 'Held-out-player accuracy', stratified: 'Stratified CV accuracy'};
  const modeNames = {gru: 'GRU', hybrid: 'Fixed hybrid', tuned: 'Tuned hybrid', all: ''};
  function rows() {
    return data.rows.filter((row) => row.group === state.study && (row.mode === 'all' || row.mode === state.mode));
  }
  function recallRows(values) {
    return values.map((value, index) => `<div class="recall-row"><span>${classes[index]}</span><span class="model-track" aria-hidden="true" style="--value:${value * 100}%"><span></span></span><span>${percent(value)}</span></div>`).join('');
  }
  function renderDetail(row) {
    const score = row[state.metric];
    let detail = `<p class="eyebrow">${modeNames[row.mode] || 'Saved model result'} / ${state.metric === 'diagnostic' ? 'Diagnostic' : state.metric === 'player' ? 'Held-out player' : 'Stratified CV'}</p><h3>${escape(row.name)}</h3><div class="detail-score"><strong>${percent(score?.accuracy)}</strong><span>${score?.n ? `${score.n} clips` : state.study === 'archive' ? 'historical reported score' : 'not reported'}</span></div>`;
    const recall = score?.recall || score?.class_recall;
    if (recall) {
      detail += `<div class="recall-title"><span>Recall by serve type</span><span>${score.macro_f1 !== undefined ? `Macro-F1 ${score.macro_f1.toFixed(3)}` : ''}</span></div>${recallRows(recall)}`;
      if (score.precision) detail += `<p class="class-precision">Precision: ${score.precision.map((value, index) => `${classes[index]} ${percent(value)}`).join(' · ')}</p>`;
      if (Math.min(...recall) < .1) detail += '<p class="report-warning">At least one serve type has recall below 10%. Overall accuracy hides this class imbalance.</p>';
    } else {
      detail += `<p class="detail-note">${score ? 'Comparable per-class recall is not included in this saved score.' : 'This evaluation was not reported for this model.'}</p>`;
    }
    if (row.selected) detail += '<p class="detail-selection">Selected research replay candidate / chosen on training-player macro-F1.</p>';
    detail += `<p class="detail-note">${escape(row.detail)}</p>`;
    $('#model-detail').innerHTML = detail;
  }
  function renderResults() {
    const list = rows();
    for (const button of document.querySelectorAll('[data-metric]')) {
      button.disabled = !list.some((row) => row[button.dataset.metric]);
    }
    if (!list.some((row) => row[state.metric])) state.metric = ['diagnostic', 'player', 'stratified'].find((key) => list.some((row) => row[key])) || 'diagnostic';
    for (const button of document.querySelectorAll('[data-metric]')) {
      const active = button.dataset.metric === state.metric;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    }
    if (!list.some((row) => row.id === state.selected)) state.selected = list.find((row) => row[state.metric])?.id || list[0]?.id;
    $('#study-note').textContent = studyInfo[state.study];
    $('#chart-title').textContent = metricNames[state.metric];
    $('#chart-unit').textContent = state.study === 'archive' ? 'Historical report' : state.metric === 'diagnostic' ? 'All 175 clips' : '209 eligible training clips';
    $('#model-bars').innerHTML = list.map((row) => {
      const value = row[state.metric]?.accuracy;
      return `<button class="model-row ${row.id === state.selected ? 'selected' : ''}" data-model-id="${row.id}" aria-pressed="${row.id === state.selected}" aria-label="${escape(row.name)}, ${metricNames[state.metric]} ${percent(value)}"><span class="model-name">${escape(row.name)}</span><span class="model-track" aria-hidden="true" style="--value:${Number.isFinite(value) ? value * 100 : 0}%"><span></span></span><span class="model-score ${!Number.isFinite(value) ? 'no-score' : ''}">${percent(value)}</span></button>`;
    }).join('');
    $('#results-table tbody').innerHTML = list.map((row) => `<tr><th scope="row">${escape(row.name)}</th>${['stratified', 'player', 'diagnostic'].map((key) => `<td>${percent(row[key]?.accuracy)}</td>`).join('')}</tr>`).join('');
    if (list.length) renderDetail(list.find((row) => row.id === state.selected) || list[0]);
  }
  function configureMode() {
    const select = $('#mode-select');
    const options = state.study === 'v4' ? ['gru', 'hybrid', 'tuned'] : state.study === 'v3' ? ['gru', 'hybrid'] : state.study === 'v2' ? ['gru'] : ['all'];
    if (!options.includes(state.mode)) state.mode = options[0];
    select.innerHTML = options.map((mode) => `<option value="${mode}">${modeNames[mode] || 'All saved models'}</option>`).join('');
    select.value = state.mode;
    select.disabled = options.length === 1;
  }
  $('#study-select').addEventListener('change', (event) => {
    state.study = event.target.value;
    state.selected = null;
    configureMode();
    renderResults();
  });
  $('#mode-select').addEventListener('change', (event) => {
    state.mode = event.target.value;
    state.selected = null;
    renderResults();
  });
  document.querySelectorAll('[data-metric]').forEach((button) => button.addEventListener('click', () => {
    state.metric = button.dataset.metric;
    renderResults();
  }));
  $('#model-bars').addEventListener('click', (event) => {
    const button = event.target.closest('[data-model-id]');
    if (!button) return;
    state.selected = button.dataset.modelId;
    for (const rowButton of $('#model-bars').querySelectorAll('button')) {
      const selected = rowButton === button;
      rowButton.classList.toggle('selected', selected);
      rowButton.setAttribute('aria-pressed', String(selected));
    }
    renderDetail(rows().find((row) => row.id === state.selected));
  });
  $('#detector-table tbody').innerHTML = data.detectors.map((row) => `<tr><th scope="row">${escape(row.name)}</th><td>${percent(row.precision)}</td><td>${percent(row.recall)}</td><td>${percent(row.map50)}</td><td>${percent(row.map5095)}</td><td><a class="text-link small" href="${row.source}" download aria-label="Download ${escape(row.name)} training log">CSV ↓</a></td></tr>`).join('');
  $('#calibration-table').innerHTML = `<table><caption>Held-out-player probability calibration / lower log loss is better</caption><thead><tr><th scope="col">Condition</th><th scope="col">Fixed log loss</th><th scope="col">Nested log loss</th><th scope="col">Outer accepted accuracy</th></tr></thead><tbody>${data.calibration.map((row) => `<tr><th scope="row">${escape(row.name)}</th><td>${row.before.toFixed(3)}</td><td>${row.after.toFixed(3)}</td><td>${percent(row.accepted_accuracy)}</td></tr>`).join('')}</tbody></table>`;

  function renderOutput() {
    const sample = data.samples[state.sample];
    const output = sample.outputs[state.output];
    const report = output.report;
    const serve = report.serve;
    let html = `<div class="output-head"><h3>${escape(serve.label)}</h3><p><strong>${percent(serve.confidence)}</strong> confidence</p></div><div class="probabilities">${recallRows(classes.map((name) => serve.probabilities[name.toLowerCase()]))}</div>`;
    if (report.shift || report.paddle) {
      const shift = report.shift;
      const paddle = report.paddle;
      html += `<div class="output-measurements"><div><p>Weight-shift proxy</p><strong>${shift?.status === 'ok' ? (shift.sufficient ? 'Within reference' : 'Below reference') : 'Unavailable'}</strong><span>${Number.isFinite(shift?.value) ? `${shift.value.toFixed(4)} / threshold ${shift.threshold.toFixed(4)}` : 'No supported measurement'}</span></div><div><p>Paddle angle</p><strong>${Number.isFinite(paddle?.angle) ? `${paddle.angle.toFixed(1)}°` : 'Unavailable'}</strong><span>${Number.isFinite(paddle?.angle) ? (paddle.state === 'OPTIMAL' ? 'Within measured reference band' : 'Outside measured reference band') : 'No supported measurement'}</span></div></div>`;
    }
    if (report.feedback) {
      const feedback = report.feedback;
      const title = feedback.status === 'PARTIAL' ? 'Partial measurements' : feedback.status === 'LOW_CONFIDENCE' ? 'Coaching withheld' : 'Corrective feedback';
      html += `<div class="feedback-box"><p class="eyebrow">${title} / saved rule output</p>${feedback.messages.map((message) => `<p>${escape(message)}</p>`).join('')}</div>`;
      if (feedback.notifications?.length) html += `<p class="output-source-note">${feedback.notifications.map(escape).join(' ')}</p>`;
      html += '<p class="fineprint">Feedback appropriateness has not been validated by Coach C. A model confidence value is not a verified probability of correctness.</p>';
    } else html += `<div class="feedback-box"><p class="eyebrow">Classifier output only</p><p>${escape(output.note)}</p></div>`;
    html += `<p class="output-source-note">${serve.label.toLowerCase() === sample.truth.toLowerCase() ? 'Prediction matches' : 'Prediction differs from'} the ${sample.truth} reference label. ${serve.truncated_frames ? `${serve.truncated_frames} valid frames omitted by this legacy window.` : 'Complete-serve resampling used.'}</p>`;
    if (sample.role === 'limitation') html += '<p class="fineprint">Limitation example: the GRU predicted Lob for a Coach B-labeled Drive, despite high confidence. A correct mechanics cue does not correct an incorrect subtype prediction.</p>';
    html += `<a class="text-link small" href="${output.download}" download>Download saved JSON <span aria-hidden="true">↓</span></a>`;
    $('#sample-report').innerHTML = html;
  }
  function renderSample() {
    const sample = data.samples[state.sample];
    const video = $('#sample-video');
    video.pause();
    video.poster = sample.poster;
    video.src = sample.video;
    video.load();
    $('#sample-name').textContent = sample.name;
    $('#sample-truth').textContent = `Reference: ${sample.truth} / ${sample.label_source}`;
    $('#sample-video-link').href = sample.video;
    $('#sample-model').innerHTML = sample.outputs.map((output, index) => `<option value="${index}">${escape(output.name)}</option>`).join('');
    $('#sample-model').value = '0';
    $('#sample-model').disabled = sample.outputs.length === 1;
    state.output = 0;
    for (const button of $('#sample-tabs').querySelectorAll('button')) {
      const active = Number(button.dataset.sample) === state.sample;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    }
    renderOutput();
  }
  $('#sample-tabs').innerHTML = data.samples.map((sample, index) => `<button data-sample="${index}" aria-pressed="${index === 0}" class="${index === 0 ? 'active' : ''}">${escape(sample.tab_label || sample.name)}</button>`).join('');
  $('#sample-tabs').addEventListener('click', (event) => {
    const button = event.target.closest('[data-sample]');
    if (!button) return;
    state.sample = Number(button.dataset.sample);
    renderSample();
  });
  $('#sample-model').addEventListener('change', (event) => {
    state.output = Number(event.target.value);
    renderOutput();
  });
  $('#print-button').addEventListener('click', () => window.print());
  configureMode();
  renderResults();
  renderSample();
})();
