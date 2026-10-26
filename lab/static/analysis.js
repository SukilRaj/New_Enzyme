const $ = s => document.querySelector(s);
const fmt = (n, d=2) => typeof n === 'number' && Number.isFinite(n) ? n.toLocaleString(undefined, {maximumFractionDigits:d}) : '—';
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let token = '';

async function api(path, body) {
  const response = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json','X-Lab-Token':token}, body:JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || data.detail || 'Request failed');
  return data;
}

const tabs = [...document.querySelectorAll('[data-tab]')];
function selectTab(button) {
  tabs.forEach(tab => {tab.setAttribute('aria-selected', String(tab === button)); tab.tabIndex = tab === button ? 0 : -1;});
  document.querySelectorAll('.studio-page').forEach(page => page.hidden = page.id !== button.dataset.tab);
}
tabs.forEach((button, index) => {
  button.onclick = () => {selectTab(button); history.replaceState(null,'',`#${button.dataset.tab}`);};
  button.onkeydown = event => {
    let target;
    if (event.key === 'ArrowRight') target = tabs[(index+1)%tabs.length];
    if (event.key === 'ArrowLeft') target = tabs[(index+tabs.length-1)%tabs.length];
    if (event.key === 'Home') target = tabs[0];
    if (event.key === 'End') target = tabs.at(-1);
    if (target) {event.preventDefault(); selectTab(target); target.focus();}
  };
});

const names = {sequence:'Sequence only', environment:'Environment only', sequence_environment:'Sequence + environment', availability_control:'Sequence + environment + image availability', multimodal:'Sequence + environment + structure images', legacy:'Original EnzAIme neural checkpoint', training_mean:'Training log-mean baseline'};
function metricRow(name, m) {
  return `<tr><td>${esc(name)}</td><td>${fmt(m.mae_um)}</td><td>${fmt(m.rmse_um)}</td><td>${fmt(m.r2,3)}</td><td>${fmt(m.log_rmse,3)}</td><td>${fmt(m.spearman,3)}</td></tr>`;
}
function comparison(report) {
  $('#test-count').textContent = `${report.test_rows} rows / ${report.test_proteins} proteins`;
  $('#fixed-cohort').textContent = `Original neural split · ${report.test_rows} observations / ${report.test_proteins} proteins`;
  $('#cv-cohort').textContent = `${report.rows} observations · ${report.proteins} sequence groups`;
  $('#fixed-comparison').innerHTML = report.fixed_test.map(r => metricRow(names[r.mode] || r.name, r.metrics)).join('');
  $('#cv-comparison').innerHTML = Object.entries(report.nested_group_cv).map(([mode,m]) => metricRow(names[mode],m)).join('');
  const old = report.fixed_test.find(r => r.mode === 'legacy').metrics;
  const env = report.fixed_test.find(r => r.mode === 'environment').metrics;
  const cv = report.nested_group_cv;
  const envChange = 100*(old.mae_um-env.mae_um)/old.mae_um;
  const cvChange = 100*(cv.sequence.mae_um-cv.sequence_environment.mae_um)/cv.sequence.mae_um;
  const fusion = report.selected_test.mae_um;
  $('#improvement-cards').innerHTML = `<article><span>Environment baseline · fixed test</span><strong>${fmt(Math.abs(envChange),1)}% ${envChange>=0?'lower':'higher'} MAE</strong><small>${fmt(old.mae_um)} → ${fmt(env.mae_um)} µM. R² remains ${fmt(env.r2,3)}.</small></article><article><span>Sequence + environment · grouped CV</span><strong>${fmt(Math.abs(cvChange),1)}% ${cvChange>=0?'lower':'higher'} MAE</strong><small>Relative to sequence-only on the same outer-fold predictions.</small></article><article class="${fusion>old.mae_um?'negative':''}"><span>Selected fusion · fixed test</span><strong>${fmt(fusion)} µM MAE</strong><small>${fusion>old.mae_um?'Worse than the original. Kept experimental; no replacement.':'Lower than original on this historical test; further validation required.'}</small></article>`;
}

function renderAnalysis(result) {
  $('#analysis-empty').hidden = true;
  $('#analysis-result').hidden = false;
  $('#ranking-section').hidden = false;
  $('#result-title').textContent = result.selected_protein.protein_name;
  $('#download-analysis').href = result.artifact_url;
  const protein = result.selected_protein;
  $('#analysis-metrics').innerHTML = `<article><span>Center prediction</span><strong>${fmt(protein.center_prediction_um)}</strong><small>µM TPA · model estimate</small></article><article><span>10th-percentile scenario score</span><strong>${fmt(protein.robust_score_um)}</strong><small>µM TPA · ${result.scenario_count} condition combinations</small></article><article><span>Scenario spread</span><strong>${fmt(protein.scenario_std_um)}</strong><small>SD across conditions · not model uncertainty</small></article>`;
  $('#surface-caption').textContent = `Duration fixed at ${fmt(result.config.duration)} hours. Model: ${result.config.engine==='legacy'?'original neural checkpoint':'experimental fusion'}.`;
  const temperatures = [...new Set(result.surface.map(x => x.temperature))];
  const phValues = [...new Set(result.surface.map(x => x.ph))];
  const lo = Math.min(...result.surface.map(x => x.predicted_tpa_um));
  const hi = Math.max(...result.surface.map(x => x.predicted_tpa_um));
  $('#heatmap').style.gridTemplateColumns = `55px repeat(${phValues.length},minmax(24px,1fr))`;
  let cells = '<span class="axis-label">°C / pH</span>' + phValues.map(ph => `<span class="axis-label">${fmt(ph,2)}</span>`).join('');
  for (const t of temperatures) {
    cells += `<span class="axis-label">${fmt(t,1)} °C</span>`;
    for (const ph of phValues) {
      const point = result.surface.find(x => x.temperature===t && x.ph===ph);
      const intensity = (point.predicted_tpa_um-lo)/(hi-lo||1);
      const label = `${fmt(t)} °C, pH ${fmt(ph)}: ${fmt(point.predicted_tpa_um)} µM predicted TPA${point.outside_training_range?'; outside training range':''}`;
      cells += `<button class="${point.outside_training_range?'outside':''}" style="background:hsl(135 28% ${94-intensity*40}%)" data-detail="${esc(label)}" title="${esc(label)}" aria-label="${esc(label)}">${fmt(point.predicted_tpa_um,1)}</button>`;
    }
  }
  $('#heatmap').innerHTML = cells;
  document.querySelectorAll('[data-detail]').forEach(button => {
    const show = () => $('#heatmap-detail').textContent = button.dataset.detail;
    button.onmouseenter = show; button.onfocus = show; button.onclick = show;
  });
  $('#sweeps').innerHTML = ['temperature','ph','duration'].map(name => {
    const points = result.sweeps[name];
    const ys = points.map(p => p.predicted_tpa_um), min = Math.min(...ys), max = Math.max(...ys);
    const xy = points.map((p,i) => `${35+i/Math.max(1,points.length-1)*205},${120-(p.predicted_tpa_um-min)/(max-min||1)*85}`).join(' ');
    const unit = {temperature:'°C',ph:'pH',duration:'h'}[name];
    return `<article class="panel"><h4>${esc(name==='ph'?'pH':name[0].toUpperCase()+name.slice(1))} sensitivity · µM TPA</h4><svg viewBox="0 0 270 155" role="img" aria-label="${esc(name)} sweep in predicted micromolar TPA"><path d="M35 25V125H245" fill="none" stroke="#d7e0d0"/><polyline points="${xy}" fill="none" stroke="#45835a" stroke-width="2.4"/><text x="3" y="32" font-size="9" fill="#6f8267">${fmt(max,1)}</text><text x="3" y="122" font-size="9" fill="#6f8267">${fmt(min,1)}</text><text x="35" y="145" font-size="9" fill="#6f8267">${fmt(points[0].value,1)} ${unit}</text><text x="240" y="145" text-anchor="end" font-size="9" fill="#6f8267">${fmt(points.at(-1).value,1)} ${unit}</text></svg><small class="field-note">Other conditions held at the center.</small></article>`;
  }).join('');
  $('#support-note').textContent = `${fmt(100*result.outside_training_range_fraction,1)}% of grid points fall outside at least one training-condition range. This protein has ${protein.training_observations} training observations. The highest modeled grid value is ${fmt(result.best_grid_point.predicted_tpa_um)} µM at ${fmt(result.best_grid_point.temperature)} °C, pH ${fmt(result.best_grid_point.ph)}; this is not a verified optimum.`;
  $('#training-ranges').textContent = 'Training ranges: ' + Object.entries(result.training_condition_ranges).map(([name,range]) => `${name==='ph'?'pH':name}: ${fmt(range[0])}–${fmt(range[1])} ${{temperature:'°C',ph:'',duration:'hours'}[name]}`).join(' · ');
  $('#observed-evidence').innerHTML = result.observed_evidence.map(row => `<tr><td>${fmt(row.temperature)}</td><td>${fmt(row.ph)}</td><td>${fmt(row.duration)}</td><td>${fmt(row.measured_tpa_um)}</td><td>${row.used_in_training?'Training':'Held out / validation'}</td></tr>`).join('');
  $('#ranking').innerHTML = result.ranking.map(p => `<tr><td>${esc(p.protein_name)}</td><td>${fmt(p.center_prediction_um)}</td><td>${fmt(p.robust_score_um)}</td><td>${fmt(p.min_prediction_um)}–${fmt(p.max_prediction_um)}</td><td>${p.training_observations}${p.training_observations===0?' · unseen sequence':''}</td><td>${p.structure_used?'Used':'Not used'}</td></tr>`).join('');
}

$('#analysis-form').onsubmit = async event => {
  event.preventDefault();
  const form = Object.fromEntries(new FormData(event.target));
  for (const key in form) if (!['protein_id','engine'].includes(key)) form[key] = Number(form[key]);
  $('#analysis-error').textContent = '';
  $('#run-analysis').disabled = true;
  $('#run-analysis').textContent = 'Analyzing response surfaces…';
  try {renderAnalysis(await api('/api/analysis/run',form));}
  catch (error) {$('#analysis-error').textContent = error.message;}
  finally {$('#run-analysis').disabled = false; $('#run-analysis').textContent = 'Analyze condition window →';}
};

$('#generate-images').onclick = async () => {
  $('#image-error').textContent = ''; $('#generate-images').disabled = true;
  $('#generate-images').textContent = 'Generating verified views…';
  try {
    const result = await api('/api/analysis/images',{protein_id:$('#image-protein').value});
    const captions = {'front.svg':['Backbone · front','PCA axes 1–2; color follows sequence order.'],'side.svg':['Backbone · side','PCA axes 1–3; projection is not a new structure.'],'top.svg':['Backbone · top','PCA axes 2–3; the same verified coordinates.'],'distance.png':['Distance representation','Both axes: residue order. Brighter = closer; distances capped at 20 Å.'],'contacts.png':['C-alpha proximity','Both axes: residue order. White = C-alpha distance ≤ 8 Å; not atom contacts.']};
    $('#image-gallery').innerHTML = ['front.svg','side.svg','top.svg','distance.png','contacts.png'].map(name => result.images.find(image => image.name===name)).map(image => `<figure><img src="${esc(image.url)}" alt="${esc(captions[image.name][0])} for ${esc(result.protein_name)}"><figcaption>${esc(captions[image.name][0])}<a href="${esc(image.url)}" download aria-label="Download ${esc(captions[image.name][0])}">↓</a><small>${esc(captions[image.name][1])}</small></figcaption></figure>`).join('');
    $('#image-bundle').hidden = false;
    $('#image-bundle').href = `/api/analysis/images/${result.protein_id}/bundle.zip`;
    $('#image-provenance').hidden = false;
    $('#image-provenance').innerHTML = `<h3>${esc(result.protein_name)} · ${result.residues} residues</h3><p>${esc(result.source_kind)}. ${esc(result.note)}</p><p class="field-note">Source structure SHA-256</p><div class="provenance-hash">${esc(result.source_pdb_sha256)}</div><p><a class="inline" href="/api/analysis/images/${result.protein_id}/manifest.json" download>Download provenance manifest ↓</a></p>`;
  } catch (error) {$('#image-error').textContent = error.message;}
  finally {$('#generate-images').disabled = false; $('#generate-images').textContent = 'Generate protein images →';}
};

async function init() {
  const requestedTab = tabs.find(button => `#${button.dataset.tab}` === location.hash);
  if (requestedTab) selectTab(requestedTab);
  try {
    const [overview,state] = await Promise.all([api('/api/overview'),api('/api/analysis/status')]);
    token = overview.token;
    $('#protein').innerHTML = state.proteins.map(p => `<option value="${esc(p.id)}">${esc(p.name)} · ${p.length} residues</option>`).join('');
    const withStructures = state.proteins.filter(p => p.structure_available);
    $('#image-protein').innerHTML = withStructures.map(p => `<option value="${esc(p.id)}">${esc(p.name)} · ${p.length} residues</option>`).join('');
    $('#generate-images').disabled = withStructures.length === 0;
    $('#paper-mfs').textContent = `${fmt(state.paper.mfs)} ± ${fmt(state.paper.mfs_std)}`;
    $('#paper-afs').textContent = `${fmt(state.paper.afs)} ± ${fmt(state.paper.afs_std)}`;
    if (state.ready) {comparison(state.report); $('#run-analysis').disabled = false;}
    else $('#status-error').textContent = 'The analysis model has not been trained. Run scripts/train_condition_models.py with the project Python environment.';
  } catch (error) {$('#status-error').textContent = error.message;}
}
init();
