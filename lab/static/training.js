const fmt=(n,d=2)=>typeof n==='number'&&Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:d}):'—';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $=s=>document.querySelector(s);
const displayName={sequence_environment:'Sequence + environment',sequence_context:'Sequence + assay context',context:'Environment + assay context',training_median:'Training median'};
async function load(){
 try{
  const response=await fetch('/api/training/status');if(!response.ok)throw Error(`Training service returned ${response.status}.`);const state=await response.json();
  $('#training-status').textContent=state.status==='not_started'?'Not trained':state.status==='completed'?'Training complete':state.status[0].toUpperCase()+state.status.slice(1);
  $('#training-detail').textContent=state.detail||'No training job has been started.';$('#training-progress').style.width=`${state.progress||0}%`;$('#training-percent').textContent=`${fmt(state.progress||0,0)}%`;
  if(state.status==='running')$('#training-warning').textContent=`Live training: ${state.detail}. This is a local CPU experiment; keep its terminal open until it completes.`;
  else if(state.status==='completed')$('#training-warning').textContent='Completed the 24-configuration study and full-data refit. Predictions are still weak on held-out proteins, so this model stays experimental.';
  else if(state.status==='failed')$('#training-warning').textContent='Training did not complete. See the project terminal and saved run logs.';
  $('#training-warning').hidden=!['running','completed','failed'].includes(state.status);
  $('#training-empty').hidden=state.status!=='not_started'&&state.status!=='running';
  if(state.status==='running'){setTimeout(load,3000);return;}
  if(state.report)render(state.report,state.artifacts_url);
 }catch(error){$('#training-error').textContent=error.message;}
}
function render(r,base){
 $('#training-results').hidden=false;$('#training-cohort').textContent=`${r.rows} assay records · ${r.protein_groups} sequences · ${r.studies} papers`;
 const oof=r.evaluation.sequence.sequence_environment,strict=r.evaluation.sequence_and_study.sequence_environment;
 const prior=r.evaluation.sequence.training_median;
 const gain=(a,b)=>`${fmt(100*(b-a)/b,1)}% lower MAE`;
 $('#training-metrics').innerHTML=`<article><span>Nested group CV · sequence + environment</span><strong>${fmt(oof.mae_um)} µM MAE</strong><small>R² ${fmt(oof.r2,3)} · held-out sequence groups</small></article><article><span>Median baseline on same folds</span><strong>${gain(oof.mae_um,prior.mae_um)}</strong><small>${fmt(prior.mae_um)} → ${fmt(oof.mae_um)} µM</small></article><article><span>Protein + publication holdout</span><strong>${fmt(strict.mae_um)} µM MAE</strong><small>R² ${fmt(strict.r2,3)} · ${r.strict_groups} connected groups</small></article><article><span>Final refit · research model</span><strong>${esc(r.production_spec.kind)} · ${esc(r.production_spec.mode.replaceAll('_',' + '))}</strong><small>Trained on all ${r.production_training_rows} records · not a held-out score</small></article>`;
 $('#strict-comparison').innerHTML=['sequence_environment','sequence_context','context','training_median'].map(mode=>{const m=r.evaluation.sequence_and_study[mode];return `<article><span>${esc(displayName[mode])}</span><strong>${fmt(m.mae_um)} µM</strong><span class="${m.r2<0?'negative':''}">R² ${fmt(m.r2,3)}</span></article>`;}).join('');
 $('#historical-cohort').textContent=`Historical test · ${r.historical_selection.test.rows} measurements / 5 proteins · previously inspected`;
 $('#historical-table').innerHTML=Object.entries(r.fixed_test).map(([mode,row])=>`<tr><td>${esc(displayName[mode]||mode)}${row.spec?.kind?` · ${esc(row.spec.kind)}`:''}</td><td>${fmt(row.metrics.mae_um)}</td><td>${fmt(row.metrics.rmse_um)}</td><td>${fmt(row.metrics.r2,3)}</td><td>${fmt(row.metrics.log_rmse,3)}</td><td>${fmt(100*row.metrics.within_twofold_fraction,1)}%</td></tr>`).join('');
 $('#context-coverage').innerHTML=Object.entries(r.context_coverage).map(([key,n])=>`<tr><td>${esc(key.replaceAll('_',' '))}</td><td>${n} / ${r.rows} · ${fmt(100*n/r.rows,0)}%</td></tr>`).join('');
 $('#training-artifacts').innerHTML=['report.json','protocol.json','context_audit.json','splits.json','sequence_oof.csv','sequence_and_study_oof.csv','historical_predictions.csv'].map(name=>`<a href="${base}${name}">Download ${esc(name)} ↓</a>`).join('');
 fetch('/api/training/predictions').then(response=>response.json()).then(({rows})=>draw(rows)).catch(()=>{});
}
function draw(rows){
 const sampled=rows.filter((_,i)=>i%Math.max(1,Math.ceil(rows.length/220))===0),max=Math.max(1,...sampled.map(p=>p.observed_um),...sampled.map(p=>p.sequence_environment_um));
 const circles=sampled.map(p=>`<circle cx="${50+p.observed_um/max*270}" cy="${245-p.sequence_environment_um/max*210}" r="3.4" fill="#358460" opacity=".62"><title>Protein ${esc(p.protein_id)} · observed ${fmt(p.observed_um)} µM · predicted ${fmt(p.sequence_environment_um)} µM</title></circle>`).join('');
 $('#training-scatter').innerHTML=`<svg viewBox="0 0 355 280" role="img" aria-label="Held-out observed versus predicted TPA"><line x1="50" y1="245" x2="325" y2="25" stroke="#90a988" stroke-dasharray="5 4"/><line x1="50" y1="245" x2="325" y2="245" stroke="#cbd7ca"/><line x1="50" y1="245" x2="50" y2="25" stroke="#cbd7ca"/>${circles}<text x="185" y="273" text-anchor="middle" font-size="10" fill="#728271">Observed TPA · µM</text><text x="12" y="135" transform="rotate(-90 12 135)" text-anchor="middle" font-size="10" fill="#728271">Predicted · µM</text></svg>`;
}
load();
