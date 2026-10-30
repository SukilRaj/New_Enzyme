const fmt=(n,d=2)=>typeof n==='number'&&Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:d}):'—';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $=s=>document.querySelector(s);
const displayName={sequence_environment:'Sequence + environment',sequence_context:'Sequence + assay context',context:'Environment + assay context',training_median:'Training median'};
const mse=m=>Number.isFinite(m?.mse_um2)?m.mse_um2:Number.isFinite(m?.rmse_um)?m.rmse_um**2:undefined;
async function load(){
 try{
  const response=await fetch('/api/training/status');if(!response.ok)throw Error(`Training service returned ${response.status}.`);const state=await response.json();
  $('#training-status').textContent=state.status==='not_started'?'Not trained':state.status==='completed'?'Training complete':state.status[0].toUpperCase()+state.status.slice(1);
  $('#training-detail').textContent=state.detail||'No training job has been started.';$('#training-progress').style.width=`${state.progress||0}%`;$('#training-percent').textContent=`${fmt(state.progress||0,0)}%`;
  if(state.status==='running')$('#training-warning').textContent=`Live training: ${state.detail}. This is a local CPU experiment; keep its terminal open until it completes.`;
  else if(state.status==='completed')$('#training-warning').textContent='Completed the 26-configuration study, including separate log-error and raw-MSE model selection. Predictions are experimental; compare every metric and group split before making claims.';
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
 const mseSelected=r.evaluation.sequence.sequence_environment_mse||oof;
 const strictMseSelected=r.evaluation.sequence_and_study.sequence_environment_mse||strict;
 const r2Flag=m=>Number.isFinite(m?.r2)&&m.r2<0?' class="negative"':'';
 $('#training-metrics').innerHTML=`<article><span>Nested CV · selected for log-error</span><strong>${fmt(oof.mae_um)} µM MAE</strong><small>RMSE ${fmt(oof.rmse_um)} µM · MSE ${fmt(mse(oof),0)} µM² · <span${r2Flag(oof)}>R² ${fmt(oof.r2,3)}</span></small></article><article><span>Nested CV · selected for raw MSE</span><strong>${fmt(mseSelected.mae_um)} µM MAE</strong><small>RMSE ${fmt(mseSelected.rmse_um)} µM · MSE ${fmt(mse(mseSelected),0)} µM² · <span${r2Flag(mseSelected)}>R² ${fmt(mseSelected.r2,3)}</span></small></article><article><span>Median baseline on same folds</span><strong>${gain(oof.mae_um,prior.mae_um)}</strong><small>${fmt(prior.mae_um)} → ${fmt(oof.mae_um)} µM MAE</small></article><article><span>Protein + publication holdout</span><strong>${fmt(strict.mae_um)} µM MAE</strong><small>RMSE ${fmt(strict.rmse_um)} µM · MSE ${fmt(mse(strict),0)} µM² · <span${r2Flag(strict)}>R² ${fmt(strict.r2,3)}</span></small></article><article><span>Final refit · research model</span><strong>${esc(r.production_spec.kind)} · ${esc(r.production_spec.mode.replaceAll('_',' + '))}</strong><small>${esc(r.production_spec.target_transform||'log1p')} target · trained on ${r.production_training_rows} rows, not a held-out score</small></article>`;
 evidenceWarning(r,[['Sequence-grouped · log-error',oof],['Sequence-grouped · raw-MSE',mseSelected],['Sequence + study-grouped · log-error',strict],['Sequence + study-grouped · raw-MSE',strictMseSelected]]);
 $('#objective-comparison').innerHTML=[['Sequence grouped','Log-error selection',oof],['Sequence grouped','Raw-MSE selection',mseSelected],['Sequence + study grouped','Log-error selection',strict],['Sequence + study grouped','Raw-MSE selection',strictMseSelected]].map(([design,objective,m])=>`<tr><td>${design}</td><td>${objective}</td><td>${fmt(m.mae_um)}</td><td>${fmt(mse(m),0)}</td><td>${fmt(m.rmse_um)}</td><td>${fmt(m.r2,3)}</td></tr>`).join('');
 $('#strict-comparison').innerHTML=['sequence_environment','sequence_environment_mse','sequence_context','context','training_median'].map(mode=>{const m=mode==='sequence_environment_mse'?strictMseSelected:r.evaluation.sequence_and_study[mode];return `<article><span>${esc(displayName[mode]||'Sequence + environment · raw-MSE selection')}</span><strong>${fmt(m.mae_um)} µM MAE</strong><small>RMSE ${fmt(m.rmse_um)} µM · MSE ${fmt(mse(m),0)} µM²</small><span class="${m.r2<0?'negative':''}">R² ${fmt(m.r2,3)}</span></article>`;}).join('');
 $('#historical-cohort').textContent=`Historical test · ${r.historical_selection.test.rows} measurements / 5 proteins · previously inspected`;
 $('#historical-table').innerHTML=Object.entries(r.fixed_test).map(([mode,row])=>`<tr><td>${esc(displayName[mode]||mode)}${row.spec?.kind?` · ${esc(row.spec.kind)}`:''}</td><td>${fmt(row.metrics.mae_um)}</td><td>${fmt(mse(row.metrics),0)}</td><td>${fmt(row.metrics.rmse_um)}</td><td>${fmt(row.metrics.r2,3)}</td><td>${fmt(row.metrics.log_rmse,3)}</td><td>${fmt(100*row.metrics.within_twofold_fraction,1)}%</td></tr>`).join('');
 $('#context-coverage').innerHTML=Object.entries(r.context_coverage).map(([key,n])=>`<tr><td>${esc(key.replaceAll('_',' '))}</td><td>${n} / ${r.rows} · ${fmt(100*n/r.rows,0)}%</td></tr>`).join('');
 $('#training-artifacts').innerHTML=['report.json','protocol.json','context_audit.json','splits.json','sequence_oof.csv','sequence_and_study_oof.csv','baseline_comparison.json','historical_predictions.csv'].map(name=>`<a href="${base}${name}">Download ${esc(name)} ↓</a>`).join('');
 fetch('/api/training/predictions').then(response=>response.json()).then(({rows})=>draw(rows)).catch(()=>{});
}
function evidenceWarning(report,entries){
 const el=$('#training-evidence-warning');if(!el)return;
 const negative=entries.filter(([,m])=>Number.isFinite(m?.r2)&&m.r2<0).map(([label])=>label);
 const parts=[];
 if(negative.length)parts.push(`Negative R² on ${negative.length} of ${entries.length} headline evaluations (${negative.join('; ')}): squared error exceeds predicting the evaluation-set mean, so the fitted model is not demonstrably better than a constant on those splits.`);
 else parts.push('No headline R² is negative, but the improvements remain small and split-dependent.');
 parts.push(`Evidence limits: ${report.protein_groups} exact-sequence groups and ${report.strict_groups} connected sequence+study groups, with the extremes concentrated in a few studies. This is exploratory nested cross-validation on one small dataset, not external validation, and does not validate a deployable or biological model.`);
 el.textContent=parts.join(' ');
 el.hidden=false;
}
function draw(rows){
 const sampled=rows.filter((_,i)=>i%Math.max(1,Math.ceil(rows.length/220))===0),max=Math.max(1,...sampled.map(p=>p.observed_um),...sampled.map(p=>p.sequence_environment_um));
 const circles=sampled.map(p=>{const raw=p.sequence_environment_mse_um??p.sequence_environment_um;return `<circle cx="${50+p.observed_um/max*270}" cy="${245-p.sequence_environment_um/max*210}" r="3.4" fill="#358460" opacity=".58"><title>Protein ${esc(p.protein_id)} · observed ${fmt(p.observed_um)} µM · log-error-selected ${fmt(p.sequence_environment_um)} µM</title></circle><circle cx="${50+p.observed_um/max*270}" cy="${245-raw/max*210}" r="3.4" fill="#d28a3c" opacity=".65"><title>Protein ${esc(p.protein_id)} · observed ${fmt(p.observed_um)} µM · raw-MSE-selected ${fmt(raw)} µM</title></circle>`;}).join('');
 $('#training-scatter').innerHTML=`<svg viewBox="0 0 355 280" role="img" aria-label="Held-out observed versus predicted TPA for log-error and raw-MSE model selection"><line x1="50" y1="245" x2="325" y2="25" stroke="#90a988" stroke-dasharray="5 4"/><line x1="50" y1="245" x2="325" y2="245" stroke="#cbd7ca"/><line x1="50" y1="245" x2="50" y2="25" stroke="#cbd7ca"/>${circles}<text x="50" y="16" font-size="9" fill="#358460">● Log-error</text><text x="135" y="16" font-size="9" fill="#d28a3c">● Raw-MSE</text><text x="185" y="273" text-anchor="middle" font-size="10" fill="#728271">Observed TPA · µM</text><text x="12" y="135" transform="rotate(-90 12 135)" text-anchor="middle" font-size="10" fill="#728271">Predicted · µM</text></svg>`;
}
load();
