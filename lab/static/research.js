const $ = s => document.querySelector(s);
const format = n => typeof n === 'number' && Number.isFinite(n) ? n.toLocaleString(undefined,{maximumFractionDigits:2}) : '—';
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let allRuns = [];
async function api(url) {const response=await fetch(url);if(!response.ok)throw new Error(`Unable to load ${url} (${response.status})`);return response.json();}
function renderRuns(){
  const domain=$('#domain').value;
  const runs=allRuns.filter(r=>domain==='all'||r.domain===domain);
  $('#run-count').textContent=`${runs.length} saved experiments`;
  $('#registry').innerHTML=runs.length?runs.map(r=>{
    const pet=r.domain==='PET'||r.domain==='PET environment';
    const method=r.domain==='PET environment'?`Condition sweep · ${r.config?.engine}`:r.domain==='PET'?'ESM-2 · condition-aware':`${r.config?.engine||'unknown'} · ${r.config?.selection||'greedy'}`;
    const detail=pet?`pH ${r.config?.ph} · ${r.config?.temperature} °C`:`Seed ${r.config?.seed} · ${r.config?.samples} samples`;
    return `<tr><td>${escapeHTML(r.id)}<small>${escapeHTML(new Date(r.created*1000).toLocaleString())}</small></td><td>${escapeHTML(r.domain)}<small>${escapeHTML(r.unit)}</small></td><td>${escapeHTML(method)}<small>${escapeHTML(detail)}</small></td><td>${escapeHTML(r.status)}</td><td>${r.artifact_url?`<a href="${escapeHTML(r.artifact_url)}">Download ↓</a>`:'Pending'}</td></tr>`;
  }).join(''):'<tr><td colspan="5">No experiments in this domain yet. Open a workspace to create one.</td></tr>';
}
async function refresh(){const button=$('#refresh');button.disabled=true;$('#error').textContent='';try{allRuns=await api('/api/research/runs');renderRuns();}catch(error){$('#error').textContent=error.message;}finally{button.disabled=false;}}
async function init(){const results=await Promise.allSettled([api('/api/research/overview'),api('/api/overview')]);if(results[0].status==='fulfilled'){const {pet,pet_service}=results[0].value;$('#pet-status').textContent=pet_service==='online'?'Connected':'Offline';$('#pet-dot').classList.toggle('pending',pet_service!=='online');$('#connection-note').textContent=pet_service==='online'?'Both engines are available through this workspace.':'Start-Research.cmd connects both engines. Saved evidence remains available.';$('#pet-records').textContent=format(pet.activity_records);$('#pet-proteins').textContent=`${format(pet.activity_proteins)} unique protein IDs · ${format(pet.training_rows)} training-pipeline rows`;$('#tm-records').textContent=format(pet.stability.records);$('#tm-sequences').textContent=`${format(pet.stability.sequences)} unique sequences`;$('#structures').textContent=format(pet.structures.matched_proteins);$('#pet-r2').textContent=format(pet.activity_r2);$('#image-status').textContent=pet.image_trained?'Checkpoint reports trained':'Training pending';$('#stability-status').textContent=pet.stability_checkpoint_present?'Present · verify compatibility':'Training pending';}else{$('#connection-note').textContent='Evidence could not load. Check the project terminal.';$('#pet-status').textContent='Unavailable';}if(results[1].status==='fulfilled')$('#gfp-records').textContent=format(results[1].value.rows);await refresh();}
$('#domain').onchange=renderRuns;$('#refresh').onclick=refresh;
let drawing='';for(let i=0;i<36;i++){const x=30+i*11;const a=120+Math.sin(i*.28)*66;const b=120-Math.sin(i*.28)*66;drawing+=`<line x1="${x}" y1="${a}" x2="${x}" y2="${b}" stroke="url(#strand)" opacity=".25"/><circle cx="${x}" cy="${a}" r="4" fill="#a5e6c2"/><circle cx="${x}" cy="${b}" r="4" fill="#86acd7"/>`;}$('#molecule').innerHTML=drawing;init();
