import { useEffect, useState } from 'react'
import MutationWorkspace from './MutationWorkspace'
import './App.css'
import './ResearchApp.css'

const API = import.meta.env.VITE_API_BASE || '/api'
const pages = [['design', 'Recommend', '01'], ['structure', 'Sequence & structure', '02'], ['evaluation', 'Model evaluation', '03'], ['data', 'Data & system', '04']]
const number = (value, digits = 2) => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : 'Pending'

async function request(path, body, signal) {
  const response = await fetch(`${API}${path}`, { signal, ...(body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}) })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : 'The request failed. Check your inputs and service status.')
  return payload
}

export default function ResearchApp() {
  const [page, setPage] = useState('design')
  const [health, setHealth] = useState(null)
  const [connection, setConnection] = useState('Connecting')
  useEffect(() => {
    const controller = new AbortController()
    let timer
    async function refresh() {
      try {
        const data = await request('/workspace/status', null, controller.signal)
        if (!controller.signal.aborted) { setHealth(data); setConnection('Connected') }
      } catch (error) {
        if (!controller.signal.aborted) { setConnection('Reconnecting'); setHealth(null) }
        if (error.name === 'AbortError') return
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 8000)
      }
    }
    refresh()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [])
  return <div className="research-shell">
    <aside className="research-nav"><a className="research-brand" href="#" onClick={e => { e.preventDefault(); setPage('design') }}><span>e</span>EnzAIme<small>PROTEIN RESEARCH</small></a>
      <p className="nav-caption">Research workspace</p><nav aria-label="Workspace">{pages.map(([id, label, index]) => <button key={id} className={page === id ? 'selected' : ''} onClick={() => setPage(id)}><span>{index}</span>{label}</button>)}</nav>
      <div className="research-nav-bottom"><span className={`connection-dot ${health ? 'connected' : ''}`} />{connection}<p>Sequence · Structure · Evidence</p><small>v2 / Local workspace</small></div>
    </aside>
    <main className="research-main"><header className="research-top"><span>Workspace / <strong>{pages.find(p => p[0] === page)[1]}</strong></span><span className="research-badge">EXPERIMENTAL RESEARCH</span></header>
      {page === 'design' && <Design health={health} />}
      <div className="research-page" hidden={page !== 'structure'}><p className="eyebrow">02 / Inspect the molecule</p><MutationWorkspace apiBase={API} /></div>
      {page === 'evaluation' && <Evaluation />}
      {page === 'data' && <DataSystem health={health} />}
    </main>
  </div>
}

function Design({ health }) {
  const [form, setForm] = useState({ temperature: '40', ph: '8', duration: '48', top_n: '5' })
  const [objective, setObjective] = useState('activity')
  const [result, setResult] = useState(null)
  const [submitted, setSubmitted] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [explanation, setExplanation] = useState(null)
  const [explaining, setExplaining] = useState(false)
  const [explanationError, setExplanationError] = useState('')
  const ready = health?.activity?.status === 'research_checkpoint_loaded'
  async function rank(event) {
    event.preventDefault(); setBusy(true); setError(''); setResult(null); setExplanation(null); setExplanationError('')
    const payload = { ...Object.fromEntries(Object.entries(form).map(([key, value]) => [key, Number(value)])), objective }
    try { const data = await request('/neural/recommend', payload); setResult(data); setSubmitted(payload) }
    catch (e) { setError(e instanceof TypeError ? 'Connection interrupted. The workspace will reconnect automatically.' : e.message) }
    finally { setBusy(false) }
  }
  async function explain() {
    setExplaining(true); setExplanationError('')
    try { setExplanation(await request('/neural/explain', submitted)) }
    catch (e) { setExplanationError(e.message) }
    finally { setExplaining(false) }
  }
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ ...result, llm_explanation: explanation }, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'enzaime-recommendations.json'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <div className="research-page"><div className="research-intro"><div><p className="eyebrow">01 / Model-guided exploration</p><h1>Design with evidence.<br /><em>Explore what comes next.</em></h1><p>Rank PET-degrading proteins under your assay conditions, inspect their structures, and understand the evidence behind each result.</p></div><div className="method-card"><span>THE RESEARCH LOOP</span><ol><li>Represent the protein</li><li>Predict and compare</li><li>Inspect the evidence</li><li>Prioritize for validation</li></ol><small>Inspired by ESM-guided directed evolution</small></div></div>
    <div className="capability-grid"><Capability label="Activity model" value={ready ? 'Checkpoint loaded' : 'Connecting'} detail="ESM-2 + Attention1D" good={ready} /><Capability label="Thermostability" value={health?.activity?.thermostability_ready ? 'Checkpoint available' : 'Training pending'} detail={health?.stability_data ? `${number(health.stability_data.records, 0)} linked Tm records` : 'Sequence-linked Tm data'} /><Capability label="Image branch" value={health?.activity?.image_trained ? 'Trained' : 'Training pending'} detail={health?.structure_data ? `${health.structure_data.matched_proteins} verified structures` : 'Verified structures required'} /><Capability label="Local explanations" value={health?.ollama?.status === 'ready' ? 'Available' : 'Offline'} detail={health?.ollama?.model || 'Llama 3.2 1B'} good={health?.ollama?.status === 'ready'} /></div>
    <div className="design-grid"><form className="research-panel condition-panel" onSubmit={rank}><p className="eyebrow">Your experiment</p><h2>Define assay conditions</h2><label>Optimization objective<select value={objective} onChange={e => setObjective(e.target.value)}><option value="activity">TPA activity · neural ranking</option><option value="joint" disabled={!health?.activity?.thermostability_ready}>Activity + Tm · Pareto front</option></select></label><p className="field-help">Joint activity–stability ranking becomes available after stability training and evaluation.</p>
      {[['temperature', 'Temperature', '°C', undefined, undefined], ['ph', 'pH', '', 0, 14], ['duration', 'Reaction duration', 'hours', 0.001, undefined]].map(([key, label, unit, min, max]) => <label key={key}>{label}<div className="unit-input"><input name={key} type="number" required step="any" min={min} max={max} value={form[key]} onChange={e => setForm({ ...form, [key]: e.target.value })} /><span>{unit}</span></div></label>)}
      <label>Candidates<select value={form.top_n} onChange={e => setForm({ ...form, top_n: e.target.value })}><option value="3">Top 3</option><option value="5">Top 5</option><option value="10">Top 10</option></select></label>
      <button className="run-button" disabled={!ready || busy || explaining}>{busy ? 'Scoring proteins…' : 'Rank candidates →'}</button><p className="field-help">Predicted concentrations are computational estimates. Current held-out R²: {number(health?.activity?.test_metrics?.r2)}.</p>
    </form><div className="research-results"><div className="results-heading"><h2>Candidate shortlist</h2>{result && <button className="text-button" onClick={download}>Export results ↓</button>}</div>
      {error && <p role="alert" className="research-alert">{error}</p>}
      {!result && <div className="research-empty" role="status"><span className="empty-glyph">⌁</span><h3>{busy ? 'Evaluating the protein catalog' : 'Your next experiment starts here'}</h3><p>{busy ? 'Combining sequence representations with the selected conditions.' : 'Choose assay conditions and rank candidates to inspect predicted activity and model evidence.'}</p><div className="empty-steps"><span>Sequence</span><i>→</i><span>Neural model</span><i>→</i><span>Shortlist</span></div></div>}
      {result && <><p className="field-help">{result.ranking_note}</p><p className="results-context">{submitted.temperature} °C · pH {submitted.ph} · {submitted.duration} hours · {result.recommendations.length} candidates</p>{result.recommendations.map(r => <article className="research-candidate" key={r.protein_id}><div className="candidate-title"><span className="rank-label">{String(r.rank).padStart(2, '0')}</span><div><h3>{r.protein_name}</h3><small>Protein {r.protein_id} · {r.image_used ? 'Sequence + image + conditions' : 'Sequence + conditions'}</small></div><div className="prediction-number">{number(r.predicted_activity_um)}<small>µM predicted TPA</small></div></div><div className="candidate-meta"><span>Tm: {r.predicted_tm_c == null ? 'not predicted' : `${number(r.predicted_tm_c)} °C`}</span><span>{r.outside_training_conditions ? 'Outside training conditions' : 'Within training condition ranges'}</span></div><details><summary>Inspect model evidence</summary><Attribution proteinId={r.protein_id} conditions={submitted} /><p>Highest attention at positions {r.explanation.attention_positions.join(', ')}.</p><p>{r.explanation.note}</p><p>Image-removal change: {number(r.explanation.image_ablation_delta_log1p, 4)} in log(1 + TPA). {r.image_used ? '' : 'No image was used for this prediction.'}</p></details></article>)}
        <div className="research-panel explanation-panel"><div className="results-heading"><div><p className="eyebrow">Local explanation</p><h3>Interpret this shortlist</h3></div><button className="secondary-button" onClick={explain} disabled={explaining || health?.ollama?.status !== 'ready'}>{explaining ? 'Writing explanation…' : explanation ? 'Regenerate' : 'Explain with Llama'}</button></div>{explaining && <p role="status">Your local model is working. A cold start can take up to two minutes.</p>}{explanation && <><p>{explanation.text}</p><small>{explanation.note}</small></>}{explanationError && <p role="alert" className="research-alert">{explanationError}</p>}{!explanation && !explaining && <p>Uses the computed ranking and evaluation evidence. Scores are produced by the neural model.</p>}</div></>}
    </div></div></div>
}

function Capability({ label, value, detail, good }) { return <div className="capability"><span>{label}</span><strong><i className={good ? 'is-ready' : ''} />{value}</strong><small>{detail}</small></div> }

function Evaluation() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => { request('/workspace/evaluation').then(setData).catch(e => setError(e.message)) }, [])
  const m = data?.activity_review
  const points = data?.held_out_predictions || []
  const max = Math.max(1, ...points.flatMap(p => [p.observed_um, p.predicted_um])) * 1.1
  return <div className="research-page"><p className="eyebrow">03 / Evidence before claims</p><h1>Measure the model.</h1><p className="page-lede">Saved held-out results from the imported Kaggle experiment. A single split is an initial check of feasibility.</p>{error && <p role="alert">{error}</p>}<div className="capability-grid">{[['MAE', m?.mae_um, 'µM'], ['RMSE', m?.rmse_um, 'µM'], ['R²', m?.r2, 'held-out coefficient'], ['Test proteins', m?.test_proteins, `${m?.test_rows || 0} measurements`]].map(([label, value, detail]) => <Capability key={label} label={label} value={number(value)} detail={detail} />)}</div><div className="evaluation-grid"><div className="research-panel"><h2>Observed vs. predicted activity</h2><svg className="evaluation-scatter" viewBox="0 0 460 340" role="img" aria-label="Held-out observed versus predicted TPA concentrations"><line x1="60" y1="285" x2="420" y2="285" stroke="#b8c5c8" /><line x1="60" y1="285" x2="60" y2="25" stroke="#b8c5c8" /><line x1="60" y1="285" x2="420" y2="25" stroke="#b8c5c8" strokeDasharray="5 5" />{[0, .25, .5, .75, 1].map(t => <g key={t}><text x={60 + 360*t} y="307" textAnchor="middle">{number(max*t, 0)}</text><text x="50" y={290 - 260*t} textAnchor="end">{number(max*t, 0)}</text></g>)}{points.map((p, i) => <circle key={i} cx={60+p.observed_um/max*360} cy={285-p.predicted_um/max*260} r="5" fill="#087f78" opacity=".65"><title>Protein {p.protein_id}: observed {number(p.observed_um)}, predicted {number(p.predicted_um)} µM</title></circle>)}<text x="235" y="335" textAnchor="middle">Observed TPA (µM)</text><text x="15" y="160" transform="rotate(-90 15 160)" textAnchor="middle">Predicted TPA (µM)</text></svg></div><div className="research-panel"><h2>How to interpret this run</h2><p>{m?.r2 < 0 ? 'The negative R² indicates weak performance on the held-out measurements.' : 'Interpret these metrics alongside the sample size, target range and split design.'}</p><p>Exact sequences were grouped across train, validation and test. Related proteins and publications may still cross those groups.</p><p>The next evaluation should use stronger family and study splits, matched baselines, and image ablation experiments.</p><div className="research-alert">Mutation effects, calibrated confidence and joint activity–stability improvements have not been established by this checkpoint.</div></div></div></div>
}

function DataSystem({ health }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => { request('/workspace/data').then(setData).catch(e => setError(e.message)) }, [])
  return <div className="research-page"><p className="eyebrow">04 / Research provenance</p><h1>Know what powers the result.</h1><p className="page-lede">Training inputs, scientific scope, and local services in one place.</p>{error && <p role="alert">{error}</p>}<div className="evaluation-grid"><div className="research-panel"><h2>Data readiness</h2><DataRow name="PET activity" value={`${number(data?.activity?.records, 0)} measurements · ${number(data?.activity?.proteins, 0)} proteins`} /><DataRow name="Sequence-linked thermostability" value={data?.stability ? `${number(data.stability.records, 0)} Tm records · ${number(data.stability.protein_groups, 0)} protein groups` : 'Collection pending'} /><DataRow name="Structure images" value={data?.structures ? `${data.structures.matched_proteins} verified protein mappings` : 'Preparation pending'} /><p>Activity concentrations and melting temperatures are separate targets. Structure mappings are checked against each protein sequence before images enter training.</p><a href="https://loschmidt.chemi.muni.cz/fireprotdb/api-docs/" target="_blank" rel="noreferrer">FireProtDB source documentation ↗</a></div><div className="research-panel"><h2>System health</h2><DataRow name="Neural checkpoint" value={health?.activity?.status?.replaceAll('_', ' ') || 'Connecting'} /><DataRow name="Ollama" value={health?.ollama?.status || 'Connecting'} /><DataRow name="Local language model" value={health?.ollama?.model || 'llama3.2:1b'} /><p>{health?.ollama?.detail}</p><p>Use <code>Start-Research.cmd</code> in the Enzyme_ML_new folder to start both research engines. Optional LLM explanations need their own local model service.</p></div></div><div className="research-panel research-roadmap"><h2>From paper to project</h2><div><p><strong>Protein representation</strong>ESM-2 residue embeddings and verified structural views.</p><p><strong>Learned objectives</strong>Activity and Tm models with independent evaluation.</p><p><strong>Directed evolution</strong>Bounded mutation proposals and parent retention.</p><p><strong>Explain & validate</strong>Model sensitivity, source evidence and local narration.</p></div><p>The paper's protein illustrations are not training images. Image fusion and joint optimization are project extensions that require their own evidence.</p></div></div>
}

function DataRow({ name, value }) { return <div className="data-row"><span>{name}</span><strong>{value}</strong></div> }

function Attribution({ proteinId, conditions }) {
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function calculate() {
    setBusy(true); setError('')
    try { setResult(await request('/neural/attribution', { ...conditions, protein_id: proteinId })) }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  return <div><button className="text-button" onClick={calculate} disabled={busy}>{busy ? 'Calculating sensitivity…' : 'Calculate residue attribution'}</button>{error && <p role="alert">{error}</p>}{result && <><p><strong>{result.method}</strong></p><p>{result.residues.map(r => `${r.residue}${r.position}: ${number(r.attribution, 4)}`).join(' · ')}</p><p>{result.note}</p></>}</div>
}
