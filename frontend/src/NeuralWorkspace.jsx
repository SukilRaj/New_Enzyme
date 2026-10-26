import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_NEURAL_API_BASE || 'http://127.0.0.1:8001/api/neural'

export default function NeuralWorkspace() {
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [conditions, setConditions] = useState({ temperature: 40, ph: 8, duration: 48 })
  const [explain, setExplain] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    let timer
    async function checkStatus() {
      try {
        const response = await fetch(`${API}/status`, { signal: controller.signal })
        if (!response.ok) throw new Error('Status unavailable')
        const nextStatus = await response.json()
        if (!controller.signal.aborted) setStatus(nextStatus)
      } catch (error) {
        if (!controller.signal.aborted) setStatus({ status: 'service_offline', detail: error.message })
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(checkStatus, 5000)
      }
    }
    checkStatus()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [])
  async function submit(event) {
    event.preventDefault()
    setBusy(true); setError(''); setResult(null)
    try {
      const response = await fetch(`${API}/recommend`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...Object.fromEntries(Object.entries(conditions).map(([k, v]) => [k, Number(v)])), top_n: 5, explain }) })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the input conditions.')
      setResult(data)
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  return <section id="neural" className="workspace-section" aria-labelledby="neural-heading">
    <div className="section-heading"><span className="section-number">07</span><div><p className="overline">Paper-inspired neural research</p><h2 id="neural-heading">Multimodal recommendations</h2></div></div>
    <p>Combine protein sequence representations, optional structural images, and assay conditions. Activity and thermostability are the research objectives; thermostability training is pending.</p>
    <p role="status">{status?.status === 'research_checkpoint_loaded' ? `Research checkpoint loaded · ${status.image_trained ? 'image branch trained' : 'sequence and conditions only'}` : status?.status === 'training_required' ? 'Training required: import the Kaggle checkpoint and matching features.' : status?.status === 'service_offline' ? 'Neural service is offline. Retrying the connection automatically.' : 'Checking neural model…'}</p>
    {status?.test_metrics && <p>Held-out evaluation: MAE {status.test_metrics.mae_um.toFixed(2)} µM{typeof status.test_metrics.r2 === 'number' && ` · R² ${status.test_metrics.r2.toFixed(2)}`}{status.test_metrics.test_proteins && ` · ${status.test_metrics.test_proteins} proteins`}. {status.test_metrics.r2 < 0 ? 'Performance is weak on this split; use rankings only for exploratory research.' : 'Further evaluation is required before relying on rankings.'}</p>}
    <form onSubmit={submit}>
      <div className="conditions-grid">{[['temperature', 'Temperature (°C)'], ['ph', 'pH'], ['duration', 'Duration (hours)']].map(([key, label]) => <div className="number-field" key={key}><label htmlFor={`neural-${key}`}>{label}</label><input id={`neural-${key}`} type="number" step="any" required min={key === 'ph' ? 0 : key === 'duration' ? 0.0001 : undefined} max={key === 'ph' ? 14 : undefined} value={conditions[key]} onChange={e => setConditions({ ...conditions, [key]: e.target.value })} /></div>)}</div>
      <label><input type="checkbox" checked={explain} onChange={e => setExplain(e.target.checked)} /> Explain numerical evidence with local Llama 3.2</label>
      <p><button className="primary-button" disabled={busy || status?.status !== 'research_checkpoint_loaded'}>{busy ? 'Scoring candidates…' : 'Rank with neural model'}</button></p>
    </form>
    {error && <p role="alert">{error}</p>}
    {result && <div><p>{result.status.replaceAll('_', ' ')}. Tm predictions are unavailable.</p>{result.recommendations.map(r => <article className="candidate-card" key={r.protein_id}>
      <h3>#{r.rank} {r.protein_name}</h3><p>{r.predicted_activity_um.toFixed(2)} µM predicted TPA · {r.image_used ? 'Image + sequence + conditions' : 'Sequence + conditions'}</p>
      {r.outside_training_conditions && <p>These conditions extend beyond the training range.</p>}
      <p>Highest attention at sequence positions: {r.explanation.attention_positions.join(', ')}.</p><p>{r.explanation.note}</p>
    </article>)}{result.llm_explanation && <p>{result.llm_explanation.text || 'Local LLM explanation unavailable.'} <small>{result.llm_explanation.status}</small></p>}</div>}
  </section>
}
