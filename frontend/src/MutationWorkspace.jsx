import { useEffect, useImperativeHandle, useRef, useState } from 'react'
import './MutationWorkspace.css'
import StructureViewer from './StructureViewer'
import { compareSequence, isFasta, parseFasta } from './sequenceFiles'

function downloadText(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }))
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = name; anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

async function request(base, path, body, signal) {
  const response = await fetch(`${base}${path}`, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
  } : { signal })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof payload.detail === 'string' ? payload.detail
      : payload.detail?.map?.((item) => `${item.loc?.at(-1)}: ${item.msg}`).join('; ')
    throw new Error(detail || 'The service could not complete this request.')
  }
  return payload
}

function message(error) {
  return error instanceof TypeError ? 'Unable to reach the mutation and structure service. Check the backend and retry.' : error.message
}

export default function MutationWorkspace({ apiBase }) {
  const [proteins, setProteins] = useState([])
  const [proteinId, setProteinId] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    request(apiBase, '/mutation-proteins', null, controller.signal)
      .then((data) => { setProteins(data.proteins); setProteinId(data.proteins[0]?.protein_id || '') })
      .catch((err) => { if (err.name !== 'AbortError') setError(message(err)) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [apiBase, attempt])
  const protein = proteins.find((item) => item.protein_id === proteinId)
  return <section className="content-section mutation-workspace" id="mutations" aria-labelledby="mutation-heading">
    <div className="section-heading"><span className="section-number">06</span><div><p className="overline">Mutation & structure</p><h2 id="mutation-heading">Inspect a variant in context</h2></div></div>
    <p className="block-description">Create a sequence variant, inspect published stability evidence, and map a residue onto uploaded coordinates. Mutation effects on PET degradation have not been validated.</p>
    {loading && <p role="status">Loading confirmed proteins…</p>}
    {error && <div className="alert error" role="alert">{error} <button type="button" onClick={() => { setError(''); setLoading(true); setAttempt(attempt + 1) }}>Retry</button></div>}
    {!!proteins.length && <><label htmlFor="mutation-protein">Confirmed reference protein</label><select id="mutation-protein" value={proteinId} onChange={(event) => setProteinId(event.target.value)}>{proteins.map((item) => <option key={item.protein_id} value={item.protein_id}>{item.protein_name} · {item.sequence.length} residues</option>)}</select></>}
    {protein && <ProteinWorkspace key={protein.protein_id} apiBase={apiBase} protein={protein} />}
  </section>
}

function ProteinWorkspace({ apiBase, protein }) {
  const inspectorRef = useRef(null)
  const [importNote, setImportNote] = useState('')
  const [substitutions, setSubstitutions] = useState('')
  const [preview, setPreview] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [evidence, setEvidence] = useState(null)
  const [evidenceError, setEvidenceError] = useState('')
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    request(apiBase, `/mutations/${encodeURIComponent(protein.protein_id)}/evidence`, null, controller.signal)
      .then(setEvidence).catch((err) => { if (err.name !== 'AbortError') setEvidenceError(message(err)) })
    return () => controller.abort()
  }, [apiBase, protein.protein_id, attempt])
  async function submit(event) {
    event.preventDefault(); setImportNote(''); setBusy(true); setError(''); setPreview(null)
    try { setPreview(await request(apiBase, '/mutations/preview', { protein_id: protein.protein_id, substitutions })) }
    catch (err) { setError(message(err)) }
    finally { setBusy(false) }
  }
  async function importFasta(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setError(''); setImportNote(''); setPreview(null); setSubstitutions(''); setBusy(true)
    try {
      if (file.size > 100_000) throw new Error('Choose a single-protein FASTA smaller than 100 KB.')
      const { sequence } = parseFasta(await file.text())
      const changes = compareSequence(protein.sequence, sequence)
      setSubstitutions(changes.join(', '))
      if (changes.length) {
        setPreview(await request(apiBase, '/mutations/preview', { protein_id: protein.protein_id, substitutions: changes.join(', ') }))
      }
      setImportNote(`${file.name}: ${sequence.length} residues; ${changes.length ? `${changes.length} substitutions validated against the selected reference.` : 'exact match to the selected reference; no substitutions.'}`)
    } catch (err) { setError(message(err)) }
    finally { setBusy(false); event.target.value = '' }
  }
  function inspectVariant() {
    inspectorRef.current?.loadReference(preview.changes[0].position)
    document.getElementById('structure-inspector')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  function download() {
    const name = `protein_${protein.protein_id}_${preview.changes.map((change) => change.notation).join('_')}`
    const lines = preview.mutant_sequence.match(/.{1,80}/g).join('\n')
    downloadText(`${name}.fasta`, `>${name} sequence_edit_only\n${lines}\n`)
  }
  return <div className="mutation-stack">
    <div className="mutation-panel"><span className="card-kicker">01 / Sequence variant</span><h3>Validate substitutions</h3>
      <p>Use one-based positions in the reference below. Publication and PDB numbering can differ.</p>
      <details><summary>Reference sequence · {protein.sequence.length} residues</summary><pre className="sequence-text">{protein.sequence}</pre></details>
      <div className="sequence-import" id="fasta-import"><label htmlFor="fasta-file">Import a protein FASTA</label><input id="fasta-file" type="file" accept=".fasta,.fa,.faa,.fas" disabled={busy} onChange={importFasta} /><p>Load your downloaded variant here. Its letters are compared with the selected reference to recover substitutions. FASTA contains a sequence, not 3D coordinates.</p></div>
      {importNote && <p className="mapping-success" role="status">{importNote}</p>}
      <form onSubmit={submit}><label htmlFor="substitutions">Substitutions</label><input id="substitutions" value={substitutions} maxLength={500} placeholder="For example: S160A, R280A" required disabled={busy} onChange={(event) => { setSubstitutions(event.target.value); setPreview(null); setError('') }} /><button className="primary-button" disabled={busy}>{busy ? 'Validating…' : 'Preview variant'}</button></form>
      {error && <p className="alert error" role="alert">{error}</p>}
      {preview && <div className="variant-result" role="status"><strong>{preview.status}</strong><p>{preview.changes.map((change) => change.notation).join(' + ')} · {preview.sequence_length} residues</p><pre className="sequence-text">{[...preview.mutant_sequence].map((letter, index) => preview.changes.some((change) => change.position === index + 1) ? <mark key={index} title={`Position ${index + 1}`}>{letter}</mark> : letter)}</pre><div className="variant-actions"><button type="button" className="primary-button" onClick={inspectVariant}>Inspect edits on reference structure</button><button type="button" className="secondary-button" onClick={download}>Download mutant FASTA</button></div><p>FASTA exports the edited sequence for reuse or a separate folding tool. The reference view uses existing coordinates; it does not predict the mutant fold. No activity or stability score is assigned to this edit.</p></div>}
    </div>
    <div className="mutation-panel"><span className="card-kicker">02 / Source evidence</span><h3>FireProtDB records</h3>
      {!evidence && !evidenceError && <p role="status">Loading evidence…</p>}
      {evidenceError && <p className="alert error" role="alert">{evidenceError} <button onClick={() => { setEvidenceError(''); setAttempt(attempt + 1) }}>Retry evidence</button></p>}
      {evidence && <><p>{evidence.note}</p>{!evidence.records.length && <p className="state-block">{evidence.status === 'unavailable' ? 'Evidence extract unavailable.' : 'No matching records in the local extract. This does not establish that no published evidence exists.'}</p>}
        <div className="mutation-evidence-list">{evidence.records.map((record, index) => <details key={`${record.experiment_id}-${index}`}><summary>{record.substitutions || 'Reference measurement'} <span>Experiment {record.experiment_id}</span></summary>
          <p>{record.residue_check_note} Exact source sequence identity remains unresolved.</p>
          <dl className="evidence-fields"><dt>Accession</dt><dd>{record.uniprot_accession}</dd><dt>Source sequence length</dt><dd>{record.reported_sequence_length || 'Unavailable'}</dd><dt>Method / measure</dt><dd>{record.method || 'Unavailable'} / {record.measure || 'Unavailable'}</dd><dt>Buffer</dt><dd>{record.buffer || 'Unavailable'} {record.buffer_concentration}</dd>{Object.entries(record.measurements).map(([key, value]) => <div key={key}><dt>{key} (raw)</dt><dd>{value}</dd></div>)}</dl>
          <p>{record.measurement_note}</p><p>Source: {record.source_dataset || 'Unspecified'} · {record.doi?.startsWith('10.') ? <a href={`https://doi.org/${encodeURIComponent(record.doi)}`} target="_blank" rel="noreferrer">{record.doi}</a> : 'DOI unavailable'}</p>
          {record.substitutions && record.residue_check_passed && <button type="button" disabled={busy} onClick={() => { setSubstitutions(record.substitutions); setPreview(null); setError('') }}>Copy substitutions for preview</button>}
        </details>)}</div></>}
    </div>
    <StructureWorkspace apiBase={apiBase} protein={protein} preview={preview} inspectorRef={inspectorRef} />
  </div>
}

function StructureWorkspace({ apiBase, protein, preview, inspectorRef }) {
  const [pdbText, setPdbText] = useState('')
  const [fileName, setFileName] = useState('')
  const [chain, setChain] = useState(null)
  const [chains, setChains] = useState([])
  const [position, setPosition] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function loadFile(event) {
    setCachedStructure(null)
    setPdbText(''); setResult(null); setError(''); setChains([]); setChain(null); setFileName('')
    const file = event.target.files?.[0]
    if (!file) return
    if (file.size > 2_000_000) { setError('Choose a PDB file smaller than 2 MB.'); return }
    setBusy(true)
    try {
      const text = await file.text()
      if (isFasta(text, file.name)) throw new Error('This is a FASTA sequence, not a PDB coordinate file. Use “Import a protein FASTA” above, then “Inspect edits on reference structure”. A FASTA file alone cannot show a 3D fold.')
      if (!/^(ATOM {2}|HETATM)/m.test(text)) throw new Error('No PDB atom records found. Choose a .pdb or .ent coordinate file, or load the verified catalog structure. mmCIF is not supported.')
      setPdbText(text); setFileName(file.name)
    }
    catch (err) { setError(err.message || 'The selected file could not be read.') }
    finally { setBusy(false) }
  }
  const [cachedStructure, setCachedStructure] = useState(null)
  async function loadVerifiedStructure(requestedPosition = null) {
    setBusy(true); setError(''); setResult(null); setCachedStructure(null)
    setPdbText(''); setChains([]); setChain(null); setFileName('')
    try {
      const data = await request(apiBase, `/structures/catalog/${encodeURIComponent(protein.protein_id)}`)
      setPdbText(data.pdb_text); setFileName(`${data.provenance.accession} · AlphaFold predicted`)
      const inspection = requestedPosition ? await request(apiBase, '/structures/inspect', { protein_id: protein.protein_id, pdb_text: data.pdb_text, chain_id: data.inspection.chain_id, position: requestedPosition }) : data.inspection
      setResult(inspection); setChains(inspection.chains); setChain(inspection.chain_id)
      setPosition(requestedPosition ? String(requestedPosition) : ''); setCachedStructure(data)
    } catch (err) { setError(message(err)) }
    finally { setBusy(false) }
  }
  useImperativeHandle(inspectorRef, () => ({ loadReference: loadVerifiedStructure }))
  async function inspectVariantPosition(selectedPosition) {
    if (!cachedStructure) return loadVerifiedStructure(selectedPosition)
    setBusy(true); setError(''); setResult(null); setPosition(String(selectedPosition))
    try { setResult(await request(apiBase, '/structures/inspect', { protein_id: protein.protein_id, pdb_text: pdbText, chain_id: chain, position: selectedPosition })) }
    catch (err) { setError(message(err)) }
    finally { setBusy(false) }
  }
  async function inspect(event) {
    event.preventDefault(); setResult(null); setError(''); setBusy(true)
    try {
      const payload = await request(apiBase, '/structures/inspect', { protein_id: protein.protein_id, pdb_text: pdbText, chain_id: chain, position: position ? Number(position) : null })
      setResult(payload); setChains(payload.chains || [])
      if (payload.chain_id !== undefined) setChain(payload.chain_id)
    } catch (err) { setError(message(err)) }
    finally { setBusy(false) }
  }
  return <div className="mutation-panel" id="structure-inspector"><span className="card-kicker">03 / Structural context</span><h3>Map a residue onto coordinates</h3>
    <button type="button" className="secondary-button" disabled={busy} onClick={() => loadVerifiedStructure()}>Load verified catalog structure</button>
    <p>Thirteen catalog proteins have a full sequence-matched AlphaFold structure and structural distance image prepared for the multimodal experiment.</p>
    <p>Upload a PDB file to inspect its first model as a rotatable C-alpha trace. Files are sent to this backend for inspection and are not saved. mmCIF is not supported.</p>
    <p className="file-format-help"><strong>FASTA = amino-acid sequence.</strong> Import it in the sequence panel above. <strong>PDB = 3D atom coordinates.</strong> Load a catalog structure or upload coordinates below. Changing the file extension does not convert a sequence into a structure.</p>
    {preview && <div className="variant-sites"><p>Inspect variant positions on the reference coordinates:</p><div className="variant-actions">{preview.changes.map(change => <button type="button" className="secondary-button" key={change.position} disabled={busy} aria-pressed={!!cachedStructure && Number(position) === change.position} onClick={() => inspectVariantPosition(change.position)}>{change.notation}</button>)}</div><p>These are reference residue positions. The coordinates and amino acids in this view have not been mutated.</p></div>}
    {protein.pdb_ids.length > 0 && <details><summary>Source-listed structures ({protein.pdb_ids.length})</summary><p>These metadata links are not verified matches to this sequence. Download coordinates from an entry, then inspect the chain.</p><div className="pdb-links">{protein.pdb_ids.map((id) => <a key={id} href={`https://www.rcsb.org/structure/${id}`} target="_blank" rel="noreferrer">{id}</a>)}</div></details>}
    <form onSubmit={inspect}><label htmlFor="pdb-file">PDB coordinate file (up to 2 MB)</label><input id="pdb-file" type="file" accept=".pdb,.ent" onChange={loadFile} disabled={busy} />
      {!!chains.length && <><label htmlFor="structure-chain">Chain</label><select id="structure-chain" value={chain ?? '__select__'} disabled={busy} onChange={(event) => { setChain(event.target.value === '__select__' ? null : event.target.value); setResult(null) }}><option value="__select__" disabled>Select a chain</option>{chains.map((item) => <option key={item.chain_id} value={item.chain_id}>{item.chain_id || '(blank chain)'} · {item.residue_count} residues</option>)}</select></>}
      <label htmlFor="structure-position">Reference sequence position (optional)</label><input id="structure-position" type="number" min="1" max={protein.sequence.length} step="1" value={position} disabled={busy} onChange={(event) => { setPosition(event.target.value); setResult(null) }} />
      <button className="secondary-button" disabled={busy || !pdbText}>{busy ? 'Inspecting…' : 'Inspect structure'}</button></form>
    {error && <p className="alert error" role="alert">{error} <a href="#fasta-import" onClick={event => { event.preventDefault(); document.getElementById('fasta-file')?.focus(); document.getElementById('fasta-import')?.scrollIntoView({ behavior: 'smooth', block: 'center' }) }}>Go to FASTA import</a></p>}
    {cachedStructure && <div className="variant-actions"><button type="button" className="secondary-button" onClick={() => downloadText(`protein_${protein.protein_id}_reference_${cachedStructure.provenance.accession}.pdb`, cachedStructure.pdb_text)}>Download reference PDB</button><span className="field-help">Sequence-matched AlphaFold reference; not a predicted mutant.</span></div>}
    {cachedStructure && <details><summary>Structural image prepared for training</summary><img src={`${apiBase}/structures/catalog/${encodeURIComponent(protein.protein_id)}/image`} alt={`C-alpha distance map for ${protein.protein_name}`} style={{ width:'100%', maxWidth:320 }} /><p>Full C-alpha distance matrix, clipped at 20 Å; closer residues appear brighter. Predicted structure: <a href={cachedStructure.provenance.structure_url}>{cachedStructure.provenance.accession}</a>. Image availability does not mean the running model has been trained on it.</p></details>}
    {result && <div aria-live="polite"><p className={result.status === 'mapped' ? 'mapping-success' : 'alert warning'}><strong>{result.status === 'mapped' ? 'Sequence mapped' : result.status === 'select_chain' ? 'Chain selection required' : 'Mapping unresolved'}</strong> · {result.note}</p>
      {result.residues && <><p>{fileName} · Chain {result.chain_id || '(blank)'} · Model {result.model} · {result.residues.length} C-alpha residues</p><StructureViewer key={result.sha256 + result.chain_id + result.requested_position} residues={result.residues} position={result.site?.sequence_position} />
        {result.coverage !== null && <p>Observed coverage: {(result.coverage * 100).toFixed(1)}% · Reference positions {result.reference_start}–{result.reference_end}.</p>}
        {result.site_status === 'unmapped' && <p className="alert warning">Requested position is not mapped in these coordinates.</p>}
        {result.site && <><p><strong>Selected residue:</strong> {result.site.residue}{result.site.sequence_position} → PDB {result.site.residue_number}{result.site.insertion_code}, chain {result.chain_id || '(blank)'}</p><details><summary>C-alpha neighbors within 8 Å ({result.neighbors.length})</summary><p>Distances include adjacent sequence positions and do not represent atom-level contacts.</p><ul>{result.neighbors.map((item) => <li key={item.sequence_position}>{item.residue}{item.sequence_position} · PDB {item.residue_number}{item.insertion_code} · {item.ca_distance_angstrom} Å</li>)}</ul></details></>}
        <p className="field-help">{result.limitations}</p><details><summary>Mapping provenance</summary><p>{result.mapping_method}</p><code className="sequence-text">SHA-256: {result.sha256}</code></details></>}
    </div>}
  </div>
}
