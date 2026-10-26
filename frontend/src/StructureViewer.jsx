import { useEffect, useMemo, useRef, useState } from 'react'
import { smoothBackbone } from './structureGeometry'
import './StructureViewer.css'

export default function StructureViewer({ residues, position }) {
  const canvasRef = useRef(null)
  const drag = useRef(null)
  const hitPoints = useRef([])
  const [angles, setAngles] = useState({ yaw: 20, pitch: -15 })
  const [zoom, setZoom] = useState(1)
  const [smooth, setSmooth] = useState(true)
  const [color, setColor] = useState('sequence')
  const [focus, setFocus] = useState(false)
  const [hovered, setHovered] = useState(null)
  const site = residues.find((item) => item.sequence_position != null && item.sequence_position === position)
  const segments = useMemo(() => smoothBackbone(residues, smooth), [residues, smooth])

  useEffect(() => {
    const canvas = canvasRef.current
    function draw() {
      const width = canvas.clientWidth, height = canvas.clientHeight
      if (!width || !height) return
      const ratio = Math.min(window.devicePixelRatio || 1, 3)
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio)
      const ctx = canvas.getContext('2d')
      ctx.scale(ratio, ratio)
      const background = ctx.createLinearGradient(0, 0, width, height)
      background.addColorStop(0, '#f2f7fa'); background.addColorStop(1, '#e7f0f1')
      ctx.fillStyle = background; ctx.fillRect(0, 0, width, height)
      const center = focus && site ? site.xyz : [0, 1, 2].map((axis) => residues.reduce((sum, r) => sum + r.xyz[axis], 0) / residues.length)
      const radius = focus && site ? 13 : Math.max(1, ...residues.map((item) => Math.hypot(...item.xyz.map((value, axis) => value - center[axis]))))
      const scale = Math.min(width, height) * .40 * zoom / radius
      const a = angles.yaw * Math.PI / 180, b = angles.pitch * Math.PI / 180
      function project(xyz) {
        const [x, y, z] = xyz.map((value, axis) => value - center[axis])
        const rx = x * Math.cos(a) + z * Math.sin(a), rz = -x * Math.sin(a) + z * Math.cos(a)
        return { x: width / 2 + rx * scale, y: height / 2 - (y * Math.cos(b) - rz * Math.sin(b)) * scale,
          z: y * Math.sin(b) + rz * Math.cos(b) }
      }
      const projectedSegments = segments.map((s) => ({ ...s, a: project(s.from), b: project(s.to) }))
      projectedSegments.sort((l, r) => (l.a.z + l.b.z) - (r.a.z + r.b.z))
      ctx.lineCap = 'round'; ctx.lineJoin = 'round'
      const tubeWidth = Math.max(3, Math.min(12, Math.min(width, height) / 75)) * Math.sqrt(zoom)
      for (const segment of projectedSegments) {
        const { a: p, b: q } = segment
        const depth = Math.max(0, Math.min(1, .5 + (p.z + q.z) / (4 * radius)))
        const hue = color === 'sequence' ? 230 - 195 * segment.fraction : 174
        const path = (lineWidth, stroke, offset = 0) => {
          ctx.beginPath(); ctx.moveTo(p.x + offset, p.y + offset); ctx.lineTo(q.x + offset, q.y + offset)
          ctx.lineWidth = lineWidth; ctx.strokeStyle = stroke; ctx.stroke()
        }
        path(tubeWidth, `hsl(${hue} 60% ${35 + depth * 24}%)`)
        path(tubeWidth * .30, `hsla(${hue} 85% 87% / .28)`, -.65)
      }
      const points = residues.map((item) => ({ ...item, ...project(item.xyz) }))
      hitPoints.current = points
      if (!smooth) {
        ctx.fillStyle = '#effbff'
        for (const point of points) { ctx.beginPath(); ctx.arc(point.x, point.y, 1.6, 0, Math.PI * 2); ctx.fill() }
      }
      const selected = site && project(site.xyz)
      if (selected) {
        ctx.beginPath(); ctx.arc(selected.x, selected.y, 11, 0, Math.PI * 2)
        ctx.fillStyle = '#e87524'; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = '#fff'; ctx.stroke()
        const label = `${site.residue}${position}  ·  PDB ${site.residue_number}${site.insertion_code}`
        ctx.font = '600 13px system-ui'
        const labelWidth = ctx.measureText(label).width + 20
        const lx = Math.max(8, Math.min(width - labelWidth - 8, selected.x + 18))
        const ly = Math.max(34, Math.min(height - 14, selected.y - 18))
        ctx.fillStyle = '#ffffffed'; ctx.beginPath(); ctx.roundRect(lx, ly - 23, labelWidth, 32, 7); ctx.fill()
        ctx.fillStyle = '#703c12'; ctx.fillText(label, lx + 10, ly - 2)
      }
      ctx.fillStyle = '#486477'; ctx.font = '11px system-ui'
      ctx.fillText(smooth ? 'SMOOTH BACKBONE · DISPLAY INTERPOLATION' : 'C-ALPHA COORDINATE TRACE', 16, 25)
      ctx.fillText(`${residues.length} observed residues${focus ? ' · local focus' : ''}`, 16, height - 17)
    }
    const observer = new ResizeObserver(draw)
    observer.observe(canvas); draw()
    return () => observer.disconnect()
  }, [angles, zoom, smooth, color, focus, residues, segments, site, position])

  function pointerMove(event) {
    const previous = drag.current
    if (previous) {
      const dx = event.clientX - previous.x, dy = event.clientY - previous.y
      setAngles((value) => ({ yaw: value.yaw + dx * .55, pitch: value.pitch + dy * .55 }))
      drag.current = { x: event.clientX, y: event.clientY }
    } else {
      const rect = event.currentTarget.getBoundingClientRect()
      const x = event.clientX - rect.left, y = event.clientY - rect.top
      const closest = hitPoints.current.filter((p) => Math.hypot(p.x - x, p.y - y) < 9).sort((a, b) => b.z - a.z)[0]
      setHovered(closest || null)
    }
  }
  function reset() { setAngles({ yaw: 20, pitch: -15 }); setZoom(1); setFocus(false); setHovered(null) }
  function saveImage() {
    canvasRef.current.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.download = 'enzaime-structure-view.png'; link.href = url
      document.body.appendChild(link); link.click(); link.remove()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }, 'image/png')
  }
  return <div className="structure-viewer">
    <div className="structure-toolbar"><div role="group" aria-label="Backbone representation"><button type="button" aria-pressed={smooth} onClick={() => setSmooth(true)}>Smooth tube</button><button type="button" aria-pressed={!smooth} onClick={() => setSmooth(false)}>C-alpha trace</button></div>
      <label>Color<select value={color} onChange={(event) => setColor(event.target.value)}><option value="sequence">Sequence direction</option><option value="single">Single color</option></select></label></div>
    <canvas ref={canvasRef} className="structure-canvas" tabIndex={0} role="img" aria-label={`Interactive ${smooth ? 'smooth backbone' : 'C-alpha trace'} of ${residues.length} residues. Drag or use arrow keys to rotate; plus and minus to zoom.${site ? ` Selected residue ${site.residue}${position}.` : ''}`}
      onPointerDown={(event) => { drag.current = { x: event.clientX, y: event.clientY }; event.currentTarget.setPointerCapture(event.pointerId) }}
      onPointerMove={pointerMove} onPointerUp={() => { drag.current = null }} onPointerCancel={() => { drag.current = null }} onPointerLeave={() => { setHovered(null) }}
      onKeyDown={(event) => {
        if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '-', '='].includes(event.key)) event.preventDefault()
        if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') setAngles((v) => ({ ...v, yaw: v.yaw + (event.key === 'ArrowLeft' ? -10 : 10) }))
        if (event.key === 'ArrowUp' || event.key === 'ArrowDown') setAngles((v) => ({ ...v, pitch: v.pitch + (event.key === 'ArrowUp' ? -10 : 10) }))
        if (event.key === '+' || event.key === '=') setZoom((v) => Math.min(3, v + .1))
        if (event.key === '-') setZoom((v) => Math.max(.6, v - .1))
      }}>Your browser does not support the structure canvas. Residue mapping and distances remain available below.</canvas>
    <div className="structure-caption">{hovered ? `${hovered.residue} · PDB ${hovered.residue_number}${hovered.insertion_code}${hovered.sequence_position ? ` · Reference ${hovered.sequence_position}` : ' · Reference unmapped'}` : 'Drag to rotate · Hover to inspect residues · Arrow keys also rotate'}</div>
    <div className="structure-controls"><label>Zoom<input type="range" min="0.6" max="3" step="0.05" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /></label><span>{zoom.toFixed(1)}×</span><button type="button" disabled={!site} aria-pressed={focus} onClick={() => { setFocus(!focus); setZoom(1) }}>Focus selected site</button><button type="button" onClick={reset}>Reset view</button><button type="button" onClick={saveImage}>Save image</button></div>
    {color === 'sequence' && <div className="sequence-legend"><span>Observed N terminus</span><i /><span>Observed C terminus</span></div>}
    <p className="field-help">Orange marks the selected coordinate. Smoothing changes only the drawing, not the protein or measured distances. Gaps over 4.5 Å remain disconnected. This is a backbone view, not a secondary-structure assignment or a predicted mutant fold.</p>
  </div>
}
