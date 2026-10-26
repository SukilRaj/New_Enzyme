// Display interpolation only: original coordinates remain unchanged for mapping/distances.
export function smoothBackbone(residues, smooth = true) {
  const segments = []
  const connected = (a, b) => a && b && Math.hypot(...a.xyz.map((v, k) => v - b.xyz[k])) <= 4.5
  for (let i = 0; i < residues.length - 1; i++) {
    if (!connected(residues[i], residues[i + 1])) continue
    const p1 = residues[i].xyz, p2 = residues[i + 1].xyz
    const p0 = connected(residues[i - 1], residues[i]) ? residues[i - 1].xyz : p1
    const p3 = connected(residues[i + 1], residues[i + 2]) ? residues[i + 2].xyz : p2
    const steps = smooth ? 6 : 1
    let previous = p1
    for (let step = 1; step <= steps; step++) {
      const t = step / steps
      const point = smooth ? p1.map((v, k) => .5 * ((2 * v) + (-p0[k] + p2[k]) * t +
        (2 * p0[k] - 5 * v + 4 * p2[k] - p3[k]) * t * t +
        (-p0[k] + 3 * v - 3 * p2[k] + p3[k]) * t * t * t)) : p2
      segments.push({ from: previous, to: point, fraction: (i + t / 2) / Math.max(1, residues.length - 1) })
      previous = point
    }
  }
  return segments
}
