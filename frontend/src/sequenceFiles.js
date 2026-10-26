// FASTA stores sequence letters; it never supplies three-dimensional coordinates.
export function isFasta(text, filename = '') {
  return /^\s*>/.test(text) || /\.(fasta|fa|faa|fas|fna)$/i.test(filename)
}

export function parseFasta(text) {
  if (text.length > 100_000) throw new Error('Choose a single-protein FASTA smaller than 100 KB.')
  const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/)
  if (!lines[0]?.startsWith('>')) throw new Error('A FASTA file must start with a > header followed by amino-acid letters.')
  if (lines.slice(1).some(line => line.trim().startsWith('>'))) throw new Error('Import one FASTA record at a time. This file contains multiple sequences.')
  const sequence = lines.slice(1).join('').replace(/\s/g, '').toUpperCase()
  if (!sequence) throw new Error('The FASTA record has no sequence.')
  if (!/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(sequence)) throw new Error('Use the 20 standard amino-acid letters. Gaps, stop symbols and ambiguous residues cannot be mapped as substitutions.')
  return { header: lines[0].slice(1).trim(), sequence }
}

export function compareSequence(reference, imported) {
  if (reference.length !== imported.length) throw new Error(`The imported sequence has ${imported.length} residues; this reference has ${reference.length}. Select the matching reference. Insertions and deletions require alignment and are not mapped by this substitution tool.`)
  const changes = [...reference].flatMap((from, i) => from === imported[i] ? [] : [`${from}${i + 1}${imported[i]}`])
  if (changes.length > 20) throw new Error(`Found ${changes.length} differences. This tool validates up to 20 substitutions; check that you selected the correct reference protein.`)
  return changes
}
