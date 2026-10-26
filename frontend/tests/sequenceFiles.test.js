import assert from 'node:assert/strict'
import test from 'node:test'
import { compareSequence, isFasta, parseFasta } from '../src/sequenceFiles.js'

test('wrapped lowercase FASTA and BOM preserve one-based substitutions', () => {
  const imported = parseFasta('\uFEFF>example\r\nacde\r\nfghikl\r\n')
  assert.equal(imported.sequence, 'ACDEFGHIKL')
  assert.deepEqual(compareSequence('ACAEFGHIKM', imported.sequence), ['A3D', 'M10L'])
  assert.deepEqual(compareSequence(imported.sequence, imported.sequence), [])
})
test('multiple records never silently merge into one protein', () => {
  assert.throws(() => parseFasta('>one\nACD\n>two\nEFG'), /multiple sequences/)
})
test('empty, ambiguous, gapped and coordinate inputs are rejected', () => {
  for (const text of ['>empty', '>gap\nAC-D', '>ambiguous\nACDX', '>stop\nACD*', 'ATOM      1  CA']) {
    assert.throws(() => parseFasta(text))
  }
})
test('insertions and deletions cannot shift mutation numbering silently', () => {
  assert.throws(() => compareSequence('ACDEFG', 'ACEFG'), /require alignment/)
})
test('unsupported size and variant count are explicit', () => {
  assert.throws(() => parseFasta('>large\n' + 'A'.repeat(100_000)), /100 KB/)
  assert.throws(() => compareSequence('A'.repeat(21), 'C'.repeat(21)), /up to 20/)
})
test('FASTA content is recognized even when renamed PDB', () => {
  assert.equal(isFasta('\ufeff >variant\nACDE', 'renamed.pdb'), true)
  assert.equal(isFasta('ACDE', 'sequence.FASTA'), true)
  assert.equal(isFasta('ATOM      1  CA', 'reference.pdb'), false)
})
