"""Deterministic scientific images from sequence-matched C-alpha coordinates.

The parser follows the existing EnzAIme structure service's first-model and
highest-occupancy rules. These views render existing structures; they do not fold sequences.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PET_ROOT = Path(os.environ.get('ENZAIME_ROOT', r'D:\Final_Year_Project\EnzAIme_ML'))
OUTPUT = ROOT / 'generated_structures'
AA = dict(zip('ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(), 'ACDEFGHIKLMNPQRSTVWY'))


def catalog():
    csv.field_size_limit(10_000_000)
    with (PET_ROOT / 'data/processed/tpa_pet_ml_confirmed.csv').open(encoding='utf-8-sig', newline='') as stream:
        return {r['protein_id']: r for r in csv.DictReader(stream)}


def coordinates(protein_id):
    if not isinstance(protein_id, str) or not protein_id.isdecimal() or protein_id not in catalog():
        raise ValueError('Select a protein from the catalog.')
    path = PET_ROOT / 'data/structure_images/pdb' / (protein_id + '.pdb')
    if not path.is_file():
        raise ValueError('No verified structure is available for this protein. A sequence alone does not define a validated 3D image.')
    entries = {}
    model_seen = False
    for line in path.read_text().splitlines():
        record = line[:6].strip()
        if record == 'MODEL':
            if model_seen:
                break
            model_seen = True
        if record == 'ENDMDL':
            break
        if record != 'ATOM' or line[12:16].strip() != 'CA':
            continue
        chain, number, insertion = line[21:22], int(line[22:26]), line[26:27]
        amino = AA.get(line[17:20].strip(), 'X')
        xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
        occupancy = float(line[54:60].strip() or 1)
        if not np.isfinite(xyz + [occupancy]).all() or occupancy < 0:
            raise ValueError('Nonfinite or invalid structure coordinates.')
        residues = entries.setdefault(chain, {})
        key = (number, insertion)
        order = (-occupancy, line[16:17].strip())
        old = residues.get(key)
        if old and old[1] != amino:
            raise ValueError('Ambiguous alternate residue identity.')
        if old is None or order < old[0]:
            residues[key] = (order, amino, xyz)
    reference = catalog()[protein_id]['sequence'].strip().upper()
    for values in entries.values():
        if ''.join(v[1] for v in values.values()) == reference:
            xyz = np.asarray([v[2] for v in values.values()], dtype=np.float64)
            return xyz, hashlib.sha256(path.read_bytes()).hexdigest()
    raise ValueError('Structure does not exactly match the complete reference sequence.')


def distance_image(xyz):
    distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    return np.uint8(np.clip(1 - distances / 20.0, 0, 1) * 255), distances


def descriptor(protein_id):
    try:
        xyz, _ = coordinates(protein_id)
    except ValueError:
        return np.zeros(65, dtype=np.float64)
    pixels, _ = distance_image(xyz)
    # Fixed-size block averages, independent of activity labels or image annotations.
    blocks = np.array_split(np.arange(len(pixels)), 8)
    reduced = [float(pixels[np.ix_(a, b)].mean()) / 255 for a in blocks for b in blocks]
    return np.array(reduced + [1.0])


def png(pixels):
    """Encode an 8-bit grayscale PNG without additional imaging dependencies."""
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError('Expected a two-dimensional uint8 image.')
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    height, width = pixels.shape
    raw = b''.join(b'\x00' + row.tobytes() for row in pixels)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 0, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')


def trace_svg(xyz, axes):
    centered = xyz - xyz.mean(0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    for axis in range(3):
        pivot = np.argmax(np.abs(vectors[axis]))
        if vectors[axis, pivot] < 0:
            vectors[axis] *= -1
    points = (centered @ vectors.T)[:, axes]
    scale = 480 / max(float(np.ptp(points, axis=0).max()), 1)
    points = (points - (points.max(0) + points.min(0))/2) * scale + 300
    lines = []
    for i in range(len(points)-1):
        # Break chain segments rather than drawing over missing-coordinate gaps.
        if np.linalg.norm(xyz[i+1]-xyz[i]) > 5.0:
            continue
        hue = 150 + 90 * i / max(1, len(points)-2)
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        lines.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="hsl({hue:.2f} 60% 43%)" stroke-width="3" stroke-linecap="round"/>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600"><rect width="600" height="600" fill="#f4f7f2"/>' + ''.join(lines) + '</svg>'


def generate(protein_id):
    xyz, source_hash = coordinates(protein_id)
    pixels, distances = distance_image(xyz)
    folder = OUTPUT / protein_id
    folder.mkdir(parents=True, exist_ok=True)
    images = {'distance.png': png(pixels), 'contacts.png': png(np.uint8(distances <= 8) * 255)}
    for name, axes in [('front.svg', [0,1]), ('side.svg', [0,2]), ('top.svg', [1,2])]:
        images[name] = trace_svg(xyz, axes).encode()
    for name, content in images.items():
        temporary = folder / (name + '.tmp')
        temporary.write_bytes(content)
        temporary.replace(folder / name)
    protein = catalog()[protein_id]
    manifest = {'protein_id': protein_id, 'protein_name': protein['protein_name'], 'residues': len(xyz),
                'sequence_sha256': hashlib.sha256(protein['sequence'].strip().upper().encode()).hexdigest(),
                'source_pdb_sha256': source_hash, 'source_kind': 'Sequence-verified AlphaFold predicted structure',
                'representation': '20 Å capped C-alpha distance map; 8 Å C-alpha proximity map; three PCA projections',
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'images': [{'name': name, 'url': f'/api/analysis/images/{protein_id}/{name}', 'sha256': hashlib.sha256(content).hexdigest()} for name,content in images.items()],
                'note': 'Coordinate-derived synthetic views, not de novo folding. C-alpha proximity is not an atom-level contact prediction. Structure images are not experimental activity labels.'}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
