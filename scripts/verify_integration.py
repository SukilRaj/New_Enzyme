"""Exercise the running gateway with real models; save a compact verification report."""
import json
from pathlib import Path
import time
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
opener = build_opener(ProxyHandler({}))
BASE = 'http://127.0.0.1:8765'


def call(path, body=None, token=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['X-Lab-Token'] = token
    with opener.open(Request(BASE + path, data=json.dumps(body).encode() if body is not None else None, headers=headers), timeout=120) as response:
        return json.load(response)


def main():
    token = call('/api/overview')['token']
    overview = call('/api/research/overview')
    assert overview['pet_service'] == 'online'
    status = call('/api/neural/status')
    assert status['status'] == 'research_checkpoint_loaded', status
    conditions = dict(temperature=40, ph=8, duration=48, top_n=3, objective='activity', explain=False)
    ranked = call('/api/neural/recommend', conditions, token)
    assert len(ranked['recommendations']) == 3
    rid = ranked['research_run_id']
    saved = call('/api/research/pet-runs/' + rid)
    assert saved['domain'] == 'PET' and saved['unit'] == 'µM TPA'
    assert saved['result']['recommendations'] == ranked['recommendations']
    structure = call('/api/structures/catalog/1')
    assert structure['inspection']['coverage'] == 1
    attribution = call('/api/neural/attribution', dict(conditions, protein_id=ranked['recommendations'][0]['protein_id']), token)
    assert len(attribution['residues']) > 0
    catalog = call('/api/mutation-proteins')['proteins']
    protein = catalog[0]
    old = protein['sequence'][0]
    new = 'A' if old != 'A' else 'G'
    preview = call('/api/mutations/preview', {'protein_id': protein['protein_id'], 'substitutions': f'{old}1{new}'}, token)
    registry = call('/api/research/runs')
    assert any(r['id'] == rid for r in registry)
    report = {'verified_at': time.time(), 'pet_run_id': rid, 'pet_status': status,
              'recommendations': len(ranked['recommendations']), 'structure_coverage': structure['inspection']['coverage'],
              'attribution_residues': len(attribution['residues']), 'mutation_preview_returned': bool(preview),
              'registry_domains': sorted({r['domain'] for r in registry}),
              'dataset_sha256': saved['dataset_sha256'], 'checkpoint_sha256': saved['checkpoint_sha256']}
    folder = ROOT / 'reports'
    folder.mkdir(exist_ok=True)
    (folder / 'integration_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
