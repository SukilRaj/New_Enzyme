"""Exercise the running analysis studio and retain reproducible API evidence."""
import hashlib
import io
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8765'
opener = build_opener(ProxyHandler({}))


def request(path, body=None, token=None, binary=False):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['X-Lab-Token'] = token
    data = json.dumps(body).encode() if body is not None else None
    with opener.open(Request(BASE + path, data=data, headers=headers), timeout=180) as response:
        raw = response.read()
        return raw if binary else json.loads(raw)


def main():
    token = request('/api/overview')['token']
    state = request('/api/analysis/status')
    assert state['ready'] and len(state['proteins']) == 25
    conditions = dict(temperature=40, ph=8, duration=48)
    original = request('/api/neural/recommend', dict(conditions, top_n=3, objective='activity', explain=False), token)
    top = original['recommendations'][0]
    config = dict(conditions, protein_id=top['protein_id'], engine='legacy',
                  temperature_span=5, ph_span=.5, duration_span=12)
    started = time.monotonic()
    result = request('/api/analysis/run', config, token)
    elapsed = time.monotonic() - started
    assert result['scenario_count'] == 27 and len(result['surface']) == 49
    assert len(result['ranking']) == 25
    center = result['selected_protein']['center_prediction_um']
    # Different batch sizes may introduce small float32 rounding differences.
    assert math.isclose(center, top['predicted_activity_um'], rel_tol=1e-5, abs_tol=.001)
    assert all(len(points) == 13 for points in result['sweeps'].values())
    scores = [row['robust_score_um'] for row in result['ranking']]
    assert scores == sorted(scores, reverse=True) and all(math.isfinite(x) for x in scores)
    saved = request(result['artifact_url'])
    assert saved == result

    fusion = request('/api/analysis/run', dict(config, engine='fusion', temperature_span=0, ph_span=0, duration_span=0), token)
    assert fusion['scenario_count'] == 1
    assert math.isclose(fusion['selected_protein']['scenario_std_um'], 0, abs_tol=1e-10)
    assert fusion['selected_protein']['center_prediction_um'] == fusion['selected_protein']['robust_score_um']
    assert fusion['model_sha256'] == state['report']['model_sha256']
    registry = request('/api/research/runs')
    assert any(row['id'] == result['id'] and row['domain'] == 'PET environment' for row in registry)

    pid = next(p['id'] for p in state['proteins'] if p['structure_available'])
    images = request('/api/analysis/images', {'protein_id': pid}, token)
    assert len(images['images']) == 5
    raw = request(f'/api/analysis/images/{pid}/bundle.zip', binary=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert set(archive.namelist()) == {'front.svg', 'side.svg', 'top.svg', 'distance.png', 'contacts.png', 'manifest.json'}
        manifest = json.loads(archive.read('manifest.json'))
        for image in manifest['images']:
            assert hashlib.sha256(archive.read(image['name'])).hexdigest() == image['sha256']
    for image in images['images']:
        assert hashlib.sha256(request(image['url'], binary=True)).hexdigest() == image['sha256']
    try:
        request('/api/analysis/run', dict(config, temperature=101), token)
        raise AssertionError('Out-of-bounds conditions were accepted')
    except HTTPError as error:
        assert error.code == 400

    verification = dict(verified_at=time.time(), legacy_run_id=result['id'], fusion_run_id=fusion['id'],
                        legacy_duration_seconds=round(elapsed, 2), legacy_center_matches_original=True,
                        condition_scenarios=27, heatmap_cells=49, sweep_points_per_axis=13, ranked_proteins=25,
                        structure_protein_id=pid, generated_images=5, image_zip_hashes_verified=True,
                        structures_available=sum(p['structure_available'] for p in state['proteins']),
                        invalid_conditions_rejected=True, registry_round_trip=True,
                        model_sha256=fusion['model_sha256'], dataset_sha256=result['dataset_sha256'])
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/analysis_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
