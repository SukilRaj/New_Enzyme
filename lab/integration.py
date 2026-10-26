"""Local integration: preserve engine environments, units and artifact provenance."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

from flask import Blueprint, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
PET_ROOT = Path(os.environ.get('ENZAIME_ROOT', r'D:\Final_Year_Project\EnzAIme_ML'))
PET_URL = 'http://127.0.0.1:8766'
SAVED = ROOT / 'research_runs'
bp = Blueprint('research', __name__)
opener = build_opener(ProxyHandler({}))
GET_PATH = re.compile(r'(?:health|workspace/(?:status|evaluation|data)|neural/(?:status|llm/status)|mutation-proteins|mutations/[A-Za-z0-9_-]+/evidence|stability/evidence/[A-Za-z0-9_-]+|structures/catalog/[0-9]+(?:/image)?)\Z')
POST_PATHS = {'neural/recommend', 'neural/attribution', 'neural/explain', 'mutations/preview', 'structures/inspect'}


def frontend_root():
    local = ROOT / 'frontend'
    return local if (local / 'dist/index.html').is_file() else PET_ROOT / 'frontend'


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def pet_call(path, body=None, timeout=120):
    headers = {'X-Integration-Key': os.environ.get('ENZAIME_BRIDGE_KEY', '')}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    req = Request(PET_URL + '/api/' + path, data=body, headers=headers)
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.read(), response.status, response.headers.get('Content-Type', 'application/json')
    except HTTPError as error:
        return error.read(), error.code, error.headers.get('Content-Type', 'application/json')


def snapshot():
    manifest = read_json(PET_ROOT / 'models/neural/manifest.json', {})
    heldout = read_json(PET_ROOT / 'models/neural/held_out_predictions.json', [])
    r2 = None
    if heldout:
        observed = [r['observed_um'] for r in heldout]
        mean = sum(observed) / len(observed)
        variance = sum((v - mean) ** 2 for v in observed)
        r2 = 1 - sum((r['observed_um'] - r['predicted_um']) ** 2 for r in heldout) / variance if variance else None
    data_path = PET_ROOT / 'data/processed/tpa_pet_ml_confirmed.csv'
    rows = []
    if data_path.is_file():
        with data_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    return {'source_available': PET_ROOT.is_dir(), 'source_root': str(PET_ROOT),
            'activity_records': len(rows), 'activity_proteins': len({r['protein_id'] for r in rows}),
            'training_rows': manifest.get('rows_used'), 'encoder': manifest.get('encoder'),
            'activity_r2': r2, 'test_metrics': manifest.get('test_metrics', {}),
            'dataset_sha256': digest(data_path),
            'checkpoint_sha256': digest(PET_ROOT / 'models/neural/weights.pt'),
            'image_trained': manifest.get('image_trained', False),
            'stability_checkpoint_present': all((PET_ROOT / f'models/stability/{name}').is_file() for name in ('manifest.json', 'weights.pt')),
            'stability': read_json(PET_ROOT / 'data/stability/manifest.json', {}),
            'structures': read_json(PET_ROOT / 'data/structure_images/manifest.json', {})}


@bp.get('/api/research/overview')
def overview():
    status = 'offline'
    try:
        _, code, _ = pet_call('health', timeout=2)
        if code == 200:
            status = 'online'
    except (URLError, OSError, TimeoutError):
        pass
    return jsonify(pet=snapshot(), pet_service=status,
                   integration='EnzAIme × Helix', version='1.0',
                   scientific_scope='Two distinct tasks: PET TPA ranking and avGFP evolution. Scores are not interchangeable.')


def records():
    result = []
    for path in (ROOT / 'runs').glob('*/status.json'):
        run = read_json(path)
        if not run:
            continue
        result.append({'id': run['id'], 'domain': 'avGFP', 'unit': 'dataset fitness',
                       'created': run.get('created', 0), 'status': run['status'],
                       'config': run.get('config'), 'metrics': run.get('metrics'),
                       'artifact_url': f"/api/runs/{run['id']}/download/manifest.json" if (path.parent / 'manifest.json').exists() else None})
    for path in SAVED.glob('pet-*.json'):
        run = read_json(path)
        if run:
            result.append({k: v for k, v in run.items() if k != 'result'})
    for path in (ROOT / 'analysis_runs').glob('analysis-*.json'):
        run = read_json(path)
        if run:
            result.append({k:run.get(k) for k in ('id','domain','unit','created','status','config','artifact_url','dataset_sha256','model_sha256')})
    return sorted(result, key=lambda item: item['created'], reverse=True)


@bp.get('/api/research/runs')
def runs():
    return jsonify(records())


@bp.get('/api/research/export')
def export():
    # No tokens, credentials or full sequences in the workspace index export.
    payload = {'schema_version': 1, 'exported_at': time.time(), 'evidence': snapshot(), 'experiments': records(),
               'interpretation': 'PET metrics use micromolar TPA; GFP metrics use dataset fitness. Never aggregate these tasks into one accuracy score.'}
    return Response(json.dumps(payload, indent=2, allow_nan=False), mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=research-evidence.json'})


@bp.get('/api/research/pet-runs/<rid>')
def pet_run(rid):
    if not re.fullmatch(r'pet-[0-9]+-[a-f0-9]{12}', rid):
        return jsonify(error='Unknown run'), 404
    return send_from_directory(SAVED, rid + '.json', as_attachment=True)


@bp.get('/research-guide')
def guide():
    return send_from_directory(ROOT / 'docs', 'PROJECT_COMPARISON_AND_INTEGRATION.md', as_attachment=True)


def register(app, token):
    app.register_blueprint(bp)

    @app.get('/pet')
    def pet_frontend():
        from workspace import render
        if request.args.get('embedded') != '1':
            return render('pet')
        path = frontend_root() / 'dist/index.html'
        if not path.is_file():
            return 'PET frontend is unavailable. Set ENZAIME_ROOT and build that project frontend.', 503
        # The original bundle uses /api; only allowlisted PET routes are bridged.
        bootstrap = '<script>const originalFetch=window.fetch.bind(window);window.fetch=(input,opts={})=>{const u=new URL(input instanceof Request?input.url:input,location.href);const method=(opts.method||(input instanceof Request?input.method:"GET")).toUpperCase();if(u.origin===location.origin&&u.pathname.startsWith("/api/")&&method==="POST"){const headers=new Headers(opts.headers||(input instanceof Request?input.headers:undefined));headers.set("X-Lab-Token",' + json.dumps(token) + ');opts={...opts,headers};}return originalFetch(input,opts);};</script>'
        html = path.read_text(encoding='utf-8').replace('<head>', '<head>' + bootstrap)
        return render('pet',html=html)

    @app.get('/assets/<path:name>')
    def assets(name):
        return send_from_directory(frontend_root() / 'dist/assets', name)

    @app.get('/favicon.svg')
    def favicon():
        return send_from_directory(frontend_root() / 'public', 'favicon.svg')

    @app.route('/api/<path:path>', methods=['GET', 'POST'])
    def proxy(path):
        allowed = GET_PATH.fullmatch(path) if request.method == 'GET' else path in POST_PATHS
        if not allowed:
            return jsonify(detail='Unknown integrated endpoint'), 404
        if path == 'structures/inspect' and request.method == 'POST':
            payload = request.get_json(silent=True)
            text = payload.get('pdb_text') if isinstance(payload, dict) else None
            if isinstance(text, str) and text.lstrip('\ufeff \t\r\n').startswith('>'):
                return jsonify(detail='FASTA contains an amino-acid sequence, not 3D coordinates. Use Import a protein FASTA, then Inspect edits on reference structure, or upload a PDB coordinate file.'), 422
        try:
            body = request.get_data() if request.method == 'POST' else None
            data, code, content_type = pet_call(path, body)
        except (URLError, OSError, TimeoutError) as error:
            return jsonify(detail='PET service unavailable. Launch Start-Research.cmd. The GFP workspace remains available.', error=type(error).__name__), 503
        try:
            if path == 'neural/recommend' and code == 200:
                result = json.loads(data)
                now = time.time()
                rid = f'pet-{int(now)}-{secrets.token_hex(6)}'
                evidence = snapshot()
                saved = dict(id=rid, created=now, domain='PET', unit='µM TPA', status='completed',
                             config=request.get_json(), metrics=result.get('test_metrics'), result=result,
                             dataset_sha256=evidence['dataset_sha256'], checkpoint_sha256=evidence['checkpoint_sha256'],
                             artifact_url=f'/api/research/pet-runs/{rid}')
                SAVED.mkdir(exist_ok=True)
                temporary = SAVED / (rid + '.tmp')
                temporary.write_text(json.dumps(saved, indent=2, allow_nan=False), encoding='utf-8')
                temporary.replace(SAVED / (rid + '.json'))
                result['research_run_id'] = rid
                data = json.dumps(result, allow_nan=False).encode()
            return Response(data, status=code, content_type=content_type)
        except (OSError, ValueError) as error:
            return jsonify(detail='The PET request completed but its result could not be saved or decoded. Check the project storage and server logs.', error=type(error).__name__), 500

