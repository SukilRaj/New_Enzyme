"""Read-only preflight for the linked EnzAIme and Helix local application."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Also inspect the running local services.')
    parser.add_argument('--json', action='store_true', help='Print the preflight as JSON.')
    args = parser.parse_args()
    pet = Path(os.environ.get('ENZAIME_ROOT', r'D:\Final_Year_Project\EnzAIme_ML'))
    pet_python = Path(os.environ.get('ENZAIME_PYTHON', str(pet / '.venv-1/Scripts/python.exe')))
    required = {
        'Helix interpreter': ROOT / '.venv/Scripts/python.exe',
        'PET interpreter': pet_python,
        'PET application': pet / 'backend/app/workspace_app.py',
        'Built PET frontend': ROOT / 'frontend/dist/index.html' if (ROOT / 'frontend/dist/index.html').is_file() else pet / 'frontend/dist/index.html',
        'PET checkpoint': pet / 'models/neural/weights.pt',
        'PET metadata': pet / 'models/neural/manifest.json',
        'PET measurements': pet / 'data/processed/tpa_pet_ml_confirmed.csv',
        'PET sequence features': pet / 'data/neural_features/index.json',
        'Structure catalog': pet / 'data/structure_images/manifest.json',
        'Condition analysis model': ROOT / 'models/condition_analysis/model.pkl',
        'Training study pointer': ROOT / 'models/assay_v2/latest.json',
        'Shared interface': ROOT / 'lab/static/workspace.html',
    }
    checks = [{'name': name, 'ok': path.is_file(), 'path': str(path)} for name, path in required.items()]
    for name in ('numpy', 'scipy', 'sklearn', 'pandas', 'flask', 'torch', 'transformers'):
        checks.append({'name': f'Helix package {name}', 'ok': importlib.util.find_spec(name) is not None})
    if pet_python.is_file():
        # PET inference uses cached ESM embeddings; transformers is not a serving dependency.
        code = "import importlib.util,json; print(json.dumps({m:importlib.util.find_spec(m) is not None for m in ('fastapi','uvicorn','torch','numpy')}))"
        try:
            result = subprocess.run([str(pet_python), '-c', code], capture_output=True, text=True, timeout=30)
            packages = json.loads(result.stdout) if result.returncode == 0 else {}
            checks.append({'name': 'PET Python packages', 'ok': bool(packages) and all(packages.values()), 'packages': packages})
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            checks.append({'name': 'PET Python packages', 'ok': False, 'detail': str(error)})
    if args.live:
        opener = build_opener(ProxyHandler({}))
        for path in ('/api/research/overview', '/api/training/status'):
            try:
                with opener.open('http://127.0.0.1:8765' + path, timeout=8) as response:
                    payload = json.load(response)
                ok = payload.get('pet_service') == 'online' if 'research' in path else payload.get('status') == 'completed'
                checks.append({'name': path, 'ok': ok})
            except (OSError, ValueError) as error:
                checks.append({'name': path, 'ok': False, 'detail': str(error)})
    result = {'ready': all(c['ok'] for c in checks), 'root': str(ROOT), 'pet_root': str(pet),
              'url': 'http://127.0.0.1:8765/', 'checks': checks,
              'note': 'Linked installation: both project folders are required. This checks software readiness, not scientific accuracy.'}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for c in checks:
            print(f"{'OK' if c['ok'] else 'MISSING'}  {c['name']}")
        print('\nReady. Run .\\Start-Research.cmd' if result['ready'] else '\nResolve the missing items above before starting.')
        print(result['note'])
    return 0 if result['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
