import json
import importlib.util
from pathlib import Path
import secrets
import threading
import time
import traceback
import webbrowser
from flask import Flask, jsonify, request, send_from_directory
from engine import ROOT, inventory, run, validate
from workspace import render as render_workspace

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16384
RUNS = ROOT/'runs'
RUNS.mkdir(exist_ok=True)
lock = threading.Lock()
active = None
cancellations = {}
token = secrets.token_urlsafe(32)


@app.before_request
def local_only():
    if request.path == '/api/structures/inspect':
        request.max_content_length = 2100000
    if request.host not in ('127.0.0.1:8765', 'localhost:8765'):
        return jsonify(error='Localhost access only'), 403
    if request.method == 'POST' and request.headers.get('X-Lab-Token') != token:
        return jsonify(error='Refresh the application and try again'), 403


@app.get('/')
def home():
    return render_workspace('overview', 'research.html')


@app.get('/evolution')
def evolution():
    return render_workspace('evolution', 'index.html')


@app.get('/api/overview')
def overview():
    return jsonify(**inventory(), token=token)


def read_run(folder):
    return json.loads((folder/'status.json').read_text())


def save_status(folder, state):
    tmp = folder/'status.tmp'
    tmp.write_text(json.dumps(state, allow_nan=False))
    tmp.replace(folder/'status.json')


@app.get('/api/runs')
def list_runs():
    with lock:
        return jsonify([read_run(p) for p in sorted(RUNS.iterdir(), reverse=True) if (p/'status.json').exists()])


@app.post('/api/runs')
def start():
    global active
    try:
        config = validate(request.get_json())
        if config['engine'] == 'esm' and not all(importlib.util.find_spec(p) for p in ('torch','transformers','polyleven')):
            raise ValueError('ESM dependencies are not installed. Run Setup-Helix.ps1 -IncludeESM, then restart the application.')
    except (ValueError, TypeError, AttributeError) as error:
        return jsonify(error=str(error)), 400
    with lock:
        if active:
            return jsonify(error='An experiment is already running.'), 409
        rid = time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3)
        folder = RUNS/rid
        folder.mkdir()
        state = dict(id=rid, config=config, status='running', stage='Queued', progress=0,
                     created=time.time(), log=[], metrics=None, history=[])
        save_status(folder, state)
        event = threading.Event()
        cancellations[rid] = event
        active = rid
    def update(stage, progress, message, **extra):
        with lock:
            state.update(stage=stage, progress=round(progress,1), **extra)
            state['log'] = (state['log']+[message])[-100:]
            save_status(folder, state)
    def worker():
        global active
        try:
            result = run(config, folder, update, event.is_set)
            with lock:
                state.update(status='completed', stage='Complete', progress=100, **result)
        except InterruptedError:
            with lock:
                state.update(status='cancelled', stage='Cancelled')
        except Exception as error:
            (folder/'error.log').write_text(traceback.format_exc())
            with lock:
                state.update(status='failed', stage='Failed', error=str(error))
        finally:
            with lock:
                save_status(folder, state)
                active = None
                cancellations.pop(rid, None)
    threading.Thread(target=worker, daemon=True).start()
    return jsonify(id=rid), 202


@app.post('/api/runs/<rid>/cancel')
def cancel(rid):
    with lock:
        if rid not in cancellations:
            return jsonify(error='Run is not active'), 409
        cancellations[rid].set()
    return jsonify(ok=True)


@app.get('/api/runs/<rid>/download/<name>')
def download(rid, name):
    if '/' in rid or '\\' in rid or rid.startswith('.') or name not in ('candidates.csv','holdout.csv','manifest.json','result.json','split.json','error.log'):
        return jsonify(error='Unknown artifact'), 404
    return send_from_directory(RUNS/rid, name, as_attachment=True)


from integration import register
register(app, token)
from analysis_api import bp as analysis_blueprint
app.register_blueprint(analysis_blueprint)
from training_api import bp as training_blueprint
app.register_blueprint(training_blueprint)


if __name__ == '__main__':
    # Persisted running jobs cannot survive process exit.
    for folder in RUNS.iterdir():
        if (folder/'status.json').exists():
            state = read_run(folder)
            if state['status'] == 'running':
                state.update(status='interrupted', stage='Interrupted by server shutdown')
                save_status(folder, state)
    import sys
    if '--no-browser' not in sys.argv:
        threading.Timer(1, lambda: webbrowser.open('http://127.0.0.1:8765')).start()
    app.run(host='127.0.0.1', port=8765, debug=False)
