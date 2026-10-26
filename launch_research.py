"""Supervise both local engines. Ctrl+C stops only processes started here."""
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import webbrowser
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parent


def main():
    pet_root = Path(os.environ.get('ENZAIME_ROOT', r'D:\Final_Year_Project\EnzAIme_ML'))
    pet_python = Path(os.environ.get('ENZAIME_PYTHON', str(pet_root / '.venv-1/Scripts/python.exe')))
    python = ROOT / '.venv/Scripts/python.exe'
    frontend = ROOT / 'frontend/dist/index.html'
    if not frontend.is_file():
        frontend = pet_root / 'frontend/dist/index.html'
    for path in (python, pet_python, frontend, pet_root / 'backend/app/workspace_app.py'):
        if not path.exists():
            raise SystemExit(f'Missing required file: {path}. See docs/PROJECT_COMPARISON_AND_INTEGRATION.md.')
    for port in (8765, 8766):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                raise SystemExit(f'Port {port} is occupied or unavailable. Stop the previous project terminal with Ctrl+C, then retry.')
    env = dict(os.environ, ENZAIME_ROOT=str(pet_root), ENZAIME_BRIDGE_KEY=secrets.token_urlsafe(32), PYTHONDONTWRITEBYTECODE='1')
    children = []
    opener = build_opener(ProxyHandler({}))
    try:
        for executable, script in ((pet_python, 'lab/pet_worker.py'), (python, 'lab/server.py')):
            children.append(subprocess.Popen([str(executable), str(ROOT / script), '--no-browser'], cwd=ROOT, env=env))
        print('Starting both engines. Open http://127.0.0.1:8765/ when ready. Keep this terminal open.', flush=True)
        for _ in range(90):
            if any(child.poll() is not None for child in children):
                raise RuntimeError('An engine exited during startup. Read its error above.')
            try:
                with opener.open(Request('http://127.0.0.1:8766/api/health', headers={'X-Integration-Key': env['ENZAIME_BRIDGE_KEY']}), timeout=1) as response:
                    response.read()
                with opener.open('http://127.0.0.1:8765/api/research/overview', timeout=2) as response:
                    response.read()
                print('EnzAIme × Helix is ready: http://127.0.0.1:8765/', flush=True)
                if '--no-browser' not in sys.argv:
                    webbrowser.open('http://127.0.0.1:8765/')
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('Startup timed out. Inspect the engine errors above.')
        while all(child.poll() is None for child in children):
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nStopping the integrated workspace.', flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                if os.name == 'nt':
                    # A Windows venv redirector may have a base-Python child.
                    # Stop the owned tree, not just its redirector process.
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    child.terminate()
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
