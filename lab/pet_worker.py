"""Run the existing PET application in its own compatible Python environment."""
import os
import sys
from pathlib import Path

root = Path(os.environ['ENZAIME_ROOT']).resolve()
sys.path.insert(0, str(root))
from backend.app.workspace_app import app as upstream
from pet_extensions import attach
attach(upstream)


async def app(scope, receive, send):
    if scope['type'] == 'http':
        headers = dict(scope.get('headers', []))
        expected = os.environ.get('ENZAIME_BRIDGE_KEY', '')
        import secrets
        if not expected or not secrets.compare_digest(headers.get(b'x-integration-key', b''), expected.encode()):
            await send({'type': 'http.response.start', 'status': 403, 'headers': [(b'content-type', b'application/json')]})
            await send({'type': 'http.response.body', 'body': b'{"detail":"Use the research gateway"}'})
            return
    await upstream(scope, receive, send)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8766)

