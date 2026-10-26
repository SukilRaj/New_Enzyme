"""One persistent application shell, with isolated existing module controllers."""
from pathlib import Path
from flask import Response, request, send_from_directory

STATIC=Path(__file__).resolve().parent/'static'


def render(module,filename=None,html=None):
    if request.args.get('embedded')!='1':
        return send_from_directory(STATIC,'workspace.html')
    html=html if html is not None else (STATIC/filename).read_text(encoding='utf-8')
    html=html.replace('<body>',f'<body data-workspace-module="{module}">')
    html=html.replace('</head>','<link rel="stylesheet" href="/static/embedded.css"></head>')
    html=html.replace('</body>','<script src="/static/embedded.js"></script></body>')
    return Response(html,mimetype='text/html',headers={'Cache-Control':'no-store'})
