"""Local-only web server. Each browser session has isolated in-memory NPC state."""
import argparse
import json
import threading
import uuid
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from model import load_model
from npc import PRESETS, questions_for, respond

ROOT = Path(__file__).resolve().parent
MODEL, VOCAB, IDF = load_model(ROOT / 'artifacts/intent.pt')
SESSIONS, LOCK = {}, threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, kind='application/json; charset=utf-8', cookie=None):
        payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        if cookie: self.send_header('Set-Cookie', f'npc_session={cookie}; HttpOnly; SameSite=Strict; Path=/')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/':
            self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/report':
            self.send(200, json.loads((ROOT / 'artifacts/report.json').read_text()))
        elif self.path == '/api/presets':
            self.send(200, {name: questions_for(values) for name, values in PRESETS.items()})
        else:
            self.send(404, {'error': '页面不存在'})

    def do_POST(self):
        try:
            if self.path not in ('/api/chat', '/api/reset'):
                return self.send(404, {'error': '接口不存在'})
            port = self.server.server_port
            if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                return self.send(403, {'error': '仅允许本地页面访问'})
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384: raise ValueError('请求长度不正确')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict): raise ValueError('请求格式不正确')
            cookie = SimpleCookie(self.headers.get('Cookie', ''))
            sid = cookie['npc_session'].value if 'npc_session' in cookie else str(uuid.uuid4())
            name = data.get('name', 'Aki')
            if name not in PRESETS: raise ValueError('未知NPC')
            with LOCK:
                if sid not in SESSIONS and len(SESSIONS) >= 100:
                    SESSIONS.pop(next(iter(SESSIONS)))
                state = SESSIONS.setdefault(sid, {})
                if self.path == '/api/reset':
                    state.pop(name, None)
                    return self.send(200, {'ok': True}, cookie=sid)
                text = data.get('text')
                if not isinstance(text, str) or not text.strip() or len(text) > 300:
                    raise ValueError('请输入1～300字的玩家话语')
                q = data.get('questions', questions_for(PRESETS[name]))
                memory = state.setdefault(name, {})
                result = respond(MODEL, VOCAB, IDF, text.strip(), name, q, memory)
            self.send(200, result, cookie=sid)
        except (ValueError, TypeError, KeyError) as error:
            self.send(400, {'error': str(error)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local NPC Personality Lab demo')
    parser.add_argument('--port', type=int, default=8765, help='Local HTTP port (default: 8765)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    print(f'NPC demo: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
