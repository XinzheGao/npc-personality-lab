"""Local browser demo with separate rule/live state and session-scoped audit exports."""
import argparse
import json
import threading
import uuid
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dialogue import AuditLog, DeepSeek, ModelError, live_turn, load_env, paper_scores, validate_scores
from model import load_model
from npc import PRESETS, questions_for, respond, score_tipi

ROOT = Path(__file__).resolve().parent
load_env(ROOT / '.env')
CLIENT = DeepSeek()
AUDIT = AuditLog(ROOT / 'runtime/calls.sqlite3')
MODEL, VOCAB, IDF = load_model(ROOT / 'artifacts/intent.pt')
SESSIONS, LOCK = {}, threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, kind='application/json; charset=utf-8', cookie=None, download=False):
        payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download:
            self.send_header('Content-Disposition', 'attachment; filename="npc-dialogue-records.json"')
        if cookie:
            self.send_header('Set-Cookie', f'npc_session={cookie}; HttpOnly; SameSite=Strict; Path=/')
        self.end_headers()
        self.wfile.write(payload)

    def local_request(self):
        port = self.server.server_port
        return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

    def session_id(self):
        try:
            cookie = SimpleCookie(self.headers.get('Cookie', ''))
            value = cookie['npc_session'].value if 'npc_session' in cookie else ''
            if str(uuid.UUID(value)) == value:
                return value
        except (ValueError, CookieError):
            pass
        return str(uuid.uuid4())

    def do_GET(self):
        if not self.local_request():
            return self.send(403, {'error': '仅允许本机访问'})
        if self.path == '/':
            self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/report':
            self.send(200, json.loads((ROOT / 'artifacts/report.json').read_text()))
        elif self.path == '/api/presets':
            self.send(200, {name: questions_for(values) for name, values in PRESETS.items()})
        elif self.path == '/api/config':
            self.send(200, CLIENT.public_config())
        elif self.path == '/api/export':
            self.send(200, {'schema_version': 1, 'records': AUDIT.records(self.session_id())}, download=True)
        else:
            self.send(404, {'error': '页面不存在'})

    def do_POST(self):
        sid = None
        try:
            if self.path not in ('/api/chat', '/api/reset', '/api/context'):
                return self.send(404, {'error': '接口不存在'})
            port = self.server.server_port
            if not self.local_request() or self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                return self.send(403, {'error': '仅允许本地页面访问'})
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError('请求长度不正确')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('请求格式不正确')
            name, mode = data.get('name', 'Aki'), data.get('mode', 'rule')
            if not isinstance(name, str) or name not in PRESETS:
                raise ValueError('未知NPC')
            if mode not in ('rule', 'live'):
                raise ValueError('未知对话模式')
            sid = self.session_id()
            with LOCK:
                if sid not in SESSIONS and len(SESSIONS) >= 100:
                    return self.send(503, {'error': '本地会话数量已满，请重启服务'}, cookie=sid)
                session = SESSIONS.setdefault(sid, {'lock': threading.Lock(), 'contexts': {}})
            # Serialize only this browser's requests; model I/O never holds the global lock.
            with session['lock']:
                contexts = session['contexts']
                if self.path == '/api/reset':
                    for key in list(contexts):
                        if key[:2] == (mode, name):
                            del contexts[key]
                    return self.send(200, {'ok': True}, cookie=sid)
                if mode == 'live':
                    scores = validate_scores(data.get('scores', paper_scores(name)))
                    config_key = tuple(scores)
                else:
                    questions = data.get('questions', questions_for(PRESETS[name]))
                    score_tipi(questions)
                    config_key = tuple(questions)
                key = (mode, name, config_key)
                if key not in contexts and len(contexts) >= 50:
                    raise ValueError('配置过多，请重置当前角色后继续')
                memory = contexts.setdefault(key, {})
                if self.path == '/api/context':
                    return self.send(200, {'history': memory.get('history', []), 'last': memory.get('last')}, cookie=sid)
                text = data.get('text')
                if not isinstance(text, str) or not text.strip() or len(text) > 300:
                    raise ValueError('请输入1～300字的玩家话语')
                if mode == 'live':
                    result = live_turn(CLIENT, AUDIT, sid, name, scores, text.strip(), memory)
                else:
                    result = respond(MODEL, VOCAB, IDF, text.strip(), name, questions, memory.setdefault('rule_memory', {}))
                    memory['history'] = [*memory.get('history', []), {'role': 'user', 'content': text.strip()}, {'role': 'assistant', 'content': result['reply']}][-60:]
                memory['last'] = result
            self.send(200, result, cookie=sid)
        except ModelError as error:
            self.send(502, {'error': str(error)}, cookie=sid)
        except (ValueError, TypeError, KeyError) as error:
            self.send(400, {'error': str(error)}, cookie=sid)


def main():
    parser = argparse.ArgumentParser(description='Local NPC Personality Lab demo')
    parser.add_argument('--port', type=int, default=8765, help='Local HTTP port (default: 8765)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    print(f'NPC demo: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
