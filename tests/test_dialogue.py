"""Provider contract and browser/API integration; all model outputs here are test fixtures."""
import copy
import io
import json
import os
import tempfile
import threading
import unittest
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener

from dialogue import AuditLog, DeepSeek, ModelError, PAPER, live_turn, paper_scores


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audit = AuditLog(Path(self.temp.name) / 'calls.sqlite3')
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-secret'}, clear=True):
            self.client = DeepSeek()

    def response(self, text='先确认物资清单。', finish='stop'):
        return {'id': 'test-response', 'model': 'test-model', 'choices': [
            {'message': {'role': 'assistant', 'content': text}, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 8}}

    def test_http_contract_and_missing_key(self):
        raw = self.response()
        with patch('dialogue.urlopen') as call:
            call.return_value.__enter__.return_value = io.BytesIO(json.dumps(raw).encode())
            reply, received = self.client.generate([{'role': 'user', 'content': '你好'}])
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, 'https://api.deepseek.com/chat/completions')
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertEqual(payload['messages'][0]['content'], '你好')
        self.assertEqual(reply, raw['choices'][0]['message']['content'])
        self.assertEqual(received, raw)
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-secret')
        self.assertNotIn('test-secret', json.dumps(self.client.public_config()))
        self.client.key = ''
        with self.assertRaisesRegex(ModelError, '尚未配置'), patch('dialogue.urlopen') as call:
            self.client.generate([])
        call.assert_not_called()

    def test_error_body_not_leaked_and_incomplete_reply_rejected(self):
        with patch('dialogue.urlopen', side_effect=HTTPError('https://api.deepseek.com', 401, 'secret', {}, io.BytesIO(b'test-secret'))):
            with self.assertRaises(ModelError) as error:
                self.client.generate([])
        self.assertNotIn('test-secret', str(error.exception))
        self.assertIn('401', str(error.exception))
        for raw in [self.response(finish='length'), self.response(text=''), {'choices': []}, []]:
            with patch('dialogue.urlopen') as call:
                call.return_value.__enter__.return_value = io.BytesIO(json.dumps(raw).encode())
                with self.assertRaises(ModelError):
                    self.client.generate([])
        with patch('dialogue.urlopen', side_effect=URLError('test-secret')):
            with self.assertRaisesRegex(ModelError, '网络'):
                self.client.generate([])

    def test_history_and_record_are_from_actual_request(self):
        state = {}
        with patch.object(self.client, 'generate', return_value=('先确认物资清单。', self.response())) as generate:
            first = live_turn(self.client, self.audit, 'session-a', 'Rina', paper_scores('Rina'), '先做什么？', state)
            second = live_turn(self.client, self.audit, 'session-a', 'Rina', paper_scores('Rina'), '我刚才问了什么？', state)
        messages = generate.call_args.args[0]
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(second['memory']['turns'], 2)
        self.assertEqual(first['record']['conversation_id'], second['record']['conversation_id'])
        records = self.audit.records('session-a')
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]['messages'], messages)
        self.assertEqual(records[0]['raw_response']['model'], 'test-model')
        self.assertNotIn('test-secret', json.dumps(records))
        self.assertEqual(self.audit.records('session-b'), [])

    def test_failure_does_not_mutate_history_or_fallback(self):
        state = {'history': [{'role': 'user', 'content': '前一轮'}, {'role': 'assistant', 'content': '前一轮回复'}]}
        before = copy.deepcopy(state)
        with patch.object(self.client, 'generate', side_effect=ModelError('test failure')):
            with self.assertRaises(ModelError):
                live_turn(self.client, self.audit, 's', 'Aki', paper_scores('Aki'), '新问题', state)
        self.assertEqual(state, before)
        record = self.audit.records('s')[0]
        self.assertEqual(record['status'], 'error')
        self.assertNotIn('reply', record)
        self.assertNotIn('raw_response', record)

    def test_save_failure_does_not_commit_generated_reply(self):
        state = {}
        with patch.object(self.client, 'generate', return_value=('reply', self.response())), patch.object(self.audit, 'save', side_effect=ModelError('disk full')):
            with self.assertRaises(ModelError):
                live_turn(self.client, self.audit, 's', 'Aki', paper_scores('Aki'), '问题', state)
        self.assertEqual(state, {})

    def test_parameters_and_turn_limit(self):
        for values in [[float('nan')]*5, [True]*5, [8]*5, [], ['4']*5]:
            with self.assertRaises(ValueError):
                live_turn(self.client, self.audit, 's', 'Aki', values, '问题', {})
        with self.assertRaisesRegex(ValueError, '15轮'):
            live_turn(self.client, self.audit, 's', 'Aki', paper_scores('Aki'), '问题', {'history': [{}]*30})
        self.assertEqual(paper_scores('Aki'), [5.98, 5.5, 4.0, 2.02, 5.98])
        self.assertEqual(PAPER['profiles']['Rina'], [.17,.17,.83,.17,.50])


class BrowserApiTests(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = DeepSeek()
        self.client.key = 'test-secret'
        self.audit = AuditLog(Path(self.temp.name) / 'http.sqlite3')
        self.patches = [patch.object(app, 'CLIENT', self.client), patch.object(app, 'AUDIT', self.audit), patch.object(app, 'SESSIONS', {})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.browser = build_opener(HTTPCookieProcessor(CookieJar()))

    def request(self, path, payload=None, browser=None):
        request = Request(self.base+path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with (browser or self.browser).open(request) as response:
            return json.load(response)

    def test_real_http_flow_isolation_reset_export(self):
        payload = {'name':'Aki', 'mode':'live', 'text':'先做什么？'}
        self.assertTrue(self.request('/api/config')['ready'])
        with patch.object(self.client, 'generate', return_value=('fixture-only reply', {'model':'test-model','choices':[]})):
            first = self.request('/api/chat', payload)
            self.assertEqual(first['memory']['turns'], 1)
            self.assertEqual(self.request('/api/context', payload)['history'][0]['content'], payload['text'])
            self.assertEqual(self.request('/api/context', {**payload, 'name':'Rina'})['history'], [])
            self.assertEqual(self.request('/api/context', {**payload, 'scores':[4]*5})['history'], [])
            self.assertEqual(self.request('/api/context', {**payload, 'mode':'rule'})['history'], [])
            other = build_opener(HTTPCookieProcessor(CookieJar()))
            self.assertEqual(self.request('/api/context', payload, other)['history'], [])
            self.assertEqual(self.request('/api/export', browser=other)['records'], [])
            self.assertEqual(len(self.request('/api/export')['records']), 1)
            self.request('/api/reset', payload)
            self.assertEqual(self.request('/api/context', payload)['history'], [])
            self.assertEqual(len(self.request('/api/export')['records']), 1)
            second = self.request('/api/chat', payload)
            self.assertNotEqual(first['record']['conversation_id'], second['record']['conversation_id'])
        with patch.object(self.client, 'generate', side_effect=ModelError('fixture failure')):
            with self.assertRaises(HTTPError) as error:
                self.request('/api/chat', payload)
            self.assertEqual(error.exception.code, 502)
        self.assertEqual(len(self.request('/api/context', payload)['history']), 2)
        self.assertEqual(self.request('/api/export')['records'][-1]['status'], 'error')
        for text in ('我不想打你', '今天天气如何', '你好，给我任务再买药'):
            result = self.request('/api/chat', {'mode':'rule','name':'Aki','text':text})
            self.assertEqual(result['response_mode'], '规则回应（未调用LLM）')
        with self.assertRaises(HTTPError) as error:
            self.request('/api/chat', {'mode':'live','name':[],'text':'x'})
        self.assertEqual(error.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
