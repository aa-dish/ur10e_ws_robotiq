"""Checks the Ollama code path against a fake local server (no real LLM needed)."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from bin_picking_ai.llm import CommandParser, OllamaClient, resolve_targets
from bin_picking_ai.vision import Catalog

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLY = {'obj': {}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'models': [{'name': 'llava:7b'}, {'name': 'llama3.2:latest'}]}).encode())

    def do_POST(self):
        n = int(self.headers['Content-Length']); body = json.loads(self.rfile.read(n))
        REPLY['last'] = body
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'message': {'content': json.dumps(REPLY['obj'])}}).encode())


def _server():
    srv = HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class FakeDet:
    def __init__(self, i, label):
        self.id, self.label = i, label
        self.color_name, self.height, self.center, self.size = 'yellow', 0.1, (0.0, -0.5, 0.2), (0.1, 0.05)
        self.candidates, self.area_m2, self.yaw = [], 0.005, 0.0


def test_ollama_model_autoselect_and_grounded_ids():
    srv = _server()
    cat = Catalog(os.path.join(PKG, 'data', 'catalog.json'))
    cl = OllamaClient(f'http://127.0.0.1:{srv.server_port}')
    p = CommandParser(cat, cl, use_llm=True)
    assert p.llm_ready and cl.model == 'llama3.2:latest'            # skipped the vision-only model
    dets = [FakeDet(0, 'mustard_bottle'), FakeDet(1, 'cereal')]
    REPLY['obj'] = {'action': 'pick_place', 'object_ids': [1, 99], 'all': False, 'target': 'bin', 'reply': 'ok'}
    c = p.parse('grab the breakfast thing', dets)
    assert c.source == 'ollama' and c.object_ids == [1]               # invented id 99 dropped
    assert [d.id for d in resolve_targets(c, dets, cat)] == [1]
    assert 'id=1: cereal' in REPLY['last']['messages'][1]['content']  # scene was given to the LLM


def test_ollama_garbage_falls_back_to_rules():
    srv = _server()
    cat = Catalog(os.path.join(PKG, 'data', 'catalog.json'))
    p = CommandParser(cat, OllamaClient(f'http://127.0.0.1:{srv.server_port}'), use_llm=True)
    REPLY['obj'] = {'action': 'dance'}
    c = p.parse('pick the mustard bottle', [FakeDet(0, 'mustard_bottle')])
    assert c.source == 'rules' and c.action == 'pick_place'


def test_ollama_down_falls_back_to_rules():
    cat = Catalog(os.path.join(PKG, 'data', 'catalog.json'))
    p = CommandParser(cat, OllamaClient('http://127.0.0.1:9'), use_llm=True)
    assert not p.llm_ready
    assert p.parse('pick the mustard bottle', []).action == 'pick_place'
