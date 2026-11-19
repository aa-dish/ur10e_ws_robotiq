"""Tiny client for a local Ollama server (HTTP, ``requests`` only)."""
from __future__ import annotations

import json
import logging
from typing import List, Optional

log = logging.getLogger(__name__)

# first match wins when no model name is configured
PREFERRED = ['qwen2.5', 'qwen3', 'llama3.1', 'llama3.2', 'llama3', 'mistral', 'gemma3', 'gemma2', 'phi4', 'phi3']


class OllamaClient:
    def __init__(self, host: str = 'http://localhost:11434', model: str = '', timeout: float = 60.0):
        self.host = host.rstrip('/')
        self.model = model
        self.timeout = timeout

    def available_models(self) -> List[str]:
        import requests
        r = requests.get(f'{self.host}/api/tags', timeout=3.0)
        r.raise_for_status()
        return [m['name'] for m in r.json().get('models', [])]

    def ensure_model(self) -> Optional[str]:
        """Return a usable model name (the configured one, or the best installed one)."""
        try:
            models = self.available_models()
        except Exception as e:                                         # server not running
            log.warning('Ollama not reachable at %s (%s)', self.host, e)
            return None
        if not models:
            log.warning('Ollama is running but has no models installed (ollama pull llama3.2)')
            return None
        if self.model:
            for m in models:
                if m == self.model or m.split(':')[0] == self.model.split(':')[0]:
                    self.model = m
                    return m
            log.warning('Ollama model %r not installed; installed: %s', self.model, models)
        text_models = [m for m in models if not any(k in m.lower() for k in ('embed', 'llava', 'vision', 'moondream'))]
        for pref in PREFERRED:
            for m in text_models:
                if m.lower().startswith(pref):
                    self.model = m
                    return m
        self.model = (text_models or models)[0]
        return self.model

    def chat_json(self, system: str, user: str, schema: Optional[dict] = None, temperature: float = 0.0) -> dict:
        """One chat turn that must answer with JSON.  Raises on any failure."""
        import requests
        body = {
            'model': self.model, 'stream': False,
            'format': schema if schema else 'json',
            'options': {'temperature': temperature, 'num_ctx': 4096},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        }
        r = requests.post(f'{self.host}/api/chat', json=body, timeout=self.timeout)
        r.raise_for_status()
        content = r.json()['message']['content']
        return json.loads(content)

    def describe_image(self, png_bytes: bytes, prompt: str, model: str) -> str:
        """Ask a vision model (llava, llama3.2-vision, qwen2.5vl, ...) about an image."""
        import base64
        import requests
        body = {'model': model, 'stream': False, 'options': {'temperature': 0},
                'messages': [{'role': 'user', 'content': prompt,
                              'images': [base64.b64encode(png_bytes).decode()]}]}
        r = requests.post(f'{self.host}/api/chat', json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()['message']['content']
