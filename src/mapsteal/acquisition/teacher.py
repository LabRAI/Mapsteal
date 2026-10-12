"""Provider-agnostic Teacher layer (family version). propose_and_answer(state, graph_state) -> validated structured response; exactly ONE external request per attempt; extra requests are only explicit, counted retries after an invalid response. No other model exists in this module."""
from __future__ import annotations
import hashlib, json, datetime, os
from dataclasses import dataclass, field
from pathlib import Path
from . import schema as sc

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

@dataclass
class ProviderResult:
    text: str; input_tokens: int; output_tokens: int; status: str = 'completed'; model: str = ''; raw: dict = field(default_factory=dict)

class TeacherProvider:
    name = 'abstract'; model = ''; price_in_per_m = 0.0; price_out_per_m = 0.0
    def complete(self, prompt: str, schema: dict, max_output_tokens: int) -> ProviderResult: raise NotImplementedError
    def cost_usd(self, r: ProviderResult): return (r.input_tokens * self.price_in_per_m + r.output_tokens * self.price_out_per_m) / 1e6

class OpenAIResponsesProvider(TeacherProvider):
    """GPT-4.1 through the OpenAI Responses API with a strict json_schema."""
    name = 'openai_responses'; price_in_per_m = 2.0; price_out_per_m = 8.0
    def __init__(self, model='gpt-4.1-2025-04-14', temperature=0, timeout=120): self.model = model; self.temperature = temperature; self.timeout = timeout
    def complete(self, prompt, schema, max_output_tokens):
        from openai import OpenAI
        raw = OpenAI(max_retries=0, timeout=self.timeout).responses.create(model=self.model, input=prompt, temperature=self.temperature, max_output_tokens=max_output_tokens, store=False,
                                                                         text={'format': {'type': 'json_schema', 'name': 'emergent_graph_v1', 'strict': True, 'schema': schema}}).model_dump()
        text = raw.get('output_text') if isinstance(raw.get('output_text'), str) else ''
        if not text:
            for x in raw.get('output', []):
                for y in x.get('content', []):
                    if isinstance(y.get('text'), str): text = y['text']
        u = raw.get('usage') or {}
        return ProviderResult(text=text, input_tokens=int(u.get('input_tokens', 0) or 0), output_tokens=int(u.get('output_tokens', 0) or 0), status=raw.get('status', 'completed'), model=raw.get('model', self.model), raw={'id': raw.get('id'), 'status': raw.get('status')})

class CallLedger:
    """Append-only ledger of EVERY external model request; also the reuse store (identical (state, attempt, request) is never paid twice, also across arms that share the ledger)."""
    def __init__(self, path): self.path = Path(path); self.rows = [json.loads(l) for l in open(self.path)] if self.path.exists() else []
    def find(self, state_id, attempt, request_sha):
        for r in self.rows:
            if r['state_id'] == state_id and r['attempt'] == attempt and r['request_sha256'] == request_sha: return r
    def append(self, row):
        self.rows.append(row)
        with open(self.path, 'a') as f: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'); f.flush()

def propose_and_answer(provider, ledger, state_id, prompt, schema, common, known_ids, max_output_tokens, max_retries=1, blind=False, ledger_state_id=None):
    req = sha(prompt); attempts = []; lsid = ledger_state_id or state_id   # AMENDMENT 1: optional ledger identity (default: unchanged)
    for attempt in range(1 + max_retries):
        led = ledger.find(lsid, attempt, req)
        if led is not None: res = ProviderResult(text=led['response_text'], input_tokens=led['usage']['input_tokens'], output_tokens=led['usage']['output_tokens'], status=led['status'], model=led['model']); reused = True
        else: res = provider.complete(prompt, schema, max_output_tokens); reused = False
        err = None; obj = None
        if res.status != 'completed': err = f'status {res.status}'
        else:
            try: obj = json.loads(res.text); err = sc.validate(obj, schema, common, known_ids, blind=blind)
            except Exception as e: err = 'json parse: ' + repr(e)[:80]
        cost = provider.cost_usd(res)
        row = {'role': 'teacher', 'state_id': lsid, 'attempt': attempt, 'request_sha256': req, 'ts': now(), 'model': res.model or provider.model, 'provider': provider.name, 'usage': {'input_tokens': res.input_tokens, 'output_tokens': res.output_tokens}, 'cost_usd': cost, 'status': res.status, 'valid': err is None, 'error': err, 'response_text': res.text}
        if not reused: ledger.append(row)
        attempts.append({k: row[k] for k in ('attempt', 'request_sha256', 'usage', 'cost_usd', 'status', 'valid', 'error')} | {'reused_from_ledger': reused})
        if err is None: return {'obj': obj, 'attempts': attempts, 'error': None}
    return {'obj': None, 'attempts': attempts, 'error': err}
