"""Frozen Cybersecurity evaluation of Table 1: CyberMetric (2,500 rows; cybermetric:1729 is excluded from every reported accuracy), SecEval (1,521), CTI-Bench (2,482).
Zero-shot lettered prompt, next-token option-letter argmax (TransformersScorer, batch size 1, max_length 8192, bf16, no chain of thought) as in the formal evaluation.
All rows are scored and written; the reported accuracy excludes the frozen exclusion. Panels are NOT shipped (supply the three files under their original names, sha256-checked)."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from ..models import resolve
PANELS = {  # benchmark -> (file, sha256, n)
    'cybermetric': ('cyb_cybermetric.jsonl', 'ed1543856baa61d77e095238ec1aa1cec30be931631964a14073a1f58541eedf', 2500),
    'seceval': ('cyb_seceval.jsonl', '0049dff4e7c336f862902e21452a41adca37b6b3eac5bf1aae4a5c9f40097a9f', 1521),
    'cti_bench': ('cyb_cti_bench.jsonl', '0b7bc60b27f2948222da2a33c68bf0deafb5b1ee6d9861c37baf6e88d6bb3e60', 2482)}
EXCLUDE = {'cybermetric': {'cybermetric:1729'}}
MAXLEN = 8192
HEAD = 'Answer the following multiple-choice question. Choose exactly one option label.\n\n'
def prompt(r): ls = sorted(r['options']); return HEAD + f"Question: {r['question']}\n" + '\n'.join(f'{k}. {r["options"][k]}' for k in ls) + '\n\nAnswer:', tuple(ls)
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evaluate(checkpoint, student, panels_dir, output, benchmarks=('cybermetric', 'seceval', 'cti_bench'), limit=None, device='cuda:0'):
    import torch
    from .scorer import TransformersScorer
    m = resolve(student); out = Path(output); out.mkdir(parents=True, exist_ok=True); t0 = time.time(); summary = {'student': m['id'], 'revision': m['rev'], 'checkpoint': str(checkpoint), 'benchmarks': {}}
    if str(checkpoint).lower() != 'none': summary['adapter_sha256'] = sha256(Path(checkpoint) / 'adapter_model.safetensors')
    data = {}
    for b in benchmarks:
        fname, sha, n = PANELS[b]; p = Path(panels_dir) / fname
        if not p.is_file(): raise SystemExit(f'missing frozen panel {p} (build it with: python tools/prepare_benchmarks.py panels, see README)')
        if sha256(p) != sha: raise SystemExit(f'panel {p} does not match the frozen sha256 {sha}')
        rows = jl(p); assert len(rows) == n, (b, len(rows), n); data[b] = rows[:limit] if limit else rows
        if (out / f'{b}.jsonl').exists(): raise SystemExit(f'refusing to overwrite {out / (b + ".jsonl")}')
    sc = TransformersScorer(m['base_path'], m['rev'], checkpoint, device, max_length=MAXLEN); tok = sc.tokenizer
    for b, rows in data.items():
        res = []
        for r in rows:
            p, labels = prompt(r); ntok = len(tok(p, add_special_tokens=False)['input_ids']); lp = sc.score_batch([p], labels, 'next_token')[0]; pred = max(lp, key=lp.get); gold = r.get('gold_set', [r['gold']])
            res.append({'item_id': r['item_id'], 'subject': r['subject'], 'gold': r['gold'], 'prediction': pred, 'correct': pred in gold, 'invalid': ntok > MAXLEN, 'prompt_tokens': ntok, 'lp': {k: round(float(v), 4) for k, v in lp.items()}})
        (out / f'{b}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in res))
        kept = [x for x in res if x['item_id'] not in EXCLUDE.get(b, set())]; nc = sum(x['correct'] for x in kept); full = not limit or limit >= PANELS[b][2]
        summary['benchmarks'][b] = {'n_scored': len(res), 'n_reported': len(kept), 'n_correct': nc, 'accuracy_pct': 100 * nc / len(kept), 'excluded_item_ids': sorted(EXCLUDE.get(b, set())), 'panel_sha256': PANELS[b][1],
                                    'coverage': 'full frozen panel' if full else f'SUBSET: first {limit} items (smoke test, not comparable to Table 1)'}
        print(b, summary['benchmarks'][b], flush=True)
    summary['seconds'] = round(time.time() - t0, 1); summary['torch'] = torch.__version__; (out / 'evaluation.json').write_text(json.dumps(summary, indent=1)); return summary
