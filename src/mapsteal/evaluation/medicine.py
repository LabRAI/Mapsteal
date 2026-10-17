"""Frozen Medicine evaluation of Table 2: MedMCQA (500-item rendered panel), MedQA (1,273-item USMLE test, A-E), MedXpertQA-Text (2,450 items, A-J).
Next-token option-letter argmax (TransformersScorer, batch size 1, bf16, no chain of thought); accuracy = correct / n. Prompts are those of the formal evaluators:
MedMCQA rows carry their rendered prompt (s2_eval_rendered_rows_v1.jsonl); MedQA and MedXpertQA prompts are the frozen lettered multiple-choice prompts of the formal evaluation.
No benchmark-specific exclusions apply to the Medicine benchmarks of Table 2. The panels are NOT shipped: supply them in one directory under their original file names (sha256-checked)."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from ..models import resolve

PANELS = {  # benchmark -> (file name, sha256 of the frozen panel, n)
    'medmcqa': ('s2_eval_rendered_rows_v1.jsonl', '14d72625e957247bd3156f89de17cb25fbaf62c556a8761335f6dc1a9c52d6d5', 500),
    'medqa': ('medqa_test.jsonl', 'cfa75cbcc53c29b023752b0a395ff49d8202121041392735ae44771a5ec734a3', 1273),
    'medxpertqa_text': ('medxpertqa_text.jsonl', '1866e7ff66328aede184563104c42fac2ba84dd43f56fd5696d91a62d69eb046', 2450)}
MAX_LENGTH = {'medmcqa': 2048, 'medqa': 2048, 'medxpertqa_text': 4096}
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]

def _options(options, order): return "\n".join(f"{key}. {str(options[key]).strip().replace(chr(10), ' ')}" for key in order if key in options)
def medqa_example(row, index):   # task_example('medqa_a1')
    labels = ("A", "B", "C", "D", "E"); item_id = str(row.get("item_id", row.get("item_index", row.get("id", index))))
    options = {str(k).upper(): v for k, v in row["options"].items()}
    prompt = ("You are a medical multiple-choice answering system.\nRead the question and options carefully.\n"
              "Answer with ONLY one letter: A, B, C, D, or E.\n\nQuestion:\n"
              f"{row['question'].strip()}\n\nOptions:\n{_options(options, labels)}\n\nAnswer:")
    gold = str(row["answer_idx"]).upper()
    if gold not in labels: raise ValueError(f"medqa/{item_id}: gold label outside label space: {gold}")
    return item_id, prompt, gold, labels
def letters(ls): return ', '.join(ls[:-1]) + ', or ' + ls[-1]
def medxpert_example(r):   # score_model.mcq_prompt
    ls = sorted(r['options']); opts = '\n'.join(f'{k}. {r["options"][k]}' for k in ls)
    return r['item_id'], ('You are a medical multiple-choice answering system.\nRead the question and options carefully.\n' f'Answer with ONLY one letter: {letters(ls)}.\n\nQuestion:\n{r["question"]}\n\nOptions:\n{opts}\n\nAnswer:'), r['gold'], tuple(ls)

def examples(bench, panels_dir, limit=None):
    """Returns (list of (item_id, prompt, gold, labels), is_full_panel). Verifies the panel hash; `limit` takes the first N items (a labelled subset)."""
    fname, sha, n = PANELS[bench]; p = Path(panels_dir) / fname
    if not p.is_file(): raise SystemExit(f'missing frozen panel {p} (build it with: python tools/prepare_benchmarks.py panels, see README)')
    if sha256(p) != sha: raise SystemExit(f'panel {p} does not match the frozen sha256 {sha}')
    rows = jl(p); assert len(rows) == n, (bench, len(rows), n)
    if bench == 'medmcqa': ex = [(r['item_id'], r['prompt'], r['gold'], ("A", "B", "C", "D")) for r in rows]
    elif bench == 'medqa': ex = [medqa_example(r, i) for i, r in enumerate(rows)]
    else: ex = [medxpert_example(r) for r in rows]
    return (ex[:limit] if limit else ex), (not limit or limit >= n)

def evaluate(checkpoint, student, panels_dir, output, benchmarks=('medmcqa', 'medqa', 'medxpertqa_text'), limit=None, device='cuda:0'):
    """checkpoint: LoRA adapter directory or 'none' (pristine Student). Writes <output>/<bench>.jsonl predictions and <output>/evaluation.json."""
    import torch
    from .scorer import TransformersScorer
    m = resolve(student); out = Path(output); out.mkdir(parents=True, exist_ok=True); t0 = time.time(); summary = {'student': m['id'], 'revision': m['rev'], 'checkpoint': str(checkpoint), 'benchmarks': {}}
    if str(checkpoint).lower() != 'none': summary['adapter_sha256'] = sha256(Path(checkpoint) / 'adapter_model.safetensors')
    todo = {b: examples(b, panels_dir, limit) for b in benchmarks}
    for b, (ex, full) in todo.items():
        if (out / f'{b}.jsonl').exists(): raise SystemExit(f'refusing to overwrite {out / (b + ".jsonl")}')
    sc = TransformersScorer(m['base_path'], m['rev'], checkpoint, device, max_length=max(MAX_LENGTH.values()))   # one model load; max_length is only consulted by the (removed) continuation modes
    for b, (ex, full) in todo.items():
        sc.max_length = MAX_LENGTH[b]; preds = []
        for item_id, prompt, gold, labels in ex:
            lp = sc.score_batch([prompt], labels, 'next_token')[0]; pred = max(lp, key=lp.get)
            preds.append({'item_id': item_id, 'gold': gold, 'prediction': pred, 'correct': pred == gold, 'lp': {k: round(float(v), 4) for k, v in lp.items()}})
        (out / f'{b}.jsonl').write_text(''.join(json.dumps(p, sort_keys=True) + '\n' for p in preds)); nc = sum(p['correct'] for p in preds)
        summary['benchmarks'][b] = {'n': len(preds), 'n_correct': nc, 'accuracy_pct': 100 * nc / len(preds), 'panel_sha256': PANELS[b][1], 'panel_n': PANELS[b][2],
                                    'coverage': 'full frozen panel' if full else f'SUBSET: first {limit} items (smoke test, not comparable to Table 2)'}
        print(b, summary['benchmarks'][b], flush=True)
    summary['seconds'] = round(time.time() - t0, 1); summary['torch'] = torch.__version__; (out / 'evaluation.json').write_text(json.dumps(summary, indent=1)); return summary
