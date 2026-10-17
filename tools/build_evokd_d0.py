#!/usr/bin/env python3
"""Build the initial data D0 (4 question/reference-answer rows) for `python -m baselines evokd run --d0 FILE`.
Sources and construction rules: manifests/evokd_d0_sources.json (the formal texts themselves are not distributed; their sha256 are recorded and compared below).

  medicine:       python tools/build_evokd_d0.py medicine --questions QUESTIONS.jsonl --output d0_medicine.jsonl
                  QUESTIONS.jsonl: one {"source_id": ..., "question": ...} per row of the manifest (your own hand-written rewrite of the source item).
                  The reference answer is the source dataset's own correct option, fetched with `datasets` (pip install datasets; needs internet).
  cybersecurity:  python tools/build_evokd_d0.py cybersecurity --trajectory runs/cyb/trajectory.jsonl --output d0_cybersecurity.jsonl
                  or --answers ANSWERS.jsonl ({"source_id": "CYB-001", "answer": ...}). The question is the root question of roots40.jsonl; the reference answer is the Teacher answer
                  to that root question taken from a MapSteal Cybersecurity acquisition (rows of the trajectory whose realized_question equals the root question)."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sha = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()
jl = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def gold_answer(row):
    from datasets import load_dataset
    L = row['loader']; kw = dict(split=L['split']); args = [L['hf_dataset']] + ([L['config']] if L['config'] else [])
    r = load_dataset(*args, **kw)[L['index']]; ds = row['dataset']
    if ds == 'MedQA': return r['options'][r['answer']]
    if ds == 'MedMCQA': return [r['opa'], r['opb'], r['opc'], r['opd']][r['cop']]
    if ds == 'MMLU-Med': return r['choices'][r['answer']]
    if ds == 'PubMedQA': return r['final_decision']
    raise SystemExit(f'unknown dataset {ds}')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('domain', choices=['medicine', 'cybersecurity']); ap.add_argument('--output', required=True)
    ap.add_argument('--questions'); ap.add_argument('--trajectory'); ap.add_argument('--answers'); a = ap.parse_args()
    man = json.loads((ROOT / 'manifests/evokd_d0_sources.json').read_text())[a.domain]; out = []
    if a.domain == 'medicine':
        if not a.questions:
            print('Each row needs your hand-written question. Sources and rewrite rules:'); [print(f"  {r['source_id']}  ({r['dataset']} {r['split']}, topic: {r['knowledge_object']}) -- {r['question']['origin']}") for r in man['rows']]
            raise SystemExit('rerun with --questions QUESTIONS.jsonl')
        q = {r['source_id']: r['question'] for r in jl(a.questions)}
        for r in man['rows']:
            if r['source_id'] not in q: raise SystemExit(f"missing question for {r['source_id']}")
            ans = gold_answer(r); out.append(dict(dataset=r['dataset'], knowledge_object=r['knowledge_object'], question=q[r['source_id']], reference_answer=ans, source_id=r['source_id']))
            print(f"{r['source_id']}: reference answer {'equals' if sha(ans) == r['reference_answer']['sha256'] else 'DIFFERS FROM'} the formal one; question {'equals' if sha(q[r['source_id']]) == r['question']['sha256'] else 'differs from'} the formal one")
    else:
        roots = {r['state_id']: r for r in jl(ROOT / 'src/mapsteal/data/cybersecurity/roots40.jsonl')}
        if a.answers: ans = {r['source_id']: r['answer'] for r in jl(a.answers)}
        elif a.trajectory:
            tr = jl(a.trajectory); ans = {}
            for r in man['rows']:
                hit = [t for t in tr if t.get('realized_question') == roots[r['source_id']]['T']]
                if not hit: raise SystemExit(f"no trajectory row for root {r['source_id']}")
                ans[r['source_id']] = hit[0]['answer']
        else: raise SystemExit('give --trajectory or --answers')
        for r in man['rows']:
            rt = roots[r['source_id']]; out.append(dict(dataset=r['dataset'], knowledge_object=r['knowledge_object'], question=rt['T'], reference_answer=ans[r['source_id']], source_id=r['source_id']))
            print(f"{r['source_id']}: question {'equals' if sha(rt['T']) == r['question']['sha256'] else 'DIFFERS FROM'} the formal one; reference answer {'equals' if sha(ans[r['source_id']]) == r['reference_answer']['sha256'] else 'differs from'} the formal one")
    Path(a.output).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out), encoding='utf-8'); print(f'wrote {len(out)} rows -> {a.output}')
main()
