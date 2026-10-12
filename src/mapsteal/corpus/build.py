"""Training-corpus construction (rule of the formal Medicine corpora).
Corpus = [cold-start seed rows] + the first N executed (LOCAL) trajectory rows after QC. QC (frozen): drop a row whose answer or explanation is empty; drop a row whose
normalised question equals an earlier kept question (or a seed question); NO top-up. Row format: {"input": "", "instruction": question, "output": answer + "\\n\\nExplanation: " + explanation}.
Budget accounting: B_total = seed rows + N executed steps (successful Teacher QA, QC drops can make the final row count smaller)."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

def norm(s): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').casefold())).strip()
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]

def build(trajectory_rows, seed_rows=(), n_steps=None):
    """Returns (rows, report). trajectory_rows: parsed trajectory.jsonl rows of an acquisition run."""
    loc = [r for r in trajectory_rows if r['call_type'] == 'LOCAL']
    if n_steps is not None: loc = loc[:n_steps]
    seen = {norm(x['instruction']) for x in seed_rows}; kept, drops = [], {}
    for r in loc:
        a, e = (r['answer'] or '').strip(), (r['explanation'] or '').strip(); q = r['realized_question'].strip(); why = None
        if not a or not e: why = 'empty_answer_or_explanation'
        elif norm(q) in seen: why = 'exact_duplicate_question'
        if why: drops[why] = drops.get(why, 0) + 1; continue
        seen.add(norm(q)); kept.append({'input': '', 'instruction': q, 'output': a + '\n\nExplanation: ' + e})
    rows = list(seed_rows) + kept
    return rows, {'seed_rows': len(seed_rows), 'executed_steps_used': len(loc), 'qc_drops': drops, 'trajectory_rows_kept': len(kept), 'corpus_rows': len(rows),
                  'budget_total_nominal': len(seed_rows) + len(loc), 'root_rows_used': sum(1 for r in loc if r['state']['generation_depth'] == 0)}

def write(rows, path):
    text = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows); Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()

def build_from_run(run_dir, out_path, seed_path=None, n_steps=None):
    """Read <run_dir>/trajectory.jsonl, write the corpus and <out>.manifest.json."""
    run_dir = Path(run_dir); traj = jl(run_dir / 'trajectory.jsonl'); seed = jl(seed_path) if seed_path else []
    rows, rep = build(traj, seed, n_steps); rep['sha256'] = write(rows, out_path); rep['trajectory_sha256'] = hashlib.sha256((run_dir / 'trajectory.jsonl').read_bytes()).hexdigest()
    rep['seed_corpus'] = str(seed_path) if seed_path else None
    Path(str(out_path) + '.manifest.json').write_text(json.dumps(rep, indent=1)); return rows, rep
