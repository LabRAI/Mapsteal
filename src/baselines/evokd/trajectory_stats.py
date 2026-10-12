#!/usr/bin/env python3
"""Descriptive per-trajectory record (never used to adjust the algorithm). usage: trajectory_stats.py <run_dir>..."""
import json, re, sys
from pathlib import Path
norm = lambda s: re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()
for rd in map(Path, sys.argv[1:]):
    ev = [json.loads((rd / f'events/event_{k:02d}_chat.json').read_text()) for k in range(1, 64) if (rd / f'events/event_{k:02d}_chat.json').is_file()]
    seen, per, dup_tot, tot = set(), [], 0, 0
    for e in ev:
        d = e['identify_diagnostics']; qs = [norm(r['question']) for r in e['rows']]
        nd = sum(q in seen for q in qs); seen.update(qs); dup_tot += nd; tot += len(qs)
        per.append({'event': e['iteration'], 'step': e['step'], 'n_rows': len(qs), 'n_unusable': len(e['unusable_rows']), 'wrong': d['n_wrong'], 'correct': d['n_correct'], 'repeat_questions': nd, 'unique_cum': len(seen)})
    base = json.loads((rd / 'baseline_d0_identify.json').read_text())
    steps = [json.loads(l) for l in (rd / 'step_log.jsonl').read_text().splitlines() if l.strip()]
    summ = json.loads((rd / 'train_summary.json').read_text()) if (rd / 'train_summary.json').is_file() else {}
    wall = json.loads((rd / 'gpu_wall_seconds.json').read_text()) if (rd / 'gpu_wall_seconds.json').is_file() else {}
    out = {'run': rd.name, 'events': len(ev), 'candidates': tot, 'unique_questions': len(seen), 'repeat_rate': dup_tot / tot if tot else None,
           'd0_identify': {'wrong': len(base['wrong']), 'correct': len(base['correct'])}, 'total_wrong': sum(p['wrong'] for p in per), 'total_correct': sum(p['correct'] for p in per),
           'final_training_pool_rows': 4 + sum(p['n_rows'] for p in per), 'steps': len(steps), 'step_seconds_sum': sum(s['seconds'] for s in steps),
           'wall_seconds_last_invocation': wall.get('wall_seconds_this_invocation'), 'teacher_totals': summ.get('teacher_totals'), 'per_event': per}
    (rd / 'trajectory_stats.json').write_text(json.dumps(out, indent=1)); print(rd.name, {k: v for k, v in out.items() if k != 'per_event'})
