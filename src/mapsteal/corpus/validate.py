"""Validation of an acquisition run and its corpus (schema, identifiers, parent-child references, ordering, budget accounting, failures, deduplication, training-example construction).
Read-only; returns a machine-readable report. No filtering rule is changed here: the checks only observe what the formal rules produced."""
from __future__ import annotations
import json, re
from pathlib import Path
from . import build as cb
LOCAL_KEYS = {'call_index', 'call_type', 'state', 'realized_question', 'answer', 'explanation', 'reusable_knowledge', 'proposal_items', 'raw_proposals', 'admitted_children', 'graph_events', 'prompt_sha256', 'model_calls'}
STATE_KEYS = {'state_id', 'source_root_id', 'parent_state_id', 'generation_depth', 'child_type', 'O', 'F', 'C', 'T', 'D'}

def validate_run(run_dir, corpus_path, seed_path=None):
    run = Path(run_dir); rows = cb.jl(run / 'trajectory.jsonl'); corpus = cb.jl(corpus_path); seed = cb.jl(seed_path) if seed_path else []
    loc = [r for r in rows if r['call_type'] == 'LOCAL']; chk = {}; det = {}
    bad = [r['call_index'] for r in loc if not LOCAL_KEYS <= set(r) or not STATE_KEYS <= set(r['state'])]
    chk['schema_required_fields'] = not bad; det['schema_violations'] = bad[:5]
    ci = [r['call_index'] for r in rows]; chk['call_index_strictly_increasing'] = all(a < b for a, b in zip(ci, ci[1:]))
    sid = [r['state']['state_id'] for r in loc]; chk['unique_executed_state_ids'] = len(sid) == len(set(sid))
    adm = [c['state_id'] for r in loc for c in r['admitted_children']]; chk['unique_admitted_child_ids'] = len(adm) == len(set(adm))
    roots_ok = all(r['state']['parent_state_id'] is None for r in loc if r['state']['generation_depth'] == 0); executed = {}; refs = []
    for r in loc:
        s = r['state']
        if s['generation_depth'] > 0:
            par = executed.get(s['parent_state_id'])
            if par is None or s['state_id'] not in {c['state_id'] for c in par['admitted_children']}: refs.append(s['state_id'])
            elif par['state']['source_root_id'] != s['source_root_id'] or par['state']['generation_depth'] + 1 != s['generation_depth']: refs.append(s['state_id'])
        executed[s['state_id']] = r
    chk['parent_child_references'] = roots_ok and not refs; det['dangling_or_inconsistent_children'] = refs[:5]
    q = run / 'queue_snapshots/queue_state.jsonl'
    if q.exists():
        sel = [json.loads(l)['selected_state'] for l in open(q)]; popped = [r['state']['state_id'] for r in rows if r['call_type'] in ('LOCAL', 'REALIZATION_FAIL', 'API_INVALID_FAIL')]
        chk['acquisition_order_matches_scheduler_log'] = sel == popped
    cl = json.loads((run / 'cost_ledger.json').read_text()) if (run / 'cost_ledger.json').exists() else {}
    chk['budget_accounting_steps_equal_QA_rows'] = (cl.get('executed_acquisition_steps') == len(loc)) if cl else None; chk['call_invariant_calls_eq_steps_failures_retries'] = bool(cl.get('invariant_calls_eq_executed_plus_failed_steps_plus_retries')) if cl else None   # None = not applicable (no cost_ledger.json)
    attempts = [m for r in rows for m in r.get('model_calls', [])]; ledger = run / 'model_call_ledger.jsonl'
    if ledger.exists(): chk['ledger_rows_equal_paid_attempts'] = len(cb.jl(ledger)) == sum(1 for m in attempts if not m['reused_from_ledger'])
    det['failed_steps'] = {t: sum(1 for r in rows if r['call_type'] == t) for t in ('REALIZATION_FAIL', 'API_INVALID_FAIL')}
    empty = [r['call_index'] for r in loc if not (r['answer'] or '').strip() or not (r['explanation'] or '').strip()]; det['empty_answer_or_explanation_rows'] = empty
    chk['all_recorded_attempts_valid_or_failed_step'] = all(m['valid'] or any(x['call_type'] == 'API_INVALID_FAIL' for x in rows) or m['attempt'] == 0 for m in attempts)
    rebuilt, rep = cb.build(rows, seed); chk['corpus_equals_rebuild_from_trajectory'] = rebuilt == corpus; det['qc'] = rep
    qs = [cb.norm(r['instruction']) for r in corpus]; chk['no_duplicate_questions_in_corpus'] = len(qs) == len(set(qs))
    chk['training_example_format'] = all(set(r) == {'input', 'instruction', 'output'} and r['input'] == '' and '\n\nExplanation: ' in r['output'] for r in corpus[len(seed):])
    chk['seed_rows_prefixed_unchanged'] = corpus[:len(seed)] == seed
    det['corpus_rows'] = len(corpus); det['executed_steps'] = len(loc)
    return {'checks': chk, 'details': det, 'all_checks_pass': all(v is not False for v in chk.values())}
