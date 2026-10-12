"""MapSteal acquisition driver. `run` is the loop of the formal runner: with amendment1=False the Medicine runner (run_branch.py sha256 27de3b7f...), with amendment1=True the Cybersecurity runner (run_branch.py, AMENDMENT 1:
failed-root re-queue, root-completion guard, first-attempt validity logged instead of a stop).
ONE external Teacher request per executed acquisition step (+ explicit counted retries after an invalid response); no auxiliary model of any kind.
Per step: pop the next state with the frontier scheduler (`fair` = active-root fair rule), realize the question deterministically from T and D (roots use their frozen question),
build PROMPT_C + the direction-registry addition, ask the Teacher for answer / explanation / typed child proposals (strict JSON schema), validate, admit children by the
deterministic admission rules (engine.proposals), update the per-root direction graph, and append one trajectory row. B = number of executed (successful) steps = QA rows.
Mechanical differences from run_branch.py: package-relative imports, runtime built from the contract/engine/realizer modules, the optional dashboard (unused in the formal
runs) removed (rows keep `dashboard_text: null`), argparse main replaced by mapsteal.cli."""
from __future__ import annotations
import json, sys
from pathlib import Path
import hashlib
from types import SimpleNamespace
from . import scheduler as sch, schema as sc, graph as gr, teacher as tc, engine
CAP = 1500
def ser(r): return json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n'
def load_roots(path):
    rs = [json.loads(l) for l in open(path)]
    for r in rs: r['identity'] = tuple(r['identity'])
    return rs

def make_runtime(provider, a, domain):
    """domain: acquisition.domains.get(...) object (contract, realizer, prompt addition, roots, amendment flag)."""
    schema, common = sc.build_schema(domain.contract)
    eng = SimpleNamespace(proposals=lambda q, meta, seen: engine.proposals(q, meta, seen, domain.realizer))
    return {'component': domain.contract, 'engine': eng, 'realizer': domain.realizer, 'schema': schema, 'common': common, 'provider': provider, 'addition': domain.prompt_addition, 'a': a}

def paired_items(meta): return [(typ, item) for typ, field, _ in sc.TYPES for item in meta[field]]

def kill_check(rows, a, n_exec):
    """Per-branch kill rules (frozen): first-attempt JSON validity < 95 % (>= 40 steps), failed steps > 5 %, Returns reason or None."""
    steps = [r for r in rows if r['call_type'] in ('LOCAL', 'API_INVALID_FAIL')]
    if len(steps) >= 40:
        if not getattr(a, 'amendment1', False):   # Medicine runner: first-attempt validity is a stop rule; Cybersecurity (amendment 1): diagnostic only
            first_valid = sum(1 for r in steps if r['model_calls'] and r['model_calls'][0]['valid']) / len(steps)
            if first_valid < 0.95: return f'first-attempt JSON/schema validity {first_valid:.3f} < 0.95 after {len(steps)} steps'
        if sum(1 for r in steps if r['call_type'] == 'API_INVALID_FAIL') / len(steps) > 0.05: return 'failed steps > 5 %'
    # fragmentation (NEW share) is a logged diagnostic in this study, not a kill rule (see PROTOCOL.md)
    return None

def counts_from_rows(rows, B):
    loc = [r for r in rows if r['call_type'] == 'LOCAL']; att = [m for r in rows for m in r.get('model_calls', [])]
    paid = [m for m in att if not m['reused_from_ledger']]
    return {'executed_acquisition_steps': len(loc), 'external_teacher_calls_nominal': len(att), 'external_teacher_calls_actually_paid_now': len(paid), 'reused_identical_request_calls': len(att) - len(paid), 'retries': sum(1 for m in att if m['attempt'] > 0),
            'failed_steps': sum(1 for r in rows if r['call_type'] == 'API_INVALID_FAIL'), 'realization_fail_steps_no_call': sum(1 for r in rows if r['call_type'] == 'REALIZATION_FAIL'), 'auxiliary_classifier_calls': 0, 'auxiliary_semantic_calls': 0, 'auxiliary_model_calls': 0,
            'invariant_calls_eq_executed_plus_failed_steps_plus_retries': len(att) == len(loc) + sum(1 for r in rows if r['call_type'] == 'API_INVALID_FAIL') + sum(1 for m in att if m['attempt'] > 0),
            'nominal_cost_usd': sum(m['cost_usd'] for m in att), 'actual_paid_cost_usd': sum(m['cost_usd'] for m in paid)}

def run(rt, root_path, out, store, B, checkpoints, hard_cost):
    a = rt['a']; A = getattr(a, 'amendment1', False); out = Path(out); out.mkdir(parents=True, exist_ok=True); (out / 'queue_snapshots').mkdir(exist_ok=True); (out / 'checkpoints').mkdir(exist_ok=True); store = Path(store); store.mkdir(parents=True, exist_ok=True)
    TRAJ = out / 'trajectory.jsonl'; QUEUE = out / 'queue_snapshots/queue_state.jsonl'; ledger = tc.CallLedger(store / 'model_call_ledger.jsonl'); blind = a.registry == 'hidden'
    component, engine, realizer = rt['component'], rt['engine'], rt['realizer']; policy = a.policy
    frontier = load_roots(root_path); seen = {tuple(s['identity']) for s in frontier}; graphs = {s['source_root_id']: gr.DirectionGraph(s['source_root_id']) for s in frontier}; counts = {}; seenq = []; rows = []; cost_nominal = 0.0; roots_by_id = {s['state_id']: s for s in frontier}; n_roots = len(roots_by_id); requeued = set(); DIAG = out / 'first_attempt_validity_diagnostic.jsonl'
    if TRAJ.exists():
        rows = [json.loads(l) for l in open(TRAJ) if l.strip()]
        for r in rows:
            if A and r['call_type'] == 'ROOT_REQUEUE': frontier.insert(0, roots_by_id[r['state']['state_id']]); requeued.add(r['state']['state_id']); continue
            s = r['state']; popped = sch.pop_next(frontier, counts, policy); assert popped['state_id'] == s['state_id'], f"resume drift at {s['state_id']}"
            cost_nominal += sum(m['cost_usd'] for m in r.get('model_calls', []))
            if r['call_type'] in ('REALIZATION_FAIL', 'API_INVALID_FAIL'): graphs[s['source_root_id']].mark_dropped(popped.get('direction_id')); continue
            g = graphs[s['source_root_id']]; ev, asg = g.apply_call(s['state_id'], r['call_index'], [(p['child_type'], p['item']) for p in r['proposal_items']], r['raw_proposals']); assert ev == r['graph_events'], f"graph replay mismatch at call {r['call_index']}"
            g.mark_executed(popped.get('direction_id')); sch.record_executed(counts, popped)
            for c in r['admitted_children']: assert asg.get(c['state_id']) == c['direction_id']
            seen.update(tuple(x['identity']) for x in r['admitted_children']); frontier.extend(r['admitted_children']); seenq.append(r['realized_question'])
    n_exec = sum(1 for r in rows if r['call_type'] == 'LOCAL'); call_i = n_exec + 1
    executed_ids = {r['state']['state_id'] for r in rows if r['call_type'] == 'LOCAL'}
    for r in (list(rows) if A else []):
        if r['call_type'] == 'API_INVALID_FAIL' and r['state']['generation_depth'] == 0 and r['state']['state_id'] not in requeued and r['state']['state_id'] not in executed_ids:
            sid = r['state']['state_id']; rows.append({'call_index': call_i, 'call_type': 'ROOT_REQUEUE', 'state': roots_by_id[sid], 'model_calls': [], 'note': 'AMENDMENT 1: pre-amendment failed root re-queued once'}); open(TRAJ, 'w').write(''.join(ser(x) for x in rows)); frontier.insert(0, roots_by_id[sid]); requeued.add(sid); print('ROOT_REQUEUE', sid, flush=True)
    def write_checkpoint(k):
        p = out / f'checkpoints/B{k}.json'
        if p.exists(): return
        pend = {}
        for x in frontier: pend[x['source_root_id']] = pend.get(x['source_root_id'], 0) + 1
        p.write_text(json.dumps({'budget': k, 'executed': n_exec, 'frontier_size': len(frontier), 'E_r': dict(counts), 'pending_per_root': pend, 'graphs': {r: g.snapshot() for r, g in graphs.items()}, 'trajectory_rows': len(rows), 'trajectory_sha256': hashlib.sha256(''.join(ser(r) for r in rows).encode()).hexdigest(), 'cost_nominal_usd': cost_nominal, 'counts': counts_from_rows(rows, B)}, indent=1))
    for k in checkpoints:
        if n_exec >= k: write_checkpoint(k)
    while n_exec < B:
        assert cost_nominal < hard_cost, f'HARD COST HALT {cost_nominal:.3f} >= {hard_cost}'
        if getattr(a, 'max_attempts', 0) and sum(len(r.get('model_calls', [])) for r in rows) >= rt['a'].max_attempts: print(f'ATTEMPT CAP HALT: {a.max_attempts} Teacher attempts reached after {n_exec} executed steps; B not reached, no top-up'); break
        if A and n_exec % 10 == 0:
            st = [r for r in rows if r['call_type'] in ('LOCAL', 'API_INVALID_FAIL')]
            if st and (not DIAG.exists() or json.loads(open(DIAG).read().strip().splitlines()[-1])['executed'] != n_exec):
                with open(DIAG, 'a') as f: f.write(json.dumps({'executed': n_exec, 'steps': len(st), 'first_attempt_validity': round(sum(1 for r in st if r['model_calls'] and r['model_calls'][0]['valid']) / len(st), 4), 'failed_steps': sum(1 for r in st if r['call_type'] == 'API_INVALID_FAIL')}) + '\n')
        why = kill_check(rows, a, n_exec) if n_exec % 10 == 0 else None
        if why: (out / 'KILL.json').write_text(json.dumps({'reason': why, 'executed': n_exec})); print('KILL RULE', why, flush=True); sys.exit(7)
        if not frontier: raise RuntimeError('frontier exhausted')
        pend = {}
        for x in frontier: pend[x['source_root_id']] = pend.get(x['source_root_id'], 0) + 1
        s = sch.pop_next(frontier, counts, policy)
        if A and s['generation_depth'] > 0 and sum(1 for r in rows if r['call_type'] == 'LOCAL' and r['state']['generation_depth'] == 0) < n_roots:
            (out / 'HALT.json').write_text(json.dumps({'reason': 'root-completion guard: a depth>0 state was selected before all roots have a QA row', 'state': s['state_id']})); print('ROOT GUARD HALT', s['state_id'], flush=True); sys.exit(9)
        with open(QUEUE, 'a') as f: f.write(json.dumps({'call_index': call_i, 'selected_state': s['state_id'], 'E_r': dict(counts), 'pending_per_root_before_pop': pend}, sort_keys=True) + '\n')
        root = s['source_root_id']; g = graphs[root]
        if s['generation_depth'] == 0: question = s['question']
        else:
            rr = realizer.realize(s, seen_questions=tuple(seenq))
            if rr['realization_status'] != 'PASS':
                g.mark_dropped(s.get('direction_id')); rows.append({'call_index': call_i, 'call_type': 'REALIZATION_FAIL', 'state': s, 'realization': rr, 'model_calls': []}); open(TRAJ, 'w').write(''.join(ser(r) for r in rows)); continue
            question = rr['deterministic_question']
        q = {**s, 'question': question, 'realized_question': question}
        reg_text = g.render()
        prompt_text = component.PROMPT_C + '\n\n' + (rt['addition'].replace('{direction_registry}', reg_text) if not blind else rt['addition']); rendered = component.render(q, prompt_text)
        known = g.ids(); reg_snapshot = g.snapshot()
        lsid = s['state_id'] + '#R1' if (s['state_id'] in requeued and s['generation_depth'] == 0) else s['state_id']
        if lsid != s['state_id']:
            old = {x['request_sha256'] for x in ledger.rows if x['state_id'] == s['state_id']}; assert tc.sha(rendered) in old, f"AMENDMENT 1: re-attempt prompt of {s['state_id']} differs from the failed prompt"; print('ROOT RE-ATTEMPT (same prompt, new ledger key)', lsid, flush=True)
        res = tc.propose_and_answer(rt['provider'], ledger, s['state_id'], rendered, rt['schema'], rt['common'], known, a.cap, blind=blind, ledger_state_id=lsid)
        cost_nominal += sum(m['cost_usd'] for m in res['attempts'])
        if res['obj'] is None:
            g.mark_dropped(s.get('direction_id')); rows.append({'call_index': call_i, 'call_type': 'API_INVALID_FAIL', 'state': s, 'error': res['error'], 'model_calls': res['attempts']}); open(TRAJ, 'w').write(''.join(ser(r) for r in rows)); print('INVALID_FAIL', s['state_id'], res['error'], flush=True)
            if A and s['generation_depth'] == 0:
                if s['state_id'] in requeued: (out / 'HALT.json').write_text(json.dumps({'reason': 'AMENDMENT 1: re-attempted root failed again; stop, no further attempts', 'state': s['state_id'], 'error': res['error']})); print('ROOT RE-ATTEMPT FAILED - HALT', s['state_id'], flush=True); sys.exit(8)
                rows.append({'call_index': call_i, 'call_type': 'ROOT_REQUEUE', 'state': roots_by_id[s['state_id']], 'model_calls': [], 'note': 'AMENDMENT 1: failed root re-queued once'}); open(TRAJ, 'w').write(''.join(ser(r) for r in rows)); frontier.insert(0, roots_by_id[s['state_id']]); requeued.add(s['state_id']); print('ROOT_REQUEUE', s['state_id'], flush=True)
            continue
        meta = res['obj']; raw, admitted = engine.proposals(q, meta, seen); items = paired_items(meta); assert len(items) == len(raw)
        events, asg = g.apply_call(s['state_id'], call_i, items, raw)
        for c in admitted: c['direction_id'] = asg[c['state_id']]
        g.mark_executed(s.get('direction_id')); sch.record_executed(counts, s); seen.update(x['identity'] for x in admitted); frontier.extend(admitted); seenq.append(question)
        rows.append({'call_index': call_i, 'call_type': 'LOCAL', 'state': s, 'realized_question': question, 'answer': meta['answer'], 'explanation': meta['explanation'], 'reusable_knowledge': meta['reusable_knowledge'], 'proposal_items': [{'child_type': t, 'item': it} for t, it in items],
                     'raw_proposals': raw, 'admitted_children': admitted, 'graph_events': events, 'registry_shown_to_teacher': reg_snapshot if not blind else None, 'dashboard_text': None, 'graph_after_root': g.snapshot(), 'prompt_sha256': tc.sha(rendered), 'model_calls': res['attempts']})
        open(TRAJ, 'w').write(''.join(ser(r) for r in rows)); n_exec += 1; call_i += 1
        if n_exec in checkpoints: write_checkpoint(n_exec)
        if n_exec % 10 == 0 or n_exec <= 3: print(f"[{a.branch}] {n_exec}/{B} root={root} depth={s['generation_depth']} prop={len(raw)} adm={len(admitted)} dirs={len(g.dirs)} nominal=${cost_nominal:.3f} paid_calls={sum(1 for m in (m for r in rows for m in r.get('model_calls', [])) if not m['reused_from_ledger'])}", flush=True)
    c = counts_from_rows(rows, B); json.dump(c, open(out / 'cost_ledger.json', 'w'), indent=1); json.dump({k: g.snapshot() for k, g in graphs.items()}, open(out / 'graph_state.json', 'w'), indent=1); return c

