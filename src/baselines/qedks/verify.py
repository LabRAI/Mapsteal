"""Verification of one finished QEDKS-Adapted acquisition run (CPU, 0 API); same checks as the formal verify_acquisition.py. Checks that need the raw Teacher/structured logs are reported as
'skipped' when those logs are absent (the public release ships only summary, corpus_raw, training corpus and ACQ_MANIFEST of the archived runs). Exit code 1 if any executed check fails.
usage: python -m baselines.qedks verify RUN_DIR <med|cyb> <2B|4B|9B> <acq_seed> [--write]"""
import collections, hashlib, json
from pathlib import Path
from . import corpus
P = Path(__file__).resolve().parents[3]
DOMS = {'med': {'qedks': 'medical', 'name': 'Medicine'}, 'cyb': {'qedks': 'cyber', 'name': 'Cybersecurity'}}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]

def code_hashes():
    exp = json.load(open(P / 'manifests/qedks_code_hashes.json'))['files']
    return exp, {f: sha(Path(__file__).parent / f) for f in exp}

def verify(run, dom, size, seed, write=False):
    run = Path(run); C = {}
    def chk(name, ok, detail=None, skipped=False): C[name] = {'ok': bool(ok), 'detail': detail, 'skipped': skipped}
    summ = json.load(open(run / 'summary.json'))
    chk('reached_target_exactly_500', summ['reached_target'] and summ['qa_rows'] == 500 and summ['target_qa'] == 500 and not summ['frontier_exhausted'], {'qa_rows': summ['qa_rows'], 'iterations': summ['iterations']})
    chk('domain_and_seed_match', summ['domain'] == DOMS[dom]['qedks'] and summ['seed'] == seed, {'domain': summ['domain'], 'seed': summ['seed']})
    raw = jl(run / 'corpus_raw.jsonl'); chk('corpus_500_rows_no_empty_answers', len(raw) == 500 and all(r['output'].strip() for r in raw), {'rows': len(raw)})
    chk('main_plus_correction_equals_500', summ['main_rows'] + summ['correction_rows'] == 500, {'main': summ['main_rows'], 'correction': summ['correction_rows']})
    led_p, log_p = run / 'teacher_ledger.jsonl', run / 'structured_log.jsonl'; kinds = collections.Counter(); cost = None
    if led_p.exists() and log_p.exists():
        led = jl(led_p); keys = [r['key'] for r in led]
        chk('ledger_requests_unique', len(keys) == len(set(keys)), {'requests': len(led)})
        chk('ledger_models_temperature', {r['model'] for r in led} == {'gpt-4.1'} and {r['temperature'] for r in led} == {0.0})
        kinds = collections.Counter(r.get('kind') for r in led); chk('all_requests_labelled', None not in kinds, dict(kinds))
        cost = sum(r['cost_usd'] for r in led); chk('cost_below_cap', cost < 3.0, {'cost_usd': round(cost, 4)})
        log = jl(log_p); its = [r['iteration'] for r in log]
        chk('iterations_contiguous', its == list(range(1, len(its) + 1)) and len(its) == summ['iterations'], {'n': len(its)})
        qa = [r['qa_rows_after'] for r in log]; chk('500_first_reached_at_last_iteration', qa[-1] == 500 and all(x < 500 for x in qa[:-1]) and all(b >= a for a, b in zip(qa, qa[1:])))
        chk('no_extraction_request_after_500th_row', kinds.get('entity_extraction', 0) in (len(log) - 1, len(log)), {'extraction': kinds.get('entity_extraction', 0), 'iterations': len(log)})
        chk('ledger_failed_requests_none', all(r.get('text') is not None for r in led))
    else:
        for n in ('ledger_requests_unique', 'ledger_models_temperature', 'all_requests_labelled', 'cost_below_cap', 'iterations_contiguous', '500_first_reached_at_last_iteration', 'no_extraction_request_after_500th_row', 'ledger_failed_requests_none'):
            chk(n, True, 'raw Teacher/structured logs not present in this directory', skipped=True)
    exp, now = code_hashes(); chk('code_hashes_equal_frozen', now == exp, [f for f in exp if now[f] != exp[f]])
    keep = corpus.export_rows(raw); data = corpus.dump(keep)
    info = {'training_rows': len(keep), 'distinct_instructions': len({r['instruction'] for r in keep}), 'training_corpus_sha256': hashlib.sha256(data.encode()).hexdigest()}
    chk('training_corpus_500_rows', len(keep) == 500, info)
    t = run / 'training_corpus.jsonl'
    if t.exists(): chk('training_corpus_file_equals_export_of_raw', t.read_bytes() == data.encode(), {'sha256': sha(t)})
    passed = all(v['ok'] for v in C.values())
    if write:
        (run / 'training_corpus.jsonl').write_text(data)
        json.dump({'passed': passed, 'checks': C, **info}, open(run / 'ACQ_VERIFY.json', 'w'), indent=1)
    return {'passed': passed, 'checks': C, **info}
