#!/usr/bin/env python3
"""Replay-equivalence test (no network, no cost): run src/mapsteal acquisition with the ARCHIVED Teacher responses (ledger of the formal Medicine run) and compare every
trajectory row with the archived formal trajectory. Equal rows <=> the new implementation makes the same prompts (sha256), applies the same validation/admission/realization/
scheduling/direction-graph rules and writes the same records. Only `reused_from_ledger` (True here, False when the calls were paid) is normalised.
Run: python3 tests/test_replay_equivalence.py   (exit code 0 = identical)"""
import gzip, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 'src'))
from mapsteal.acquisition import branch, domains

class NoCallProvider:
    name = 'openai_responses'; model = 'gpt-4.1-2025-04-14'; price_in_per_m = 2.0; price_out_per_m = 8.0
    def complete(self, *a, **k): raise AssertionError('a Teacher call was attempted: the replay ledger did not cover this request (prompt differs from the archived one)')
    def cost_usd(self, r): return (r.input_tokens * self.price_in_per_m + r.output_tokens * self.price_out_per_m) / 1e6

def norm(rows):
    for r in rows:
        for m in r.get('model_calls', []): m['reused_from_ledger'] = False
    return rows

def main(domain):
    dom = domains.get(domain); arch = [json.loads(l) for l in gzip.open(HERE / f'fixtures/{domain}_trajectory_prefix.jsonl.gz', 'rt')]
    steps = sum(r['call_type'] == 'LOCAL' for r in arch)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t); (t / 'store').mkdir(); (t / 'store/model_call_ledger.jsonl').write_bytes(gzip.open(HERE / f'fixtures/{domain}_ledger_prefix.jsonl.gz').read())
        a = SimpleNamespace(branch='replay', policy='fair', registry='shown', dashboard=False, cap=branch.CAP, amendment1=dom.amendment1, max_attempts=0)
        rt = branch.make_runtime(NoCallProvider(), a, dom)
        counts = branch.run(rt, dom.roots, t / 'out', t / 'store', steps, [], 20.0)
        new = [json.loads(l) for l in open(t / 'out/trajectory.jsonl')]
        n, a_ = norm(new), norm(arch)
        bad = [i for i, (x, y) in enumerate(zip(n, a_)) if x != y]
        print(json.dumps({'domain': domain, 'steps': steps, 'archived_rows': len(a_), 'new_rows': len(n), 'rows_identical': len(n) == len(a_) and not bad, 'first_difference_row': bad[:3],
                          'teacher_calls_paid': counts['external_teacher_calls_actually_paid_now'], 'reused_calls': counts['reused_identical_request_calls']}))
        if bad:
            x, y = n[bad[0]], a_[bad[0]]; print('differing keys:', [k for k in set(x) | set(y) if x.get(k) != y.get(k)])
        return 0 if (len(n) == len(a_) and not bad) else 1
if __name__ == '__main__': sys.exit(max(main(d) for d in (sys.argv[1:] or ['medicine', 'cybersecurity'])))
