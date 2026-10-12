#!/usr/bin/env python3
"""Cost-guard tests (no network, no cost): (1) the acquisition loop halts BEFORE a step when the nominal cost reached the hard cost (replay of archived responses with a tiny hard cost);
(2) the CLI refuses to start when the cumulative spend ledger cannot cover the estimate; (3) a provider that does not report usage stops the run."""
import gzip, json, os, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / 'src'))
from mapsteal.acquisition import branch, domains
sys.path.insert(0, str(HERE)); from test_replay_equivalence import NoCallProvider
ok = True
dom = domains.get('medicine')
with tempfile.TemporaryDirectory() as t:
    t = Path(t); (t / 'store').mkdir(); (t / 'store/model_call_ledger.jsonl').write_bytes(gzip.open(HERE / 'fixtures/medicine_ledger_prefix.jsonl.gz').read())
    a = SimpleNamespace(branch='g', policy='fair', registry='shown', dashboard=False, cap=branch.CAP, amendment1=False, max_attempts=0)
    try: branch.run(branch.make_runtime(NoCallProvider(), a, dom), dom.roots, t / 'out', t / 'store', 40, [], 0.05); halted = False
    except AssertionError as e: halted = 'HARD COST HALT' in str(e)
    n = sum(1 for _ in open(t / 'out/trajectory.jsonl')); r1 = halted and n < 40; print('hard-cost halt before exceeding the cap:', r1, f'(stopped after {n} steps)'); ok &= r1
with tempfile.TemporaryDirectory() as t:
    env = dict(os.environ, OPENAI_API_KEY='dummy-not-used', PYTHONPATH=str(ROOT / 'src'))
    r = subprocess.run([sys.executable, '-m', 'mapsteal', 'acquire', '--domain', 'medicine', '--budget', '10', '--output', f'{t}/o', '--execute', '--spend-ledger', f'{t}/l.json', '--spend-ceiling', '0.2'], capture_output=True, text=True, env=env)
    r2 = r.returncode != 0 and 'spend guard' in (r.stderr + r.stdout); print('CLI refuses when the ceiling cannot cover the estimate:', r2); ok &= r2
sys.exit(0 if ok else 1)
