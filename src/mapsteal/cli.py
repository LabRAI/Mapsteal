"""MapSteal command line. Subcommands follow the pipeline: acquire -> (corpus) -> train -> evaluate. All outputs go to a user-chosen directory; formal artifacts are never touched.
  python -m mapsteal acquire  --domain medicine --budget 10 --output runs/smoke --execute
  python -m mapsteal corpus   --run runs/smoke --output runs/smoke/corpus.jsonl
  python -m mapsteal train    --corpus runs/smoke/corpus.jsonl --student 2B --output runs/smoke/student
  python -m mapsteal evaluate --checkpoint runs/smoke/student/adapter --student 2B --panels PANELS_DIR --output runs/smoke/evaluation --limit 100"""
import argparse, json, os, subprocess, sys
from pathlib import Path

EPILOG = """Examples (Medicine):
  python -m mapsteal acquire  --domain medicine --budget 10 --output runs/smoke --execute      # real GPT-4.1 Teacher calls (needs OPENAI_API_KEY); without --execute only a plan and a cost estimate is printed
  python -m mapsteal train    --corpus runs/smoke/corpus.jsonl --student 2B --output runs/smoke/student          # fresh LoRA, selected training configuration (GPU)
  python -m mapsteal evaluate --checkpoint runs/smoke/student/adapter --student 2B --panels /path/to/panels --output runs/smoke/evaluation [--limit 100]
Formal results: `bash run.sh --aggregate-only` rebuilds Table 1 from the archived predictions (no GPU, no Teacher). See README.md and docs/REPRODUCTION_GUIDE.md."""

def _refuse_formal(path):
    p = Path(path).resolve(); root = Path(__file__).resolve().parents[2]
    for guarded in ('data', 'archive', 'manifests', 'validation', 'configs'):
        if (root / guarded) in p.parents or p == root / guarded: raise SystemExit(f'refusing to write into archived/formal artifacts: {p}')

def _load_env(env_file):
    """Optional KEY=VALUE file (never printed, never logged); existing environment variables win."""
    if not env_file: return
    for line in Path(env_file).read_text().splitlines():
        line = line.strip()
        if line.startswith('export '): line = line[7:].strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def cmd_acquire(a):
    from .acquisition import branch, teacher as tc, domains
    from .corpus import build as cb
    from . import cost, paths
    from types import SimpleNamespace
    dom = domains.get(a.domain); _refuse_formal(a.output); seed_path = None if (a.no_corpus or a.domain != 'medicine') else paths.resolve_seed(a.seed_corpus); out = Path(a.output); roots = Path(a.roots) if a.roots else dom.roots
    est = a.budget * cost.EST_USD_PER_STEP
    print(json.dumps({'plan': {'domain': a.domain, 'budget_B_executed_QA_steps': a.budget, 'policy': a.policy, 'registry': 'shown', 'teacher': 'gpt-4.1-2025-04-14, temperature 0, one request per executed step (+counted retries)',
                      'roots': str(roots), 'output': str(out), 'estimated_cost_usd': round(est, 3), 'spend_ceiling_usd': a.spend_ceiling}}, indent=1))
    if not a.execute: print('dry plan only: add --execute for real Teacher calls'); return 0
    _load_env(a.env_file)
    if not os.environ.get('OPENAI_API_KEY'): raise SystemExit('OPENAI_API_KEY is not set (export it or pass --env-file); the key is never printed')
    led = cost.SpendLedger(a.spend_ledger, a.spend_ceiling); key = str(out.resolve()); remaining = led.remaining_for(key)
    if remaining - est < cost.MARGIN: raise SystemExit(f'spend guard: remaining ceiling ${remaining:.2f} cannot cover the estimate ${est:.2f}; refusing to start')
    class Guarded(tc.OpenAIResponsesProvider):
        n_requests = 0
        def complete(self, prompt, schema, max_output_tokens):
            r = super().complete(prompt, schema, max_output_tokens); Guarded.n_requests += 1
            if r.input_tokens <= 0 or r.output_tokens <= 0: raise RuntimeError('API usage was not reported: cost cannot be accounted, stopping before further spending')
            return r
    ns = SimpleNamespace(branch=out.name, policy=a.policy, registry='shown', dashboard=False, cap=branch.CAP, amendment1=dom.amendment1, max_attempts=0)
    rt = branch.make_runtime(Guarded(), ns, dom); hard = max(remaining - cost.MARGIN, 0.0)
    try: counts = branch.run(rt, roots, out, out, a.budget, [], hard)
    finally:
        rows = [json.loads(l) for l in open(out / 'trajectory.jsonl')] if (out / 'trajectory.jsonl').exists() else []
        spent = sum(m['cost_usd'] for r in rows for m in r.get('model_calls', []) if not m['reused_from_ledger']); led.record(key, spent)
    report = acquisition_report(out, counts); (out / 'acquisition_report.json').write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))
    if not a.no_corpus:
        rows_, rep = cb.build_from_run(out, out / 'corpus.jsonl', seed_path); print('corpus:', json.dumps(rep))
    return 0

def acquisition_report(out, counts):
    rows = [json.loads(l) for l in open(Path(out) / 'trajectory.jsonl')]; loc = [r for r in rows if r['call_type'] == 'LOCAL']; att = [m for r in rows for m in r.get('model_calls', [])]
    paid = [m for m in att if not m['reused_from_ledger']]; depth = {}
    for r in loc: depth[r['state']['generation_depth']] = depth.get(r['state']['generation_depth'], 0) + 1
    return {'executed_steps_QA_records': len(loc), 'rejected_or_failed_steps': {t: sum(1 for r in rows if r['call_type'] == t) for t in ('REALIZATION_FAIL', 'API_INVALID_FAIL')},
            'api_requests_total': len(att), 'api_requests_paid_this_invocation_or_before': len(paid), 'retries': sum(1 for m in att if m['attempt'] > 0),
            'input_tokens': sum(m['usage']['input_tokens'] for m in paid), 'output_tokens': sum(m['usage']['output_tokens'] for m in paid), 'estimated_cost_usd_from_reported_usage': round(sum(m['cost_usd'] for m in paid), 6),
            'children_proposed': sum(len(r['raw_proposals']) for r in loc), 'children_admitted': sum(len(r['admitted_children']) for r in loc), 'executed_by_depth': depth, 'cost_ledger': counts}

def cmd_corpus(a):
    from .corpus import build as cb, validate as cv
    from . import paths
    _refuse_formal(a.output); seed = paths.resolve_seed(a.seed_corpus)   # Medicine only (the Cybersecurity corpora have no seed rows)
    rows, rep = cb.build_from_run(a.run, a.output, seed, a.steps); print(json.dumps(rep, indent=1))
    if a.validate: v = cv.validate_run(a.run, a.output, seed); Path(str(a.output) + '.validation.json').write_text(json.dumps(v, indent=1)); print(json.dumps(v, indent=1)); return 0 if v['all_checks_pass'] else 1
    return 0

def _isolated(argv, device):
    """Run the GPU subcommand inside the formal single-GPU isolation child (same pattern as the formal experiments)."""
    from .gpu_isolation import resolve_device_identity, child_environment, is_isolated_child, ISOLATION_MARKER
    if is_isolated_child(): return None
    ident = resolve_device_identity(device); env = child_environment(ident, base_env=dict(os.environ)); env[ISOLATION_MARKER] = '1'
    if not os.environ.get('MAPSTEAL_ALLOW_HF_NETWORK'): env.setdefault('HF_HUB_OFFLINE', '1'); env.setdefault('HF_DATASETS_OFFLINE', '1')
    return subprocess.run([sys.executable, '-m', 'mapsteal', *argv], env=env).returncode

def cmd_train(a, argv):
    _refuse_formal(a.output); rc = _isolated(argv, a.device)
    if rc is not None: return rc
    from .training import train as tr
    man = tr.train(a.corpus, a.output, a.student, a.seed, 'cuda:0')
    ver = tr.verify_adapter(Path(a.output) / 'adapter', a.student, 'cuda:0'); man['adapter_reload_check'] = ver; json.dump(man, open(Path(a.output) / 'run_manifest.json', 'w'), indent=1, default=str)
    print(json.dumps({k: man[k] for k in ('rows', 'optimizer_steps', 'trainable_parameters', 'final_loss', 'all_losses_finite', 'adapter_sha256', 'train_seconds')} | {'reload': ver}, indent=1))
    return 0 if (ver['counts_match'] and ver['only_lora_trainable'] and ver['adapter_weights_finite'] and man['all_losses_finite']) else 1

def cmd_evaluate(a, argv):
    _refuse_formal(a.output); rc = _isolated(argv, a.device)
    if rc is not None: return rc
    if a.domain == 'medicine':
        from .evaluation import medicine as ev
        bm = a.benchmarks or 'medmcqa,medqa,medxpertqa_text'
    else:
        from .evaluation import cybersecurity as ev
        bm = a.benchmarks or 'cybermetric,seceval,cti_bench'
    ev.evaluate(a.checkpoint, a.student, a.panels, a.output, tuple(bm.split(',')), a.limit, 'cuda:0'); return 0

def build_parser():
    ap = argparse.ArgumentParser(prog='python -m mapsteal', description='MapSteal: structured recursive Teacher-driven knowledge acquisition -> corpus -> Student adaptation -> evaluation.', epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    q = sp.add_parser('acquire', help='run the formal acquisition (real Teacher calls with --execute) and build the corpus', description=cmd_acquire.__doc__)
    q.add_argument('--domain', choices=['medicine', 'cybersecurity'], required=True, help='formal domain profile: medicine (35 roots) | cybersecurity (domain-neutral contract, 40 roots)'); q.add_argument('--budget', type=int, required=True, help='B = number of executed (successful) acquisition steps = QA records, NOT the number of API requests')
    q.add_argument('--output', required=True); q.add_argument('--roots', help='root manifest (default: the 35 frozen Medicine roots)'); q.add_argument('--policy', choices=['fair', 'fifo'], default='fair')
    q.add_argument('--execute', action='store_true', help='make real Teacher API calls (costs money)'); q.add_argument('--env-file', help='optional KEY=VALUE file providing OPENAI_API_KEY (never printed)')
    q.add_argument('--spend-ledger', default='runs/.spend_ledger.json'); q.add_argument('--spend-ceiling', type=float, default=5.0, help='cumulative USD ceiling over all runs recorded in the ledger (default 5.0)')
    q.add_argument('--seed-corpus', default='auto', help="Medicine only: cold-start rows prepended to corpus.jsonl: 'auto' (the 40 formal rows if present, else none), 'none', or a JSONL file with input/instruction/output rows; ignored for Cybersecurity"); q.add_argument('--no-corpus', action='store_true')
    c = sp.add_parser('corpus', help='build / validate the training corpus from an acquisition run'); c.add_argument('--run', required=True); c.add_argument('--output', required=True); c.add_argument('--steps', type=int)
    c.add_argument('--seed-corpus', default='none', help="'medicine' (packaged cold-start rows, not distributed), 'none', or a JSONL file of cold-start rows to prepend"); c.add_argument('--validate', action='store_true', help='also write the machine-readable corpus validation report')
    t = sp.add_parser('train', help='fresh LoRA adaptation of a pristine Student with the selected training configuration (GPU)'); t.add_argument('--corpus', required=True); t.add_argument('--student', required=True, help='2B | 4B | 9B | Qwen/Qwen3.5-<size>')
    t.add_argument('--output', required=True); t.add_argument('--seed', type=int, default=42); t.add_argument('--device', default='cuda:0')
    e = sp.add_parser('evaluate', help='frozen evaluation of a checkpoint on the Table 1 benchmarks of a domain (GPU)'); e.add_argument('--domain', choices=['medicine', 'cybersecurity'], default='medicine'); e.add_argument('--checkpoint', required=True, help="adapter directory or 'none' (pristine)"); e.add_argument('--student', required=True)
    e.add_argument('--panels', required=True, help='directory with the frozen panel files (built by tools/prepare_benchmarks.py)'); e.add_argument('--output', required=True); e.add_argument('--benchmarks', help='comma-separated subset (default: all three of the domain)')
    e.add_argument('--limit', type=int, help='evaluate only the first N items of each benchmark (labelled subset; smoke tests)'); e.add_argument('--device', default='cuda:0')
    return ap

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a = build_parser().parse_args(argv)
    return {'acquire': lambda: cmd_acquire(a), 'corpus': lambda: cmd_corpus(a), 'train': lambda: cmd_train(a, argv), 'evaluate': lambda: cmd_evaluate(a, argv)}[a.cmd]()
if __name__ == '__main__': sys.exit(main())
