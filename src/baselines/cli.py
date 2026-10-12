"""Baselines command line (the evaluation of every baseline is the shared `python -m mapsteal evaluate`).
  python -m baselines qedks   acquire --domain medicine --student 4B --seed 42 --output runs/qedks/med4B_s42 [--execute]   # Student-driven acquisition (real Teacher calls + GPU only with --execute)
  python -m baselines qedks   corpus  --run RUN --output RUN/training_corpus.jsonl
  python -m baselines qedks   verify  --run RUN --domain medicine --student 4B --seed 42 [--write]
  python -m baselines qedks   train   --corpus RUN/training_corpus.jsonl --student 4B --output runs/qedks/train_s42 [--manifest RUN/ACQ_MANIFEST.json]
  python -m baselines evokd   run     --domain medicine --student 4B --out runs/evokd/med_s42 [--dry-run | --execute]       # online continual adaptation, 63 chat events
  python -m baselines pristine evaluate --domain medicine --student 4B --panels PANELS --output runs/pristine/4B/medicine"""
import argparse, json, os, subprocess, sys
from pathlib import Path
ACK = '--i-understand-this-spends-real-teacher-tokens-and-gpu-time'
DOM = {'medicine': ('medical', 'med', 'med'), 'cybersecurity': ('cyber', 'cyb', 'cyber')}   # QEDKS name, short tag, EvoKD domain id

def _env(env_file):
    from mapsteal.cli import _load_env; _load_env(env_file)

def _gpu_child(cmd, device='cuda:0', cwd=None, extra_env=None):
    from mapsteal.gpu_isolation import resolve_device_identity, child_environment, ISOLATION_MARKER
    env = child_environment(resolve_device_identity(device), base_env=dict(os.environ)); env[ISOLATION_MARKER] = '1'; env.update(extra_env or {})
    if not os.environ.get('MAPSTEAL_ALLOW_HF_NETWORK'): env.setdefault('HF_HUB_OFFLINE', '1'); env.setdefault('HF_DATASETS_OFFLINE', '1')
    src = str(Path(__file__).resolve().parents[1]); env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run(cmd, env=env, cwd=cwd).returncode

def qedks_acquire(a):
    out = Path(a.output).resolve(); qd = DOM[a.domain][0]
    plan = {'plan': {'method': 'QEDKS-Adapted acquisition', 'domain': qd, 'student': a.student, 'seed': a.seed, 'target_qa_rows': a.target_qa, 'teacher': 'gpt-4.1 (temperature 0)', 'budget_usd_cap': a.budget_usd, 'output': str(out)}}
    print(json.dumps(plan, indent=1))
    if not a.execute: print('dry plan only: add --execute for real Teacher calls and GPU use'); return 0
    _env(a.env_file)
    if not os.environ.get('OPENAI_API_KEY'): raise SystemExit('OPENAI_API_KEY is not set (export it or pass --env-file); the key is never printed')
    scratch = out / '_scratch'; scratch.mkdir(parents=True, exist_ok=True)   # the original code writes expv21.1/ and checkpoints/ into the working directory
    cmd = [sys.executable, '-u', '-m', 'baselines.qedks.trainStudentv21', '--domain', qd, '--size', a.student, '--seed', str(a.seed), '--target-qa', str(a.target_qa), '--budget-usd', str(a.budget_usd), '--out', str(out), '--device', 'cuda:0', ACK]
    return _gpu_child(cmd, a.device, cwd=scratch)

def qedks_corpus(a):
    from .qedks import corpus; print(json.dumps(corpus.export_corpus(a.run, a.output), indent=1)); return 0

def qedks_verify(a):
    from .qedks import verify
    r = verify.verify(a.run, DOM[a.domain][1], a.student, a.seed, a.write); print(json.dumps({'passed': r['passed'], **{k: ('skipped' if v['skipped'] else v['ok']) for k, v in r['checks'].items()}}, indent=1)); return 0 if r['passed'] else 1

def qedks_train(a, argv):
    from mapsteal.gpu_isolation import is_isolated_child
    if not is_isolated_child(): return _gpu_child([sys.executable, '-m', 'baselines', *argv], a.device)
    from .qedks import train
    man = train.train(a.corpus, a.output, a.student, a.seed, 'cuda:0', manifest=a.manifest); print(json.dumps({k: man[k] for k in ('rows', 'expected_optimizer_steps', 'trainable_parameters', 'adapter_sha256', 'train_seconds')}, indent=1)); return 0

def evokd_run(a):
    d0 = Path(a.d0) if a.d0 else Path(__file__).resolve().parents[2] / 'data/evokd' / ('d0_root4_medicine.jsonl' if a.domain == 'medicine' else 'd0_root4_cyber.jsonl')
    if not d0.is_file(): raise SystemExit(f'initial-state file not found: {d0}. Provide --d0 FILE: 4 JSONL rows with keys dataset, knowledge_object, question, reference_answer, source_id (see docs/REPRODUCTION_GUIDE.md).')
    env = {'EVOKD_V2_DOMAIN': DOM[a.domain][2], 'EVOKD_V2_SIZE': a.student}
    cmd = [sys.executable, '-m', 'baselines.evokd.run', '--out', str(a.output), '--events', str(a.events), '--seed', str(a.seed), '--budget-usd', str(a.budget_usd)] + (['--dry-run'] if a.dry_run else []) + ['--d0', str(d0)]
    if not a.dry_run:
        if not a.execute: print(json.dumps({'plan': {'method': 'EvoKD-Adapted', 'domain': a.domain, 'student': a.student, 'chat_events': a.events, 'output': str(a.output)}}, indent=1)); print('dry plan only: use --dry-run for the CPU schedule check, --execute for real Teacher calls and GPU training'); return 0
        _env(a.env_file); cmd += ['--device', 'cuda:0', ACK]
    cmd += list(a.extra)
    src = str(Path(__file__).resolve().parents[1]); e = dict(os.environ, **env, PYTHONPATH=src + os.pathsep + os.environ.get('PYTHONPATH', ''))
    return subprocess.run(cmd, env=e).returncode   # run.py performs its own GPU isolation re-exec

def pristine_evaluate(a):
    from . import pristine; return pristine.evaluate(a.domain, a.student, a.panels, a.output, a.benchmarks, a.limit, a.device)

def build_parser():
    ap = argparse.ArgumentParser(prog='python -m baselines', description='Baselines of the MapSteal comparison: QEDKS-Adapted, EvoKD-Adapted, Pristine.', epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest='method', required=True)
    q = s.add_parser('qedks'); qs = q.add_subparsers(dest='cmd', required=True); dom = dict(choices=list(DOM), required=True); stu = dict(choices=['2B', '4B', '9B'], required=True)
    p = qs.add_parser('acquire'); p.add_argument('--domain', **dom); p.add_argument('--student', **stu); p.add_argument('--seed', type=int, required=True); p.add_argument('--output', required=True); p.add_argument('--target-qa', type=int, default=500)
    p.add_argument('--budget-usd', type=float, default=3.0); p.add_argument('--execute', action='store_true'); p.add_argument('--env-file'); p.add_argument('--device', default='cuda:0')
    p = qs.add_parser('corpus'); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    p = qs.add_parser('verify'); p.add_argument('--run', required=True); p.add_argument('--domain', **dom); p.add_argument('--student', **stu); p.add_argument('--seed', type=int, required=True); p.add_argument('--write', action='store_true')
    p = qs.add_parser('train'); p.add_argument('--corpus', required=True); p.add_argument('--student', **stu); p.add_argument('--output', required=True); p.add_argument('--seed', type=int, default=42); p.add_argument('--manifest'); p.add_argument('--device', default='cuda:0')
    e = s.add_parser('evokd'); es = e.add_subparsers(dest='cmd', required=True); p = es.add_parser('run'); p.add_argument('--domain', **dom); p.add_argument('--student', default='4B', choices=['2B', '4B', '9B']); p.add_argument('--out', '--output', dest='output', required=True)
    p.add_argument('--events', type=int, default=63); p.add_argument('--seed', type=int, default=42); p.add_argument('--budget-usd', type=float, default=1.5); p.add_argument('--dry-run', action='store_true'); p.add_argument('--execute', action='store_true'); p.add_argument('--env-file'); p.add_argument('--d0', help='initial-state file (4 rows); default data/evokd/d0_root4_<domain>.jsonl')
    p.add_argument('extra', nargs=argparse.REMAINDER, help='extra arguments passed to baselines.evokd.run (after --)')
    r = s.add_parser('pristine'); rs = r.add_subparsers(dest='cmd', required=True); p = rs.add_parser('evaluate'); p.add_argument('--domain', **dom); p.add_argument('--student', **stu); p.add_argument('--panels', required=True); p.add_argument('--output', required=True)
    p.add_argument('--benchmarks'); p.add_argument('--limit', type=int); p.add_argument('--device', default='cuda:0')
    return ap

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv; a = build_parser().parse_args(argv)
    if a.method == 'qedks': return {'acquire': lambda: qedks_acquire(a), 'corpus': lambda: qedks_corpus(a), 'verify': lambda: qedks_verify(a), 'train': lambda: qedks_train(a, argv)}[a.cmd]()
    if a.method == 'evokd': return evokd_run(a)
    return pristine_evaluate(a)
if __name__ == '__main__': sys.exit(main())
