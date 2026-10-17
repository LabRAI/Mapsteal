#!/usr/bin/env python3
"""Unified entry point (called by run.sh).
Modes: aggregate (A: archived predictions -> Table 2; CPU), train (B: frozen acquisition data -> train + evaluate; GPU, no Teacher),
       full (C: acquisition + train + evaluate; GPU + Teacher API). Expensive modes only PRINT a plan unless --execute is given AND confirmed."""
import argparse, collections, hashlib, json, os, subprocess, sys
from pathlib import Path
from . import aggregate as agg, workflows as W
PKG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PKG / 'src'))
DOMS = {'medicine': 'Medicine', 'cybersecurity': 'Cybersecurity'}
def build_parser():
    ap = argparse.ArgumentParser(prog='run.sh', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--all', action='store_true', help='select every student/domain/method')
    ap.add_argument('--student', choices=W.SIZES, help='select one student scale'); ap.add_argument('--domain', choices=list(DOMS)); ap.add_argument('--method', choices=list(W.METHODS)); ap.add_argument('--seed', type=int, choices=W.SEEDS, help='restrict to one formal seed')
    ap.add_argument('--mode', choices=['aggregate', 'train', 'full'], default=None, help='aggregate (A, default), train (B), full (C)')
    ap.add_argument('--quiet', action='store_true', help='with --aggregate-only: compact output instead of the formatted tables'); ap.add_argument('--aggregate-only', action='store_true', help='Mode A only: rebuild Table 2 from archived predictions (no GPU, no Teacher)')
    ap.add_argument('--eval-only', action='store_true', help='with --mode train: run only the evaluation commands on adapters you supply (see README); skips acquisition/training')
    ap.add_argument('--list', action='store_true', help='print the status matrix (VERIFIED / AVAILABLE / UNSUPPORTED) and exit')
    ap.add_argument('--check', action='store_true', help='verify every selected command points to existing code and (Mode B) a hash-verified frozen corpus; executes nothing')
    ap.add_argument('--execute', action='store_true', help='actually run the selected GPU/Teacher commands (asks for confirmation)'); ap.add_argument('--yes', action='store_true', help='skip the interactive confirmation (still requires --execute)')
    ap.add_argument('--python', default=None, help='python interpreter for training/eval commands (default: python)'); ap.add_argument('--out', default=str(PKG / 'results'))
    return ap
def selection(a):
    ms = [a.method] if a.method else list(W.METHODS); ss = [a.student] if a.student else list(W.SIZES); ds = [a.domain] if a.domain else list(DOMS)
    return ms, ss, ds, ([a.seed] if a.seed else list(W.SEEDS))
def status_matrix():
    rows = []
    no_pred = not (PKG / 'data/gold/medqa.json').is_file()
    for m in W.METHODS:
        for s in W.SIZES:
            for d in DOMS:
                b = W.plan(m, s, d, [42], 'train')[0]; c = W.plan(m, s, d, [42], 'full')[0]
                cells = [] if no_pred else [r for r in agg.run()[3] if r['method'] == W.LABEL[m] and r['student'] == f'Qwen3.5-{s}' and r['domain'] == DOMS[d] and r['benchmark'] in ('MedMCQA', 'CyberMetric')]
                A = 'NEEDS GOLD (tools/prepare_benchmarks.py)' if no_pred else ('VERIFIED' if cells and all(c_['completeness'] == 'COMPLETE' for c_ in cells) else 'MISSING (no archived predictions)')
                rows.append((W.LABEL[m], s, DOMS[d], A, b['status'], c['status']))
    return rows
def corpus_check(entry_corpus, root):
    p = root / entry_corpus; return p.is_file()
def run_commands(plans, a, kind):
    env = dict(os.environ)
    if a.python: env['PYTHON'] = a.python
    n = sum(len(p['commands']) for p in plans if p['status'] != 'UNSUPPORTED')
    print(f"\nPLAN ({kind}): {len(plans)} configurations, {n} commands. Requires: GPU{' + Teacher API (OPENAI_API_KEY, spends real money)' if kind == 'full' else ' (no Teacher calls)'}; HF model snapshots; external benchmark data (README §Inputs).")
    for p in plans:
        print(f"\n[{p['status']}] {p['method']} {p['size']} {p['domain']} seed={p['seed']}" + (f"\n    NOTE: {p['note']}" if p['note'] else ''))
        for c in p['commands']: print('    $', c)
    runnable = [p for p in plans if p['status'] != 'UNSUPPORTED' and p['commands']]
    if not a.execute: print('\n(dry plan only; add --execute to run)'); return 0
    if not runnable: print('nothing runnable in this selection'); return 1
    if not a.yes:
        if not sys.stdin.isatty(): print('refusing: non-interactive session without --yes'); return 2
        if input(f"\nThis will launch {n} expensive commands. Type 'yes' to continue: ").strip() != 'yes': print('aborted'); return 2
    for p in runnable:
        base = PKG   # every command runs from the repository root (python -m mapsteal / python -m baselines)
        e = dict(env, PYTHONPATH=':'.join(str(base / q) for q in p['env_pythonpath'].split(':') if q) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
        for c in p['commands']:
            if c.lstrip().startswith('#') or '--help' in c: continue
            print('>>', c, flush=True); r = subprocess.run(['bash', '-c', c], cwd=base, env=e)
            if r.returncode: print('command failed:', r.returncode); return r.returncode
    return 0
def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.list:
        print(f"{'method':22}{'size':5}{'domain':15}{'A: aggregate':36}{'B: frozen acq.->train':26}{'C: end-to-end'}")
        for r in status_matrix(): print(f'{r[0]:22}{r[1]:5}{r[2]:15}{r[3]:36}{r[4]:26}{r[5]}')
        return 0
    mode = 'aggregate' if a.aggregate_only else (a.mode or 'aggregate')
    ms, ss, ds, seeds = selection(a)
    if not (a.all or a.method or a.student or a.domain or a.aggregate_only or a.check or a.mode):
        build_parser().print_help(); print('\nNothing selected. Try: run.sh --aggregate-only | run.sh --list | run.sh --all --check'); return 0
    if mode == 'aggregate':
        rc = agg.main(['--out', a.out] + (['--quiet'] if a.quiet else [])); return rc
    kind = mode; plans = []
    for m in ms:
        for s in ss:
            for d in ds: plans += W.plan(m, s, d, seeds, 'train' if kind == 'train' else 'full')
    if a.eval_only:
        for p in plans: p['commands'] = [c for c in p['commands'] if ' evaluate ' in c]
    if a.check:
        bad = 0; ok = 0; restr = 0
        for p in plans:
            if p['status'] == 'UNSUPPORTED': continue
            for c in p['commands']:
                base = PKG
                import shlex
                for mod, pf in (('mapsteal', 'mapsteal.cli'), ('baselines', 'baselines.cli')):
                    if f' -m {mod} ' in c:   # the public CLIs: parse the arguments with the real argparse (no execution)
                        import importlib
                        try:
                            toks = shlex.split(c.split('#')[0].replace('${PYTHON:-python}', 'python')); importlib.import_module(pf).build_parser().parse_args(toks[toks.index(mod) + 1:]); ok += 1
                        except SystemExit: bad += 1; print('CLI arguments do not parse:', c)
                if ' -m baselines.evokd.trajectory_stats ' in c:
                    import importlib.util; ok += importlib.util.find_spec('baselines.evokd.trajectory_stats') is not None
            if kind == 'train' and p['corpus']:
                base = PKG
                if (base / p['corpus']).is_file(): ok += 1
                elif p['corpus'].startswith('data/frozen_corpora/'): restr += 1
                else: bad += 1; print('MISSING corpus', p['corpus'])
        print(f'check: {ok} references resolved, {bad} missing' + (f', {restr} archived corpora are not distributed (see docs/REPRODUCTION_GUIDE.md, section Data)' if restr else '')); return 1 if bad else 0
    return run_commands(plans, a, kind)
if __name__ == '__main__': sys.exit(main())
