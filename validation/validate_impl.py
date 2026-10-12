#!/usr/bin/env python3
"""CPU-only validation of the in-repo implementation (src/), launcher and frozen acquisition data. Writes validation/IMPLEMENTATION_CHECK.md."""
import hashlib, json, os, re, subprocess, sys, tempfile
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]; L = []; ok_all = True; N_SKIP = 0
def skip(name, need):
    global N_SKIP; N_SKIP += 1; L.append(f'| {name} | SKIPPED | input not available: {need} (see docs/REPRODUCTION_GUIDE.md, section Data) |')
def chk(name, cond, detail=''):
    global ok_all; ok_all &= bool(cond); L.append(f"| {name} | {'PASS' if cond else 'FAIL'} | {detail} |")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
man = json.loads((PKG / 'manifests/table1_manifest.json').read_text())
# 1 ported baseline files equal the recorded hashes
mig = json.loads((PKG / 'manifests/baseline_source_map.json').read_text())['files']; badm = [f['new'] for f in mig if sha(PKG / f['new']) != f['new_sha256']]
chk('ported baseline files equal the hashes recorded in manifests/baseline_source_map.json (original sha256 + exact edits per file)', not badm and all('original_sha256' in f for f in mig), f'{len(mig)} files, {len(badm)} differ: {badm[:3]}')
# 2 training corpora present and hash-identical to the recorded training manifests
qm = json.loads((PKG / 'manifests/qedks_corpora_manifest.json').read_text())
have_corp = (PKG / 'data/frozen_corpora').is_dir()
seen = {}; miss = []; mism = []
for e in (man['entries'] if have_corp else []):
    c = e.get('training_corpus'); tm = e.get('training_manifest')
    if not c or c in seen: continue
    if not (PKG / c).is_file(): miss.append(c); continue
    d = json.loads((PKG / tm).read_text()).get('dataset_sha256') if tm else None; seen[c] = sha(PKG / c)
    if d and d != seen[c]: mism.append(c)
if have_corp:
    chk('frozen training corpora (MapSteal: data/frozen_corpora, QEDKS: data/frozen_corpora/qedks) present; sha256 equals dataset_sha256 recorded in the training manifests', not mism and not miss, f'{len(seen)} corpora checked, {len(mism)} mismatches, {len(miss)} missing')
    chk('QEDKS corpus sha256 equals output_sha256 in each archived ACQ_MANIFEST.json', all(json.loads((PKG / v['acq_manifest']).read_text())['output_sha256']['training_corpus.jsonl'] == sha(PKG / v['path']) == v['sha256'] for v in qm.values()), f'{len(qm)} corpora')
else:
    skip('frozen training corpora sha256 vs training manifests; QEDKS corpus sha256 vs ACQ_MANIFEST', 'data/frozen_corpora')
# 3 static imports
r = subprocess.run([sys.executable, str(PKG / 'tools/check_imports.py')], capture_output=True, text=True)
chk('every import in src/, mapsteal_repro/, validation/, tests/ resolves (stdlib / declared third-party / shipped module); no absolute machine paths; no stale directory reference in code', r.returncode == 0, r.stdout.replace('\n', ' | ')[:300])
# 3b CPU dry run of the formal EvoKD-Adapted schedule through the public CLI
d = tempfile.mkdtemp()
if (PKG / 'data/evokd/d0_root4_medicine.jsonl').exists():
    d = tempfile.mkdtemp(); r = subprocess.run([sys.executable, '-m', 'baselines', 'evokd', 'run', '--domain', 'medicine', '--out', d + '/ev', '--dry-run'], capture_output=True, text=True, cwd=PKG, env={**os.environ, 'PYTHONPATH': str(PKG / 'src')})
    chk('python -m baselines evokd run --dry-run executes the formal schedule on CPU (63 chat events, 3,159 steps)', r.returncode == 0 and re.search(r'.plan_steps.: 3159', r.stdout) is not None and re.search(r'.events.: 63', r.stdout) is not None, 'plan_steps 3159, events 63' if r.returncode == 0 else r.stderr[-200:])
else: skip('python -m baselines evokd run --dry-run (63 chat events, 3,159 steps)', 'data/evokd')
r = subprocess.run([sys.executable, '-m', 'baselines', 'qedks', 'acquire', '--domain', 'medicine', '--student', '4B', '--seed', '42', '--output', d + '/q'], capture_output=True, text=True, cwd=PKG, env={**os.environ, 'PYTHONPATH': str(PKG / 'src')})
chk('python -m baselines qedks acquire without --execute prints a plan only (no Teacher, no GPU)', r.returncode == 0 and 'dry plan only' in r.stdout)
# 4 launcher --check for B and C
for mode in ('train', 'full'):
    r = subprocess.run(['bash', str(PKG / 'run.sh'), '--all', '--mode', mode, '--check'], capture_output=True, text=True); chk(f'run.sh --all --mode {mode} --check resolves all scripts' + (' and corpora' if mode == 'train' else ''), r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
# 5 launcher smoke test with PYTHON=echo (executes nothing expensive), all selections; tracked files must be unchanged
def tree_hash():
    h = hashlib.sha256()
    for p in sorted(PKG.rglob('*')):
        if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix != '.pyc': h.update(str(p.relative_to(PKG)).encode()); h.update(p.read_bytes())
    return h.hexdigest()
before = tree_hash(); fails = []
for m in ('pristine', 'mapsteal', 'qedks', 'evokd'):
    for mode in ('train', 'full'):
        r = subprocess.run(['bash', str(PKG / 'run.sh'), '--method', m, '--mode', mode, '--execute', '--yes', '--python', 'echo'], capture_output=True, text=True, env={**os.environ, 'MAPSTEAL_PANELS': '/nonexistent/panels'})
        if r.returncode not in (0, 1) or (r.returncode == 1 and 'nothing runnable' not in r.stdout): fails.append((m, mode, r.returncode, r.stderr[-120:]))
chk('launcher smoke test (--execute --yes with PYTHON=echo, every method, modes B and C): runs end-to-end without launching training/Teacher', not fails, str(fails[:2]))
chk('smoke test left every package file byte-identical (no artifact overwritten)', before == tree_hash())
# 5b retired identifiers must not be accepted by the CLI
olds = ['unknown_method', 'evokd_adapted']; rej = [subprocess.run(['bash', str(PKG / 'run.sh'), '--method', o, '--list'], capture_output=True, text=True) for o in olds]
chk('CLI rejects unknown method identifiers (argparse invalid choice)', all(r.returncode != 0 and 'invalid choice' in r.stderr for r in rej))
# 6 non-interactive refusal without --yes
r = subprocess.run(['bash', str(PKG / 'run.sh'), '--method', 'mapsteal', '--student', '4B', '--domain', 'medicine', '--mode', 'train', '--execute'], capture_output=True, text=True, stdin=subprocess.DEVNULL)
chk('expensive execution refuses without confirmation in a non-interactive shell', r.returncode == 2 and 'refusing' in r.stdout)
# 7 aggregation CLI
if (PKG / 'data/gold/medqa.json').is_file():
    r = subprocess.run(['bash', str(PKG / 'run.sh'), '--aggregate-only', '--out', tempfile.mkdtemp()], capture_output=True, text=True); chk('run.sh --aggregate-only', r.returncode == 0 and 'COMPLETE' in r.stdout, r.stdout.strip().replace('\n', ' | ')[:200])
else:
    r = subprocess.run(['bash', str(PKG / 'run.sh'), '--aggregate-only', '--out', tempfile.mkdtemp()], capture_output=True, text=True); chk('run.sh --aggregate-only explains how to build the gold labels when they are absent', r.returncode != 0 and 'prepare_benchmarks' in (r.stderr + r.stdout))
    skip('run.sh --aggregate-only (Table 1 reconstruction)', 'data/gold (python tools/prepare_benchmarks.py)')
rep = '# Implementation / launcher validation (CPU only)\n\n| check | result | detail |\n|---|---|---|\n' + '\n'.join(L) + f'\n\n**Overall: {"NO FAILURE" if ok_all else "FAILURES PRESENT"} ({N_SKIP} checks skipped because their input data are not distributed)**\n\nNot tested here (no GPU / Teacher / model weights / external benchmark data in this environment): every Mode B and Mode C workflow itself.\n'
(PKG / 'validation/IMPLEMENTATION_CHECK.md').write_text(rep); print(rep); sys.exit(0 if ok_all else 1)
