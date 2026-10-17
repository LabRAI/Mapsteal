"""Command templates for Mode B (train+eval from frozen acquisition data) and Mode C (acquisition + train + eval) using the in-repo modules (src/mapsteal, src/baselines). All commands run from the repository root with PYTHONPATH=src.
Status vocabulary: VERIFIED (tested+reproduced) | AVAILABLE (execution path provided) | UNSUPPORTED (required original state/implementation unavailable)."""
import json
from pathlib import Path
PY = '${PYTHON:-python}'
SIZES = ('2B', '4B', '9B'); SEEDS = (42, 43, 44)
def mapsteal(size, dom, seed, mode):
    """MapSteal (src/mapsteal): Student-independent structured recursive acquisition (root states, deterministic realizer, one Teacher request per executed step, fair scheduler, direction registry)
    -> exact-B500 corpus -> FRESH LoRA (selected training configuration: lr 3e-4, 2 ep, r16/a32, d.05, MLP-only) -> frozen evaluation. Runs from the repository root with PYTHONPATH=src."""
    d = 'medicine' if dom == 'medicine' else 'cybersecurity'; run = f'runs/mapsteal/{size}/{d}/s{seed}'; corpus = f'data/frozen_corpora/mapsteal_{d}_B500.jsonl'
    ev = f'{PY} -m mapsteal evaluate --domain {d} --checkpoint {run}/adapter --student {size} --panels "${{MAPSTEAL_PANELS:?set MAPSTEAL_PANELS to the directory with the frozen benchmark panels}}" --output {run}/evaluation'
    train = [f'{PY} -m mapsteal train --corpus {corpus} --student {size} --output {run} --seed {seed}', ev]
    budget = 460 if d == 'medicine' else 500   # Medicine: B_total 500 = Seed40 cold start + 460 executed steps (495 rows after QC); Cybersecurity: 500 executed steps
    acq_run = f'runs/mapsteal_acquisition/{d}'; acq = [f'{PY} -m mapsteal acquire --domain {d} --budget {budget} --output {acq_run} --execute   # real GPT-4.1 Teacher calls (about $5 at full scale); writes {acq_run}/corpus.jsonl']
    full = acq + [f'{PY} -m mapsteal train --corpus {acq_run}/corpus.jsonl --student {size} --output {run}_fresh_acquisition --seed {seed}', ev.replace(f'{run}/', f'{run}_fresh_acquisition/').replace(f'--output {run}/evaluation', f'--output {run}_fresh_acquisition/evaluation')]
    return dict(train=train, env='src', corpus=corpus, acquisition=full, mode_b='AVAILABLE', mode_c='AVAILABLE', root=True,
                note_b='fresh LoRA from the archived corpus -> shared evaluator (docs/REPRODUCTION_GUIDE.md)',
                note_c='acquisition -> corpus -> training -> evaluation; a fresh run draws new Teacher outputs (a new trajectory, not the archived one)')

def _ev(d, size, adapter, out):
    return f'{PY} -m mapsteal evaluate --domain {d} --checkpoint {adapter} --student {size} --panels "${{MAPSTEAL_PANELS:?set MAPSTEAL_PANELS to the directory with the frozen benchmark panels}}" --output {out}'
def qedks(size, dom, seed, mode):
    """QEDKS-Adapted (src/baselines/qedks): Student-driven acquisition (original QEDKS code) -> QC corpus -> verified -> fresh LoRA (shared training configuration) -> shared evaluation; AS=TS formal pairing."""
    d = 'medicine' if dom == 'medicine' else 'cybersecurity'; t = 'med' if d == 'medicine' else 'cyb'; run = f'runs/qedks/{size}/{d}/s{seed}'; corpus = f'data/frozen_corpora/qedks/qedks_{d}_{size}_s{seed}.jsonl'
    man = f'provenance/qedks_runs/full_{t}_{size}_s{seed}/ACQ_MANIFEST.json'
    train = [f'{PY} -m baselines qedks train --corpus {corpus} --student {size} --output {run} --seed {seed} --manifest {man}', _ev(d, size, f'{run}/adapter', f'{run}/evaluation')]
    ar = f'runs/qedks_acquisition/{size}/{d}/s{seed}'
    acq = [f'{PY} -m baselines qedks acquire --domain {d} --student {size} --seed {seed} --target-qa 500 --budget-usd 3.0 --output {ar} --execute   # real GPT-4.1 Teacher calls + GPU; Student-driven',
           f'{PY} -m baselines qedks verify --run {ar} --domain {d} --student {size} --seed {seed} --write', f'{PY} -m baselines qedks train --corpus {ar}/training_corpus.jsonl --student {size} --output {run}_fresh_acquisition --seed {seed}',
           _ev(d, size, f'{run}_fresh_acquisition/adapter', f'{run}_fresh_acquisition/evaluation')]
    return dict(train=train, env='src', corpus=corpus, acquisition=acq, mode_b='AVAILABLE', mode_c='AVAILABLE', root=True,
                note_b='hash-verified archived corpus (data/frozen_corpora/qedks) -> fresh LoRA -> shared evaluator',
                note_c='needs sentence-transformers, scikit-learn, Levenshtein, rapidfuzz (pip install .[qedks]); Student-driven: a fresh run draws new Teacher outputs and a new trajectory')
def evokd(size, dom, seed, mode):
    """EvoKD-Adapted (src/baselines/evokd; formal v2): online continual adaptation, likelihood-proxy Identify (tau frozen), identity-adapter start, lr 2e-5, 63 chat events, 3,159 steps."""
    d = 'medicine' if dom == 'medicine' else 'cybersecurity'; run = f'runs/evokd/{size}/{d}/s{seed}'
    loop = f'{PY} -m baselines evokd run --domain {d} --student {size} --out {run} --events 63 --seed {seed} --budget-usd 1.5 --execute   # real GPT-4.1 Teacher calls + GPU online training'
    acq = [loop, f'{PY} -m baselines.evokd.trajectory_stats {run}', _ev(d, size, f'{run}/states/final/adapter', f'{run}/evaluation')]
    return dict(train=[], env='src', corpus=None, acquisition=acq, mode_b='UNSUPPORTED', mode_c='AVAILABLE', root=True,
                note_b='online continual adaptation interleaves Student-dependent Identify/generation with training; it cannot be replayed from the frozen QA alone.',
                note_c=f'a fresh end-to-end run draws NEW Teacher outputs (new independent trajectory). CPU dry run of the exact schedule: python -m baselines evokd run --domain {d} --student {size} --out DIR --dry-run (tested). Archived run records: archive/provenance/evokd/runs.')
def pristine(size, dom, seed, mode):
    d = 'medicine' if dom == 'medicine' else 'cybersecurity'
    ev = f'{PY} -m mapsteal evaluate --domain {d} --checkpoint none --student {size} --panels "${{MAPSTEAL_PANELS:?set MAPSTEAL_PANELS to the directory with the frozen benchmark panels}}" --output runs/pristine/{size}/{d}'
    return dict(train=[ev], env='src', corpus=None, acquisition=[], mode_b='AVAILABLE', mode_c='AVAILABLE', root=True,
                note_b='evaluation of the unadapted model (no training) with the MapSteal evaluator; the archived pristine predictions come from the original per-study evaluators (Mode A)')
METHODS = {'pristine': pristine, 'mapsteal': mapsteal, 'qedks': qedks, 'evokd': evokd}
LABEL = {'pristine': 'Pristine', 'mapsteal': 'MapSteal (Ours)', 'qedks': 'QEDKS-Adapted', 'evokd': 'EvoKD-Adapted'}
def plan(method, size, dom, seeds, kind):
    """kind: 'train' (Mode B: frozen acquisition -> train+eval), 'full' (Mode C: acquisition [+train+eval])."""
    out = []
    for s in ([None] if method == 'pristine' else seeds):
        w = METHODS[method](size, dom, s, kind)
        out.append(dict(method=method, size=size, domain=dom, seed=s, status=w['mode_b' if kind == 'train' else 'mode_c'], note=w.get('note_b' if kind == 'train' else 'note_c', ''), env_pythonpath=w['env'], corpus=w.get('corpus'),
                        root=w.get('root', False), commands=(w['train'] if kind == 'train' else (w['acquisition'] + w['train'] if method != 'evokd' else w['acquisition']))))
    return out
