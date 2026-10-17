#!/usr/bin/env python3
"""Mode A: rebuild Table 2 from archived per-item predictions. No GPU, no Teacher, no training. stdlib only.
Reads manifests/table2_manifest.json, data/gold/*.json, data/predictions/**. Never reads any previously rounded Table 2 value.
Scoring rule (frozen evaluator protocol): a prediction is correct iff its option letter equals the frozen gold letter (or is in gold_set); accuracy = correct / frozen n.
A run/benchmark is VERIFIED only if: item-id set == frozen panel minus frozen exclusions (no missing, duplicated or unexpected ids), prediction letters are allowed option letters,
the gold stored in the prediction row equals the frozen gold, and the stored `correct` flag (if present) agrees. Otherwise no accuracy is produced (never scored as wrong).
Mean and sample SD (ddof=1) are taken over formal seeds of VERIFIED runs; a cell is COMPLETE only with all expected seeds (3 adapted, 1 pristine)."""
import argparse, csv, collections, hashlib, json, statistics, sys
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]
METHODS = ['Pristine', 'MapSteal (Ours)', 'QEDKS-Adapted', 'EvoKD-Adapted']
LEGACY_METHOD_LABELS = {'MapSteal (Ours, F2)': 'MapSteal (Ours)'}   # label used by older manifests / reference files
def method_label(m): return LEGACY_METHOD_LABELS.get(m, m)
BENCH = {'Medicine': ['MedMCQA', 'MedQA', 'MedXpertQA-Text'], 'Cybersecurity': ['CyberMetric', 'SecEval', 'CTI-Bench']}
KEY = {'MedMCQA': 'medmcqa', 'MedQA': 'medqa', 'MedXpertQA-Text': 'medxpertqa_text', 'CyberMetric': 'cybermetric', 'SecEval': 'seceval', 'CTI-Bench': 'cti_bench'}
def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def load_gold(root):
    return {b: json.loads((root / f'data/gold/{k}.json').read_text()) for b, k in KEY.items()}
def score(entry, gold, root):
    """Returns row dict. entry: manifest entry."""
    g = gold[entry['benchmark']]; ex = set(g['excluded_item_ids']); row = dict(entry=entry, status='MISSING', accuracy=None, n=None, issues=[])
    if not entry['prediction']: row['issues'].append('prediction file not archived'); return row
    p = root / entry['prediction']
    if not p.is_file(): row['issues'].append('archived prediction file absent'); return row
    if sha(p) != entry['prediction_sha256']: row['issues'].append('prediction sha256 differs from manifest'); row['status'] = 'FAILED_CHECKS'; return row
    raw = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ids = [str(r.get('item_id')) for r in raw]; ct = collections.Counter(ids); items = g['items']; frozen = set(items) - ex
    dup = [k for k, v in ct.items() if v > 1]; miss = sorted(frozen - set(ids)); unexp = sorted(set(ids) - set(items))
    kept = [r for r in raw if str(r.get('item_id')) not in ex]
    if dup: row['issues'].append(f'duplicate ids: {len(dup)}')
    if miss: row['issues'].append(f'missing ids: {len(miss)}')
    if unexp: row['issues'].append(f'unexpected ids: {len(unexp)}')
    if len(kept) != g['n']: row['issues'].append(f'n={len(kept)} != frozen n={g["n"]}')
    hit = 0; bad_letter = bad_gold = flag = 0
    for r in kept:
        it = items.get(str(r.get('item_id')))
        if not it: continue
        pred = r.get('pred', r.get('prediction')); allowed = set(it['allowed'] or g['default_allowed'])
        bad_letter += pred not in allowed; bad_gold += ('gold' in r and r['gold'] != it['gold'])
        ok = pred in it.get('gold_set', [it['gold']]); hit += ok; flag += ('correct' in r and bool(r['correct']) != ok)
    if bad_letter: row['issues'].append(f'non-standard answers: {bad_letter}')
    if bad_gold: row['issues'].append(f'gold mismatch vs frozen: {bad_gold}')
    if flag: row['issues'].append(f'stored correct flag disagrees: {flag}')
    row['n'] = len(kept); row['correct'] = hit
    if not row['issues']: row['status'] = 'VERIFIED'; row['accuracy'] = 100 * hit / g['n']
    else: row['status'] = 'FAILED_CHECKS'
    return row
def run(root=PKG):
    man = json.loads((root / 'manifests/table2_manifest.json').read_text()); gold = load_gold(root)
    for e in man['entries']: e['method'] = method_label(e['method'])
    rows = [score(e, gold, root) for e in man['entries']]
    cells = {}; per_run = collections.defaultdict(dict)
    for r in rows:
        e = r['entry']; per_run[(e['method'], e['student'], e['domain'], e['training_seed'])][e['benchmark']] = r
    out_runs, summary = [], []
    for (m, s, d, ts), bs in per_run.items():
        for b, r in bs.items(): out_runs.append(dict(method=m, student=s, domain=d, benchmark=b, training_seed=ts, acquisition_seed=r['entry']['acquisition_seed'], status=r['status'], n=r['n'], correct=r.get('correct'), accuracy=r['accuracy'], prediction=r['entry']['prediction'], issues='; '.join(r['issues'])))
        macro = {'Medicine': ('Medical3', ['MedMCQA', 'MedQA', 'MedXpertQA-Text']), 'Cybersecurity': ('CyberMacro', ['CyberMetric', 'SecEval'])}[d]
        ok = all(bs[x]['status'] == 'VERIFIED' for x in macro[1]) if all(x in bs for x in macro[1]) else False
        out_runs.append(dict(method=m, student=s, domain=d, benchmark=macro[0], training_seed=ts, acquisition_seed=bs[macro[1][0]]['entry']['acquisition_seed'], status='VERIFIED' if ok else 'UNVERIFIABLE', n='', correct='', accuracy=statistics.mean(bs[x]['accuracy'] for x in macro[1]) if ok else None, prediction='', issues=''))
    groups = collections.defaultdict(list)
    for r in out_runs: groups[(r['method'], r['student'], r['domain'], r['benchmark'])].append(r)
    for (m, s, d, b), rs in groups.items():
        exp = 1 if m == 'Pristine' else 3; v = [r['accuracy'] for r in rs if r['status'] == 'VERIFIED']
        summary.append(dict(method=m, student=s, domain=d, benchmark=b, n_seeds=len(v), expected_seeds=exp, completeness='COMPLETE' if len(v) == exp else ('PARTIAL' if v else 'MISSING'), mean=statistics.mean(v) if v else None, sample_sd=statistics.stdev(v) if len(v) > 1 else None, seed_structure=rs[0]['seed_structure'] if 'seed_structure' in rs[0] else ''))
    return man, rows, out_runs, summary
def fmt(c, pristine=False):
    if c is None or c['completeness'] != 'COMPLETE': return '--'
    return f"{c['mean']:.2f}" if pristine else f"${c['mean']:.2f} \\pm {c['sample_sd']:.2f}$"
def latex(summary, man):
    S = {(c['method'], c['student'], c['domain'], c['benchmark']): c for c in summary}; N = {b: man['panels'][b]['n'] for b in KEY}
    L = [r'% Generated by mapsteal_repro/aggregate.py from archived per-item predictions. Requires booktabs and graphicx.', r'\begin{table*}[t]', r'\centering',
         r'\caption{Main Results on Medicine and Cybersecurity. Accuracy (\%). Adapted models: arithmetic mean $\pm$ sample SD ($n=3$, ddof=1); pristine: one model, no SD. MapSteal uses one acquisition trajectory per setting with three training seeds; EvoKD-Adapted and QEDKS-Adapted use three independent acquisition/training trajectories. CyberMetric excludes \texttt{cybermetric:1729}.}',
         r'\label{tab:main-results-reproduced}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{llrrrrrr}', r'\toprule', r'Size & Method & \multicolumn{3}{c}{Medicine} & \multicolumn{3}{c}{Cybersecurity} \\', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
         r' & & MedMCQA & MedQA & MedXpertQA-Text & CyberMetric & SecEval & CTI-Bench \\', ' & & ' + ' & '.join(f'$n={N[b]}$' for b in KEY) + r' \\', r'\midrule']
    for i, sz in enumerate(('2B', '4B', '9B')):
        for m in METHODS:
            lab = m
            L.append(f'{sz} & {lab} & ' + ' & '.join(fmt(S.get((m, f'Qwen3.5-{sz}', d, b)), m == 'Pristine') for d in BENCH for b in BENCH[d]) + r' \\')
        if i < 2: L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table*}']; return '\n'.join(L) + '\n'
def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)); w = csv.DictWriter(open(path, 'w', newline=''), fieldnames=keys); w.writeheader(); w.writerows(rows)
def gold_check():
    if not (PKG / 'data/gold/medqa.json').is_file(): raise SystemExit('data/gold/ is missing. Build the gold labels from the benchmarks\' own releases:\n  python tools/prepare_benchmarks.py download external && python tools/prepare_benchmarks.py panels --source external --output panels && python tools/prepare_benchmarks.py gold --panels panels')
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--out', default=str(PKG / 'results')); ap.add_argument('--quiet', action='store_true', help='compact output (counts and paths only)'); a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gold_check()
    man, rows, runs, summary = run(); write_csv(out / 'table2_per_run.csv', runs); write_csv(out / 'table2_summary.csv', summary); (out / 'table2.tex').write_text(latex(summary, man))
    c = collections.Counter(s['completeness'] for s in summary); st = collections.Counter(r['status'] for r in rows)
    if a.quiet:
        print('prediction files:', dict(st)); print('table cells:', dict(c)); print('wrote', out); return 0
    from .display import render   # presentation only; falls back to plain text when `rich` is not installed
    render(PKG, rows, summary, out, [out / 'table2.tex', out / 'table2_summary.csv', out / 'table2_per_run.csv']); return 0
if __name__ == '__main__': sys.exit(main())
