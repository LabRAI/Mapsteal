#!/usr/bin/env python3
"""CPU-only validation of Mode A for the official four-method Table 1. Writes validation/VALIDATION_REPORT.md.
Expectations are derived from the manifest and the archived data, not hard-coded counts. Reference:
  validation/reference/independent_reference_four_methods.csv      run-level accuracies recomputed by a separate stdlib implementation (not by this package's aggregator)
Rounded paper values are never used as ground truth."""
import csv, json, re, shutil, statistics, sys, tempfile, collections
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PKG))
from mapsteal_repro import aggregate as A
A.gold_check()
OFFICIAL_METHODS = ['Pristine', 'MapSteal (Ours)', 'QEDKS-Adapted', 'EvoKD-Adapted']; OFFICIAL_SIZES = ['Qwen3.5-2B', 'Qwen3.5-4B', 'Qwen3.5-9B']
OFFICIAL_BENCH = {'Medicine': ['MedMCQA', 'MedQA', 'MedXpertQA-Text'], 'Cybersecurity': ['CyberMetric', 'SecEval', 'CTI-Bench']}; MAC = {'Medicine': 'Medical3', 'Cybersecurity': 'CyberMacro'}
man, rows, runs, summ = A.run(); L = []; ok_all = True
def chk(name, cond, detail=''):
    global ok_all; ok_all &= bool(cond); L.append(f"| {name} | {'PASS' if cond else 'FAIL'} | {detail} |")
ents = man['entries']
# 1 structure derived from the manifest
chk('manifest methods == official methods (no other baseline present)', sorted({e['method'] for e in ents}) == sorted(OFFICIAL_METHODS), str(sorted({e['method'] for e in ents})))
nb = sum(len(v) for v in OFFICIAL_BENCH.values()); exp_cells = len(OFFICIAL_SIZES) * len(OFFICIAL_METHODS) * nb
bench_cells = [s for s in summ if s['benchmark'] not in MAC.values()]; macro_cells = [s for s in summ if s['benchmark'] in MAC.values()]
chk(f'Table 1 benchmark cells = students x methods x benchmarks = {len(OFFICIAL_SIZES)}x{len(OFFICIAL_METHODS)}x{nb}', len(bench_cells) == exp_cells, f'{len(bench_cells)} cells found, {exp_cells} expected')
chk('every official Table 1 cell is COMPLETE', all(s['completeness'] == 'COMPLETE' for s in bench_cells + macro_cells), f'{sum(s["completeness"]=="COMPLETE" for s in bench_cells)}/{len(bench_cells)} benchmark cells; {sum(s["completeness"]=="COMPLETE" for s in macro_cells)}/{len(macro_cells)} per-domain macro summaries')
pred_ok = sum(r['status'] == 'VERIFIED' for r in rows)
chk('every manifest entry has an archived, verified prediction file (no missing, no failed)', pred_ok == len(ents) == len(rows), f'{pred_ok} verified of {len(ents)} entries')
# 2 seeds
seeds = collections.defaultdict(set)
for e in ents: seeds[(e['method'], e['student'], e['domain'], e['benchmark'])].add(e['training_seed'])
chk('adapted cells use exactly the three formal seeds {42,43,44}; pristine cells exactly one model', all((s == {None}) if k[0] == 'Pristine' else (s == {42, 43, 44}) for k, s in seeds.items()), f'{len(seeds)} cells')
chk('n_seeds in aggregation equals expected for every cell', all(c['n_seeds'] == c['expected_seeds'] == (1 if c['method'] == 'Pristine' else 3) for c in bench_cells + macro_cells))
acq = collections.defaultdict(set)
for e in ents:
    if e['method'] in ('QEDKS-Adapted', 'EvoKD-Adapted'): acq[(e['method'], e['student'], e['domain'])].add((e['acquisition_seed'], e['training_seed']))
chk('QEDKS-Adapted / EvoKD-Adapted: three independent acquisition trajectories (acquisition seed = training seed)', all(v == {(42, 42), (43, 43), (44, 44)} for v in acq.values()), f'{len(acq)} setting(s)')
prs = collections.defaultdict(set)
for e in ents:
    if e['method'] == 'Pristine': prs[(e['student'], e['domain'], e['benchmark'])].add(e['prediction'])
chk('Pristine cells: exactly one archived original evaluation per (student, benchmark); prediction sha256 equals the independent reference (checked below)', len(prs) == len(OFFICIAL_SIZES) * nb and all(len(v) == 1 for v in prs.values()), f'{len(prs)} cells')
# 3 panels
gold = A.load_gold(PKG); exp_n = {'MedMCQA': 500, 'MedQA': 1273, 'MedXpertQA-Text': 2450, 'CyberMetric': 2499, 'SecEval': 1521, 'CTI-Bench': 2482}
chk('frozen panel sizes (scored n)', all(gold[b]['n'] == n for b, n in exp_n.items()), str({b: gold[b]['n'] for b in exp_n}))
chk('CyberMetric exclusion unchanged: only cybermetric:1729 excluded (raw 2500 -> 2499)', gold['CyberMetric']['excluded_item_ids'] == ['cybermetric:1729'] and gold['CyberMetric']['raw_n'] == 2500 and all(gold[b]['excluded_item_ids'] == [] for b in gold if b != 'CyberMetric'))
chk('every verified run scores exactly the frozen n; ids == frozen panel (none dropped, duplicated or unexpected); gold and letters match', all(r['n'] == gold[r['entry']['benchmark']]['n'] for r in rows if r['status'] == 'VERIFIED'), f'{sum(1 for r in rows if r["status"]=="VERIFIED")} files')
# 4 independent reference (separate implementation)
ref = {(r['method'], r['student'], r['domain'], r['training_seed'], r['benchmark']): r for r in csv.DictReader(open(PKG / 'validation/reference/independent_reference_four_methods.csv'))}
mine = {(r['method'], r['student'], r['domain'], '' if r['training_seed'] is None else str(r['training_seed']), r['benchmark']): r for r in runs}
chk('same run-level keys in package output and independent reference', set(ref) == set(mine), f'{len(mine)} vs {len(ref)}')
diff = [k for k in ref if k in mine and (mine[k]['accuracy'] is None or abs(mine[k]['accuracy'] - float(ref[k]['accuracy'])) > 1e-9)]
chk('all run-level accuracies (files + macros) equal the independent reference (<=1e-9 pp)', not diff, f'{len(ref)} compared, {len(diff)} mismatches {diff[:2]}')
cn = [k for k in ref if ref[k]['n'] and k in mine and mine[k]['n'] != '' and (int(ref[k]['n']) != mine[k]['n'] or int(ref[k]['correct']) != mine[k]['correct'])]
chk('n and correct counts equal the independent reference', not cn, f'{len(cn)} mismatches')
sh = [e['prediction'] for e in ents if ref[(e['method'], e['student'], e['domain'], '' if e['training_seed'] is None else str(e['training_seed']), e['benchmark'])]['prediction_sha256'] != e['original_prediction_sha256']]
chk('public prediction files derive from the files the independent reference scored (original sha256 recorded)', not sh, f'{len(ents)} files, {len(sh)} differ')
# 5 LaTeX recomputed from the independent reference (statistics module), exact string comparison, no paper values
def cell(vals, pristine):
    return f'{vals[0]:.2f}' if pristine else f'${statistics.mean(vals):.2f} \\pm {statistics.stdev(vals):.2f}$'
tex = (PKG / 'results/table1.tex').read_text(); trows = {}
for m_ in re.finditer(r'^(\dB) & ([^&]+?) & (.*?) \\\\$', tex, re.M): trows[('Qwen3.5-' + m_.group(1), m_.group(2).strip())] = [c.strip() for c in m_.group(3).split('&')]
bad = []; n_cells = 0
for sz in OFFICIAL_SIZES:
    for me in OFFICIAL_METHODS:
        exp = []
        for d in ('Medicine', 'Cybersecurity'):
            for b in OFFICIAL_BENCH[d]:
                v = [float(r['accuracy']) for k, r in ref.items() if k[:3] == (me, sz, d) and k[4] == b]; exp.append(cell(v, me == 'Pristine'))
        got = trows.get((sz, me)); n_cells += len(exp)
        if got != exp: bad.append((sz, me))
chk('LaTeX table: every cell equals mean ± sample SD recomputed from the independent reference', not bad and n_cells == exp_cells, f'{n_cells} cells compared, rows differing: {bad}')
chk('LaTeX/CSV/manifest/results contain no baseline other than the four official methods', not re.search(r'EvoKD-Adapted-v|EvoKD-v|evokd_v', tex + (PKG / 'results/table1_summary.csv').read_text() + (PKG / 'results/table1_per_run.csv').read_text() + (PKG / 'manifests/table1_manifest.json').read_text()))
# 6 naming hygiene (public-facing files); documented exceptions: archive/, data/provenance/, tools/, this validator
pat = re.compile(r'EvoKD-Adapted-v[12]|EvoKD-v[12]|evokd_v[12]|EvoKD-Adapted v[12]|EvoKD v[12]')
offenders = []
for p in list(PKG.glob('*.md')) + list(PKG.glob('docs/*.md')) + list(PKG.glob('mapsteal_repro/*.py')) + list(PKG.glob('*.sh')) + list(PKG.glob('validation/*.md')) + list(PKG.glob('results/*')) + list(PKG.glob('requirements.txt')) + list(PKG.glob('*.yml')):
    if not p.is_file(): continue
    if pat.search(p.read_text(errors='ignore')): offenders.append(p.relative_to(PKG).as_posix())
chk('no versioned EvoKD identifiers in README, docs, CLI, results (exceptions: provenance/)', not offenders, str(offenders))
# 7 hygiene
man_txt = (PKG / 'manifests/table1_manifest.json').read_text(); code_txt = ''.join(p.read_text() for p in list(PKG.glob('mapsteal_repro/*.py')) + list(PKG.glob('*.sh')))
chk('manifest and runtime code (mapsteal_repro/, run.sh) contain no absolute machine paths', '/home/' not in man_txt and '/Users/' not in man_txt and '/home/' not in code_txt)
chk('no secrets in package text files', not any(re.search(r'sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}', p.read_text(errors='ignore')) for p in PKG.rglob('*') if p.is_file() and p.suffix in ('.py', '.md', '.sh', '.json', '.yml', '.txt', '.tex', '.csv') and '.git' not in p.parts and p.stat().st_size < 3_000_000))
# 8 tamper test
with tempfile.TemporaryDirectory() as t:
    t = Path(t); shutil.copytree(PKG / 'manifests', t / 'manifests'); shutil.copytree(PKG / 'data/gold', t / 'data/gold')
    e = next(e for e in ents if e['prediction'] and e['benchmark'] == 'MedQA'); (t / e['prediction']).parent.mkdir(parents=True, exist_ok=True)
    lines = (PKG / e['prediction']).read_text().splitlines(); d = json.loads(lines[0]); d['item_id'] = json.loads(lines[1])['item_id']; lines[0] = json.dumps(d); (t / e['prediction']).write_text('\n'.join(lines) + '\n')
    r = A.score(e, A.load_gold(t), t); chk('tamper test: altered prediction file is rejected (sha256 / duplicate-id check)', r['status'] != 'VERIFIED', r['status'] + ' ' + '; '.join(r['issues']))
rep = '# Validation report (Mode A, CPU only)\n\nGenerated by `validation/validate.py`; all expectations are derived from the manifest and archived data. References: `validation/reference/` (independent audit-code recomputation; previously verified snapshot restricted to retained methods).\n\n| check | result | detail |\n|---|---|---|\n' + '\n'.join(L) + f'\n\n**Overall: {"ALL CHECKS PASS" if ok_all else "FAILURES PRESENT"}**\n'
(PKG / 'validation/VALIDATION_REPORT.md').write_text(rep); print(rep); sys.exit(0 if ok_all else 1)
