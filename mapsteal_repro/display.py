"""Presentation-only terminal rendering of the already-computed Table 2 aggregation (no recomputation, no file output).
Uses `rich` when installed; otherwise a plain-text fallback. `rich` is optional: CPU-only aggregation never requires it."""
import re, shutil, sys
from pathlib import Path
from datetime import datetime
SHOW_METHOD = {}                                           # display label = table label
SIZES = ['Qwen3.5-2B', 'Qwen3.5-4B', 'Qwen3.5-9B']; METHODS = ['Pristine', 'MapSteal (Ours)', 'QEDKS-Adapted', 'EvoKD-Adapted']
BLOCKS = [('MEDICINE', 'Medicine', [('MedMCQA', 'MedMCQA'), ('MedQA', 'MedQA'), ('MedXpertQA-Text', 'MedXpert')]),
          ('CYBERSECURITY', 'Cybersecurity', [('CyberMetric', 'CyberMetric'), ('SecEval', 'SecEval'), ('CTI-Bench', 'CTI-Bench')])]
NOTE = ('Mean accuracy (%); ± = sample SD over 3 formal seeds (Pristine: one model, no SD). Bold/underlined = highest mean within that student and benchmark '
        '(descriptive only; no significance test is implied). MapSteal rows are tinted for readability.')
PLAIN_NOTE = NOTE.replace('Bold/underlined =', '* =').replace(' MapSteal rows are tinted for readability.', '')
def _have_rich():
    import os
    if os.environ.get('MAPSTEAL_PLAIN') == '1': return False   # force the plain-text renderer
    try:
        from rich.console import Console  # noqa: F401
        from rich.table import Table  # noqa: F401
        from rich.panel import Panel  # noqa: F401
        from rich.text import Text  # noqa: F401
        return True
    except Exception:
        return False
def _cell(c, pristine):
    if c is None or c['completeness'] != 'COMPLETE': return None, '--'
    return c['mean'], (f"{c['mean']:.2f}" if pristine else f"{c['mean']:.2f} ± {c['sample_sd']:.2f}")
def _counts(rows, summ, macros=('Medical3', 'CyberMacro')):
    bench = [s for s in summ if s['benchmark'] not in macros]; mac = [s for s in summ if s['benchmark'] in macros]
    ok = lambda xs: sum(s['completeness'] == 'COMPLETE' for s in xs)
    return dict(files_ok=sum(r['status'] == 'VERIFIED' for r in rows), files=len(rows), cells_ok=ok(bench), cells=len(bench), macro_ok=ok(mac), macros=len(mac))
def _validation_status(pkg, c):
    integrity = 'PASS' if c['files_ok'] == c['files'] and c['cells_ok'] == c['cells'] and c['macro_ok'] == c['macros'] else 'ATTENTION: incomplete or failed checks above'
    rep = Path(pkg) / 'validation/VALIDATION_REPORT.md'; last = 'no validation report found (run python3 validation/validate.py)'
    if rep.is_file():
        m = re.search(r'\*\*Overall: ([^*]+)\*\*', rep.read_text()); when = datetime.fromtimestamp(rep.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        if m: last = f'{m.group(1).strip()}  (report written {when})'
    return integrity, last
def render(pkg, rows, summ, out, files):
    S = {(s['method'], s['student'], s['domain'], s['benchmark']): s for s in summ}; c = _counts(rows, summ); integrity, last = _validation_status(pkg, c)
    summary = [('Prediction files', f"{c['files_ok']}/{c['files']} {'VERIFIED' if c['files_ok'] == c['files'] else 'VERIFIED (some failed or missing)'}"),
               ('Table 2 benchmark cells', f"{c['cells_ok']}/{c['cells']} {'COMPLETE' if c['cells_ok'] == c['cells'] else 'COMPLETE (some missing)'}"),
               ('Domain macro summaries', f"{c['macro_ok']}/{c['macros']} {'COMPLETE' if c['macro_ok'] == c['macros'] else 'COMPLETE (some missing)'}"),
               ('Aggregation integrity checks', integrity), ('Last full validation', last)] + [('Output files', (str(Path(files[0]).parent.relative_to(pkg)) if Path(files[0]).parent.is_relative_to(pkg) else str(Path(files[0]).parent)) + '/  (' + ', '.join(Path(f).name for f in files) + ')')]
    if _have_rich(): _render_rich(S, summary)
    else: _render_plain(S, summary)
def _tables(S):
    for title, dom, cols in BLOCKS:
        best = {}
        for sz in SIZES:
            for b, _ in cols:
                vals = [_cell(S.get((m, sz, dom, b)), m == 'Pristine')[0] for m in METHODS]; vals = [v for v in vals if v is not None]
                best[(sz, b)] = max(vals) if vals else None
        yield title, dom, cols, best
def _render_rich(S, summary):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    con = Console(width=min(shutil.get_terminal_size((100, 24)).columns, 110))
    for title, dom, cols, best in _tables(S):
        t = Table(title=f'[bold]{title}[/bold]  [dim](accuracy %)[/dim]', box=box.SIMPLE_HEAD, header_style='bold', title_justify='left', pad_edge=False)
        t.add_column('Student', no_wrap=True); t.add_column('Method', no_wrap=True)
        for _, short in cols: t.add_column(short, justify='right', no_wrap=True)
        for si, sz in enumerate(SIZES):
            if si: t.add_section()
            for mi, m in enumerate(METHODS):
                ms = m == 'MapSteal (Ours)'; style = 'cyan' if ms else None; cells = []
                for b, _ in cols:
                    v, txt = _cell(S.get((m, sz, dom, b)), m == 'Pristine')
                    isb = v is not None and best[(sz, b)] is not None and abs(v - best[(sz, b)]) < 1e-12
                    cells.append(Text(txt + (('*' if isb else ' ') if not con.is_terminal else ''), style=('bold underline ' + (style or '')).strip() if isb else (style or '')))
                t.add_row(sz if mi == 0 else '', Text(SHOW_METHOD.get(m, m), style=style or ('dim' if m == 'Pristine' else '')), *cells)
        con.print(t)
    con.print(Text(NOTE if con.is_terminal else PLAIN_NOTE, style='dim'), width=con.width)
    body = Text()
    for k, v in summary:
        body.append(f'{k:<30}', style='bold'); bad = ('ATTENTION' in v or 'some' in v or 'FAIL' in v); body.append(v + '\n', style='red' if bad else None)
    con.print(Panel(body, title='Verification Summary', title_align='left', border_style='dim', box=box.ROUNDED, expand=False))
def _render_plain(S, summary):
    w = shutil.get_terminal_size((100, 24)).columns
    for title, dom, cols, best in _tables(S):
        print(f'\n{title}  (accuracy %)'); hdr = ['Student', 'Method'] + [s for _, s in cols]; wid = [10, 10, 16, 16, 16]
        print('  '.join(h.ljust(wid[i]) if i < 2 else h.rjust(wid[i]) for i, h in enumerate(hdr))); print('-' * min(w, sum(wid) + 2 * 4))
        for si, sz in enumerate(SIZES):
            if si: print()
            for mi, m in enumerate(METHODS):
                cells = []
                for b, _ in cols:
                    v, txt = _cell(S.get((m, sz, dom, b)), m == 'Pristine'); mark = '*' if (v is not None and best[(sz, b)] is not None and abs(v - best[(sz, b)]) < 1e-12) else ' '
                    cells.append((txt + mark).rjust(16))
                print(f"{(sz if mi == 0 else '').ljust(10)}  {SHOW_METHOD.get(m, m).ljust(10)}  " + '  '.join(cells))
    print('\n' + PLAIN_NOTE)
    print('\nVerification Summary'); [print(f'  {k:<30}{v}') for k, v in summary]
