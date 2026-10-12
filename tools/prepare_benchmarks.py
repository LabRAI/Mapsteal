#!/usr/bin/env python3
"""Build the benchmark panels and gold labels from the benchmarks' own releases.

  python tools/prepare_benchmarks.py download SOURCE_DIR          fetch the released files (MedQA: Google Drive archive of the MedQA project; others: Hugging Face / GitHub)
  python tools/prepare_benchmarks.py panels   --source SOURCE_DIR --output PANELS_DIR
  python tools/prepare_benchmarks.py gold     --panels PANELS_DIR [--output data/gold]

`panels` selects the items named in data/predictions (sample ids and order), renders them in the evaluator's format and checks the sha256 of every panel against manifests/panel_hashes.json.
`gold` extracts the gold labels from the panels for the aggregation (`bash run.sh --aggregate-only`). Gold labels are not distributed with this repository; check each benchmark's terms before use.
Needs pandas + pyarrow (MedMCQA parquet) and, for CyberMetric/SecEval/CTI-Bench, `transformers` with the Qwen3.5-4B tokenizer in the local Hugging Face cache (the panels store the prompt length)."""
import argparse, csv, hashlib, json, re, sys, urllib.request, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
HF = 'https://huggingface.co/datasets/{}/resolve/{}/{}'
SOURCES = {   # local name -> (url, description)
    'medxpertqa_text_test.jsonl': (HF.format('TsinghuaC3I/MedXpertQA', '7e7c465a68eb2b866926bfa59c8c9d17a8daba65', 'Text/test.jsonl'), 'MedXpertQA, Text/test'),
    'medmcqa_validation.parquet': (HF.format('openlifescienceai/medmcqa', '91c6572c454088bf71b679ad90aa8dffcd0d5868', 'data/validation-00000-of-00001.parquet'), 'MedMCQA, validation split (Hugging Face release)'),
    'seceval_questions.json': (HF.format('XuanwuAI/SecEval', '205dab7b0888a06f4b53ca7d9c7093e1326683e1', 'questions.json'), 'SecEval'),
    'cti-mcq.tsv': (HF.format('AI4Sec/cti-bench', '9237e1636ee3e168fbe5ebdcc1c571de0525e568', 'cti-mcq.tsv'), 'CTI-Bench, MCQ'),
    'CyberMetric-10000-v1.json': ('https://raw.githubusercontent.com/cybermetric/CyberMetric/205262cdf5022ba890e792efd176fb19d42913fa/CyberMetric-10000-v1.json', 'CyberMetric-10000'),
}
MEDQA_DRIVE = '1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw'   # data archive linked from https://github.com/jind11/MedQA (member: data_clean/questions/US/test.jsonl)
BENCH = {'MedMCQA': ('medicine', 'medmcqa'), 'MedQA': ('medicine', 'medqa'), 'MedXpertQA-Text': ('medicine', 'medxpertqa_text'), 'CyberMetric': ('cybersecurity', 'cybermetric'), 'SecEval': ('cybersecurity', 'seceval'), 'CTI-Bench': ('cybersecurity', 'cti_bench')}
PANELS = json.loads((ROOT / 'manifests/panel_hashes.json').read_text())['panels']
sha = lambda b: hashlib.sha256(b).hexdigest()
EXPECTED = json.loads((ROOT / 'manifests/benchmark_sources.json').read_text())['sources']
def verify_source(path):
    want = EXPECTED[path.name]['sha256']
    if sha(path.read_bytes()) != want: raise SystemExit(f'{path.name}: sha256 differs from manifests/benchmark_sources.json (the upstream file changed or the download is incomplete)')
def jdump(rows): return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)

def fetch(url, dst):
    req = urllib.request.Request(url, headers={'User-Agent': 'mapsteal-prepare-benchmarks'})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, 'wb') as f: f.write(r.read())

def download_medqa(src):
    import http.cookiejar
    jar = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    page = op.open(f'https://drive.google.com/uc?export=download&id={MEDQA_DRIVE}', timeout=120).read().decode('utf-8', 'ignore')
    form = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', page))
    if 'uuid' not in form: raise RuntimeError('Google Drive did not offer the download form')
    q = '&'.join(f'{k}={v}' for k, v in form.items()); zp = src / 'medqa_data.zip'
    with op.open(f'https://drive.usercontent.google.com/download?{q}', timeout=900) as r, open(zp, 'wb') as f: f.write(r.read())
    with zipfile.ZipFile(zp) as z: (src / 'medqa_US_test.jsonl').write_bytes(z.read('data_clean/questions/US/test.jsonl'))
    zp.unlink(); verify_source(src / 'medqa_US_test.jsonl')

def cmd_download(a):
    src = Path(a.source_dir); src.mkdir(parents=True, exist_ok=True)
    for name, (url, desc) in SOURCES.items():
        if (src / name).exists(): print('present', name); continue
        print('downloading', desc, '<-', url); fetch(url, src / name); verify_source(src / name)
    if not (src / 'medqa_US_test.jsonl').exists():
        try: print('downloading MedQA (US test) from the MedQA project archive'); download_medqa(src)
        except Exception as e: print(f'MedQA download failed ({e}). Download the archive manually from the link in https://github.com/jind11/MedQA and extract data_clean/questions/US/test.jsonl to {src}/medqa_US_test.jsonl')
    print('done; next: prepare_benchmarks.py panels --source', src)

def pred_ids(bench):
    dom, key = BENCH[bench]; p = ROOT / f'data/predictions/pristine/4B/{dom}/pristine/{key}.jsonl'
    if not p.is_file(): raise SystemExit(f'{p} not found')
    return [json.loads(l)['item_id'] for l in open(p) if l.strip()]

def tokenizer():
    from transformers import AutoTokenizer
    from mapsteal.models import MODELS
    return AutoTokenizer.from_pretrained(MODELS['4B']['id'], revision=MODELS['4B']['rev'], local_files_only=True)

def build(bench, src):
    ids = pred_ids(bench)
    if bench == 'MedQA': return (src / 'medqa_US_test.jsonl').read_bytes()
    if bench == 'MedXpertQA-Text':
        by = {'medxpert:' + r['id']: r for r in map(json.loads, open(src / 'medxpertqa_text_test.jsonl'))}
        return jdump([{'gold': by[i]['label'], 'item_id': i, 'meta': {k: by[i][k] for k in ('body_system', 'medical_task', 'question_type')}, 'options': by[i]['options'], 'question': by[i]['question'].split('\nAnswer Choices:')[0]} for i in ids]).encode()
    if bench == 'MedMCQA':
        import pandas as pd
        d = pd.read_parquet(src / 'medmcqa_validation.parquet').set_index('id'); out = []
        for i in ids:
            r = d.loc[i]; opts = [str(r[c]).strip().replace('\n', ' ') for c in ('opa', 'opb', 'opc', 'opd')]
            prompt = ('You are a medical multiple-choice answering system.\nRead the question and options carefully.\nAnswer with ONLY one letter: A, B, C, D, or E.\n\nQuestion:\n' + str(r['question']).strip() + '\n\nOptions:\n' + '\n'.join(f'{k}. {o}' for k, o in zip('ABCD', opts)) + '\n\nAnswer:')
            out.append({'gold': 'ABCD'[int(r['cop'])], 'item_id': i, 'prompt': prompt})
        return jdump(out).encode()
    from mapsteal.evaluation.cybersecurity import prompt as mk_prompt
    tok = tokenizer(); out = []
    if bench == 'CyberMetric':
        qs = json.load(open(src / 'CyberMetric-10000-v1.json'))['questions']
        for i in ids:
            q = qs[int(i.split(':')[1])]; r = {'gold': q['solution'], 'item_id': i, 'meta': {}, 'options': {k: str(v).strip() for k, v in q['answers'].items()}, 'question': q['question'].strip(), 'subject': 'cybermetric'}; out.append(r)
    elif bench == 'SecEval':
        by = {'seceval:' + x['id']: x for x in json.load(open(src / 'seceval_questions.json'))}
        for i in ids:
            q = by[i]; r = {'gold': (q['answer'][:1] or 'A'), 'item_id': i, 'meta': {}, 'options': {c.split(':', 1)[0].strip(): c.split(':', 1)[1].strip() for c in q['choices']}, 'question': q['question'].strip(), 'subject': q['source']}; out.append(r)
    else:
        rows = list(csv.DictReader(open(src / 'cti-mcq.tsv', newline=''), delimiter='\t'))
        for i in ids:
            q = rows[int(i.split(':')[1])]; r = {'gold': q['GT'].strip().upper(), 'item_id': i, 'meta': {}, 'options': {k: (q['Option ' + k].strip() or 'nan') for k in 'ABCD'}, 'question': q['Question'].strip(), 'subject': 'cti_mcq'}; out.append(r)
    for r in out: r['prompt_tokens'] = len(tok(mk_prompt(r)[0], add_special_tokens=False)['input_ids'])
    return jdump(out).encode()

def cmd_panels(a):
    src, out = Path(a.source), Path(a.output); out.mkdir(parents=True, exist_ok=True); bad = 0
    for bench, spec in PANELS.items():
        try: data = build(bench, src)
        except FileNotFoundError as e: print(f'{bench:16} MISSING SOURCE {e.filename}'); bad += 1; continue
        ok = sha(data) == spec['panel_sha256']; bad += (not ok)
        (out / spec['file_name']).write_bytes(data); print(f'{bench:16} {spec["file_name"]:34} sha256 {"matches the frozen panel" if ok else "DIFFERS from the frozen panel (" + sha(data)[:12] + " vs " + spec["panel_sha256"][:12] + ")"}')
    sys.exit(1 if bad else 0)

def cmd_gold(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True); bad = 0
    for bench, spec in PANELS.items():
        p = Path(a.panels) / spec['file_name']
        if not p.is_file(): print(f'{bench:16} missing panel {p}'); bad += 1; continue
        raw = p.read_bytes()
        if sha(raw) != spec['panel_sha256']: print(f'{bench:16} panel sha256 differs from the frozen panel; refusing'); bad += 1; continue
        rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]; key = BENCH[bench][1]; items = {}
        for i, r in enumerate(rows):
            if bench == 'MedQA': iid, gold, allowed = str(i), str(r['answer_idx']).upper(), sorted(str(k).upper() for k in r['options'])
            elif bench == 'MedMCQA': iid, gold, allowed = r['item_id'], r['gold'], None
            else: iid, gold, allowed = r['item_id'], r['gold'], sorted(r['options'])
            items[iid] = {'allowed': allowed, 'gold': gold}
        exc = spec['excluded_item_ids']
        default = {'MedMCQA': 'ABCD', 'MedQA': 'ABCDE', 'MedXpertQA-Text': 'ABCDEFGHIJ'}.get(bench, 'ABCD')
        g = {'benchmark': bench, 'default_allowed': default, 'excluded_item_ids': exc, 'items': items, 'n': len(rows) - len(exc), 'panel_sha256': spec['panel_sha256'], 'raw_n': len(rows)}
        (out / f'{key}.json').write_text(json.dumps(g, indent=0, sort_keys=True)); print(f'{bench:16} {len(rows)} items, {len(exc)} excluded -> {out / (key + ".json")}')
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__); s = ap.add_subparsers(dest='cmd', required=True)
    d = s.add_parser('download'); d.add_argument('source_dir'); d.set_defaults(f=cmd_download)
    p = s.add_parser('panels'); p.add_argument('--source', required=True); p.add_argument('--output', required=True); p.set_defaults(f=cmd_panels)
    g = s.add_parser('gold'); g.add_argument('--panels', required=True); g.add_argument('--output', default=str(ROOT / 'data/gold')); g.set_defaults(f=cmd_gold)
    a = ap.parse_args(); a.f(a)
