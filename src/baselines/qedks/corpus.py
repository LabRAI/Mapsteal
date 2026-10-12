"""corpus_raw.jsonl (all QEDKS QA rows: main + correction) -> training corpus (keys input/instruction/output). QC: drop rows with an empty Teacher answer ONLY
(no de-duplication: correction rows intentionally share the original question). Logic identical to the formal scripts/export_corpus.py and to the export inside verify_acquisition.py."""
import hashlib, json
from pathlib import Path

def export_rows(raw_rows):
    return [{'input': '', 'instruction': r['instruction'], 'output': r['output']} for r in raw_rows if r['output'].strip()]

def dump(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows)

def export_corpus(run_dir, out):
    run, out = Path(run_dir), Path(out)
    raw = [json.loads(l) for l in open(run / 'corpus_raw.jsonl') if l.strip()]; keep = export_rows(raw)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(dump(keep))
    info = {'raw_rows': len(raw), 'training_rows_after_qc': len(keep), 'dropped_empty_answer': len(raw) - len(keep), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix('.manifest.json').write_text(json.dumps(info, indent=1)); return info
