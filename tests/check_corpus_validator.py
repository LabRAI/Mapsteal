"""Corpus validator on the archived trajectory fixture (needs the restricted Teacher-output fixtures)."""
import gzip, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, 'src')
from mapsteal.corpus import build as cb, validate as cv
from mapsteal import paths
t = Path(tempfile.mkdtemp()); (t / 'trajectory.jsonl').write_bytes(gzip.open('tests/fixtures/medicine_trajectory_prefix.jsonl.gz').read())
seed = paths.seed_corpus('medicine'); cb.build_from_run(t, t / 'corpus.jsonl', seed); r = cv.validate_run(t, t / 'corpus.jsonl', seed)
print(json.dumps({'all_checks_pass': r['all_checks_pass'], 'checks_failed': [k for k, v in r['checks'].items() if v is False]})); sys.exit(0 if r['all_checks_pass'] else 1)
