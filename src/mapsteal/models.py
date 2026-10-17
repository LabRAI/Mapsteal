"""Frozen Student models of Table 2 (exact revisions). 4B is read from the Hugging Face cache by id + revision (offline); 2B and 9B from local snapshots of exactly that commit
at $MAPSTEAL_MODELS_DIR/Qwen3.5-<size>-<revision> (default ~/models), as in the formal experiments."""
import os
from pathlib import Path
MODELS_DIR = Path(os.environ.get('MAPSTEAL_MODELS_DIR', Path.home() / 'models'))
MODELS = {
    '2B': {'id': 'Qwen/Qwen3.5-2B', 'rev': '15852e8c16360a2fea060d615a32b45270f8a8fc', 'local': MODELS_DIR / 'Qwen3.5-2B-15852e8c16360a2fea060d615a32b45270f8a8fc'},
    '4B': {'id': 'Qwen/Qwen3.5-4B', 'rev': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'local': None},
    '9B': {'id': 'Qwen/Qwen3.5-9B', 'rev': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'local': MODELS_DIR / 'Qwen3.5-9B-c202236235762e1c871ad0ccb60c8ee5ba337b9a'}}
BY_ID = {m['id']: k for k, m in MODELS.items()}
def resolve(student):
    """student: '2B' | '4B' | '9B' | 'Qwen/Qwen3.5-<size>'. Returns dict(size, id, rev, base_path)."""
    size = student if student in MODELS else BY_ID.get(student)
    if size is None: raise SystemExit(f'unknown Student {student!r}: the formal Table 2 Students are {sorted(BY_ID)} (or 2B/4B/9B)')
    m = MODELS[size]
    return {'size': size, 'id': m['id'], 'rev': m['rev'], 'base_path': str(m['local']) if m['local'] else m['id']}
