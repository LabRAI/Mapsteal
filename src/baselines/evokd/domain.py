"""Domain switch for EvoKD v2: EVOKD_V2_DOMAIN=med (default, identical to the passed pilot) | cyber (evokd_cyber package: the 11-line wording conversion of the same code)."""
import os, sys, importlib
from pathlib import Path
R = Path(__file__).resolve().parents[3]   # repository root
DOMAIN = os.environ.get('EVOKD_V2_DOMAIN', 'med')
PKG = {'med': 'baselines.evokd.paper_faithful', 'cyber': 'baselines.evokd.cyber'}[DOMAIN]
# (legacy PKG_DIR line removed)
D0 = R / 'data/evokd' / {'med': 'd0_root4_medicine.jsonl', 'cyber': 'd0_root4_cyber.jsonl'}[DOMAIN]
# (legacy line removed: packages are imported as baselines.evokd.*; no sys.path manipulation)
def mod(name): return importlib.import_module(f'{PKG}.{name}')

# ---- backbone size (cross-scale extension). Default 4B == the committed formal runs (behaviour unchanged).
SIZE = os.environ.get('EVOKD_V2_SIZE', '4B'); assert SIZE in ('2B', '4B', '9B')
HOME = Path.home()
MODELS = {  # frozen revisions (identical to medical_multibase_b500_v1 / f2_scaling_chem_cyber_v1); 2B/9B load from sha-verified local snapshots on HPC
    '2B': {'id': 'Qwen/Qwen3.5-2B', 'rev': '15852e8c16360a2fea060d615a32b45270f8a8fc', 'local': HOME / 'models/Qwen3.5-2B-15852e8c16360a2fea060d615a32b45270f8a8fc'},
    '4B': {'id': 'Qwen/Qwen3.5-4B', 'rev': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'local': None},
    '9B': {'id': 'Qwen/Qwen3.5-9B', 'rev': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'local': HOME / 'models/Qwen3.5-9B-c202236235762e1c871ad0ccb60c8ee5ba337b9a'}}
def base_path(size=None): m = MODELS[size or SIZE]; return str(m['local']) if m['local'] else m['id']
def base_rev(size=None): return MODELS[size or SIZE]['rev']
# execution-only settings (memory); objective, effective batch (8) and training semantics identical. 2B/9B follow the v1 low-seed precedent (9B: micro-batch 2 + gradient checkpointing).
EXEC = {'4B': {'micro_batch': 4, 'gradient_checkpointing': False}, '2B': {'micro_batch': 2, 'gradient_checkpointing': False}, '9B': {'micro_batch': 2, 'gradient_checkpointing': True}}
