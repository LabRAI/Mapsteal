"""Frozen Student snapshots (identical to the other studies); 2B/9B load from sha-verified local snapshots on HPC."""
from pathlib import Path
HOME = Path.home()
MODELS = {
    "2B": {"id": "Qwen/Qwen3.5-2B", "rev": "15852e8c16360a2fea060d615a32b45270f8a8fc", "local": HOME / "models/Qwen3.5-2B-15852e8c16360a2fea060d615a32b45270f8a8fc"},
    "4B": {"id": "Qwen/Qwen3.5-4B", "rev": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "local": None},
    "9B": {"id": "Qwen/Qwen3.5-9B", "rev": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "local": HOME / "models/Qwen3.5-9B-c202236235762e1c871ad0ccb60c8ee5ba337b9a"},
}
def base_path(size):
    m = MODELS[size]
    return str(m["local"]) if m["local"] else m["id"]
