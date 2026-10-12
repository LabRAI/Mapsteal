"""Locations of the packaged frozen data (roots, prompt addition, Teacher contract, Seed40 cold-start corpus of Medicine)."""
from pathlib import Path
DATA = Path(__file__).resolve().parent / 'data'
ROOTS = {'medicine': 'roots35.jsonl', 'cybersecurity': 'roots40.jsonl'}
def roots(domain):
    return DATA / domain / ROOTS[domain]
def prompt_addition(domain): return (DATA / domain / 'prompt_addition_rfc.txt').read_text()
def seed_corpus(domain):
    """Path of the Medicine cold-start rows (not distributed; may be absent) or None for Cybersecurity."""
    return DATA / domain / 'seed40.jsonl' if domain == 'medicine' else None
def resolve_seed(arg, domain='medicine'):
    """--seed-corpus value: 'auto', 'medicine', 'none' or a JSONL file with instruction/output rows."""
    if arg in (None, 'none') or domain != 'medicine' and arg in ('medicine', 'auto'): return None
    if arg == 'auto':
        if seed_corpus('medicine').is_file(): return seed_corpus('medicine')
        print('note: the 40 Medicine cold-start rows are not distributed; the corpus will contain the acquisition rows only (pass --seed-corpus FILE to add your own)'); return None
    p = seed_corpus('medicine') if arg == 'medicine' else Path(arg)
    if not p.is_file(): raise SystemExit(f'cold-start rows not found: {p}. Use --seed-corpus none (acquisition rows only) or a JSONL file of cold-start rows (keys input/instruction/output); the 40 formal Medicine rows are not distributed.')
    return p
