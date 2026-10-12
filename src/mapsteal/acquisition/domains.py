"""Domain profiles of the formal MapSteal acquisition: Medicine and Cybersecurity (domain-neutral Teacher contract and realizer)."""
from types import SimpleNamespace
from pathlib import Path
from .. import paths

def get(domain):
    if domain == 'medicine':
        from .medicine import contract, realizer
        return SimpleNamespace(name='medicine', contract=contract, realizer=realizer, prompt_addition=paths.prompt_addition('medicine'), roots=paths.roots('medicine'), amendment1=False, seed_corpus=paths.seed_corpus('medicine'))
    if domain == 'cybersecurity':
        from .cybersecurity import contract, realizer
        return SimpleNamespace(name='cybersecurity', contract=contract, realizer=realizer, prompt_addition=paths.prompt_addition('cybersecurity'), roots=paths.roots('cybersecurity'), amendment1=True, seed_corpus=None)
    raise SystemExit(f'unknown domain {domain!r} (formal domains: medicine, cybersecurity)')
