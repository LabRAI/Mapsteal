#!/usr/bin/env python3
"""Static check: every absolute import in src/**/*.py, mapsteal_repro/ and validation/ resolves to the stdlib, a declared third-party package, or a module shipped under src/ (mapsteal, baselines). Also flags absolute machine paths and references to a non-existent impl/ directory."""
import ast, re, sys
from pathlib import Path
P = Path(__file__).resolve().parents[1]
THIRD = {'torch', 'transformers', 'peft', 'numpy', 'scipy', 'pandas', 'yaml', 'openai', 'safetensors', 'tqdm', 'accelerate', 'datasets', 'sklearn', 'sentence_transformers', 'Levenshtein', 'rapidfuzz', 'rich', 'pytest', 'tiktoken', 'httpx', 'requests', 'matplotlib', 'numba', 'bitsandbytes', 'huggingface_hub', 'sentencepiece', 'nltk', 'networkx', 'faiss'}
std = set(sys.stdlib_module_names); local = {p.name for p in (P / 'src').iterdir() if p.is_dir()} | {'mapsteal_repro'}
unres = {}; absp = {}; implref = {}
for root in ('src', 'mapsteal_repro', 'validation', 'tests'):
    for p in sorted((P / root).rglob('*.py')):
        t = p.read_text(errors='ignore'); rel = str(p.relative_to(P))
        for m in re.findall(r"['\"](/home/[^'\"]+|/Users/[^'\"]+)['\"]", t): absp.setdefault(rel, set()).add(m)
        if root in ('src', 'mapsteal_repro') and re.search(r"['\"][^'\"\n]*\bimpl/[^'\"\n]*['\"]", t): implref.setdefault(rel, set()).add('impl/')
        try: tree = ast.parse(t)
        except SyntaxError: unres.setdefault('SYNTAX', set()).add(rel); continue
        for n in ast.walk(tree):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else ([n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else [])
            for m in names:
                top = m.split('.')[0]
                if top in std or top in THIRD or top in local or (P / root / f'{top}.py').is_file() or (P / root / top).is_dir(): continue
                unres.setdefault(top, set()).add(rel)
print('UNRESOLVED:'); [print(' ', k, sorted(v)[:3]) for k, v in sorted(unres.items())]
print('ABSOLUTE PATHS:'); [print(' ', k, sorted(v)[:2]) for k, v in sorted(absp.items())]
print('IMPL REFERENCES:'); [print(' ', k, sorted(v)[:2]) for k, v in sorted(implref.items())]
sys.exit(1 if (unres or absp or implref) else 0)
