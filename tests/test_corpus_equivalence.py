#!/usr/bin/env python3
"""Corpus-equivalence test (Medicine; optionally Cybersecurity via --cyber-full TRAJECTORY): build the corpus from the archived 150-step trajectory prefix with src/mapsteal.corpus and compare with the formal exact-B500 Medicine corpus
(data/frozen_corpora/mapsteal_medicine_B500.jsonl, sha256 3f74c9af...): the new rows must be a byte-identical prefix of it. Optionally (--full TRAJECTORY)
rebuild the complete B500/B1000 corpora from the full archived trajectory and compare their sha256 with the archived corpus manifests. No network, no GPU."""
import gzip, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT / 'src'))
from mapsteal.corpus import build as cb
from mapsteal import paths
def main():
    traj = [json.loads(l) for l in gzip.open(HERE / 'fixtures/medicine_trajectory_prefix.jsonl.gz', 'rt')]
    seed = cb.jl(paths.seed_corpus('medicine')); rows, rep = cb.build(traj, seed)
    ref = open(ROOT / 'data/frozen_corpora/mapsteal_medicine_B500.jsonl').read().splitlines(keepends=True)
    mine = [json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows]
    ok = mine == ref[:len(mine)]; print(json.dumps({'new_rows': len(mine), 'prefix_of_formal_B500_corpus_byte_identical': ok, **rep}))
    if '--full' in sys.argv:
        full = Path(sys.argv[sys.argv.index('--full') + 1]); trows = [json.loads(l) for l in open(full)]
        man = json.load(open(ROOT / 'manifests/mapsteal_medicine_corpus_manifest.json'))['corpora']
        for B, key in ((500, 'MEDB500T'), (1000, 'MEDB1000T')):
            r2, rp = cb.build(trows, seed, B - len(seed)); h = hashlib.sha256(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in r2).encode()).hexdigest()
            print(json.dumps({'budget_total': B, 'rows': len(r2), 'sha256_equals_archived': h == man[key]['sha256'], 'qc_drops': rp['qc_drops']})); ok &= (h == man[key]['sha256'])
    if '--cyber-full' in sys.argv:   # Cybersecurity: first 500 executed rows, no seed rows -> the archived exact-B500 Cybersecurity corpus (sha256 768114d0...)
        crows = [json.loads(l) for l in open(sys.argv[sys.argv.index('--cyber-full') + 1])]; r3, rp3 = cb.build(crows, [], 500)
        ref3 = open(ROOT / 'data/frozen_corpora/mapsteal_cybersecurity_B500.jsonl').read(); mine3 = ''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in r3)
        print(json.dumps({'cybersecurity_B500_rows': len(r3), 'byte_identical_to_archived_corpus': mine3 == ref3, 'sha256': hashlib.sha256(mine3.encode()).hexdigest(), 'qc_drops': rp3['qc_drops']})); ok &= (mine3 == ref3)
    return 0 if ok else 1
if __name__ == '__main__': sys.exit(main())
