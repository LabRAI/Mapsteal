#!/usr/bin/env python3
"""Maintainer tool: writes the public prediction files from the archived per-item prediction files. Only the sample identifier and the predicted option letter are kept, in the original row order;
benchmark labels, correctness flags, scores and text fields are dropped. Records original and exported sha256 in manifests/table2_manifest.json.
usage: export_predictions.py ORIGINAL_DIR   (ORIGINAL_DIR contains data/predictions/...)"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; src = Path(sys.argv[1]); man = json.loads((ROOT / 'manifests/table2_manifest.json').read_text()); sha = lambda b: hashlib.sha256(b).hexdigest()
for e in man['entries']:
    p = src / e['prediction']; raw = p.read_bytes(); orig = sha(raw); rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    if 'original_prediction_sha256' in e: assert e['original_prediction_sha256'] == orig, p
    else: assert e['prediction_sha256'] == orig, p
    out = ''.join(json.dumps({'item_id': str(r['item_id']), 'pred': r.get('pred', r.get('prediction'))}, sort_keys=True) + '\n' for r in rows).encode()
    dst = ROOT / e['prediction']; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(out)
    e['original_prediction_sha256'] = orig; e['prediction_sha256'] = sha(out)
(ROOT / 'manifests/table2_manifest.json').write_text(json.dumps(man, indent=1)); print(len(man['entries']), 'prediction files exported')
