#!/usr/bin/env python3
"""Compare predictions of the new evaluator (subset or full) with ARCHIVED formal predictions of the same adapter: item ids, gold, predicted letter must be identical for every item both contain.
usage: compare_eval.py NEW_EVAL_DIR ARCHIVE_PRED_DIR   (ARCHIVE_PRED_DIR holds medmcqa.jsonl / medqa.jsonl / medxpertqa_text.jsonl, e.g. data/predictions/mapsteal/2B/medicine/s42)"""
import json, sys
from pathlib import Path
new, arch = Path(sys.argv[1]), Path(sys.argv[2]); res = {}; ok = True
for b in ('medmcqa', 'medqa', 'medxpertqa_text'):
    if not (new / f'{b}.jsonl').exists(): continue
    n = [json.loads(l) for l in open(new / f'{b}.jsonl')]; a = {r['item_id']: r for r in map(json.loads, open(arch / f'{b}.jsonl'))}
    miss = [r['item_id'] for r in n if r['item_id'] not in a]; diff = [r['item_id'] for r in n if r['item_id'] in a and (r['gold'] != a[r['item_id']]['gold'] or r['prediction'] != a[r['item_id']].get('pred', a[r['item_id']].get('prediction')))]
    res[b] = {'compared': len(n) - len(miss), 'missing_in_archive': len(miss), 'prediction_or_gold_differs': len(diff), 'identical': not miss and not diff}; ok &= res[b]['identical']
print(json.dumps(res, indent=1)); sys.exit(0 if ok else 1)
