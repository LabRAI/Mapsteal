#!/usr/bin/env python3
"""Restore restricted inputs (files not redistributed in the public repository) from a local bundle directory holding the original files, verifying every file against the sha256 recorded in manifests/restricted_inputs.json.
usage: restore_restricted.py BUNDLE_DIR [--categories R1_benchmark_labels,R2_teacher_outputs,...] [--dry-run]
BUNDLE_DIR must contain the files at their repository-relative paths (e.g. BUNDLE_DIR/data/gold/medqa.json). Nothing is overwritten unless the sha256 matches the manifest; mismatching files are reported and skipped."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument('bundle'); ap.add_argument('--categories'); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
items = json.loads((ROOT / 'manifests/restricted_inputs.json').read_text())['items']; cats = set(a.categories.split(',')) if a.categories else None
ok = bad = absent = 0
for i in items:
    if cats and i['category'] not in cats: continue
    src = Path(a.bundle) / i['path']
    if not src.is_file(): absent += 1; continue
    if hashlib.sha256(src.read_bytes()).hexdigest() != i['sha256']: bad += 1; print('SHA MISMATCH (skipped):', i['path']); continue
    if not a.dry_run: dst = ROOT / i['path']; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    ok += 1
print(f'{"verified" if a.dry_run else "restored"}: {ok}, sha mismatch: {bad}, not in bundle: {absent}'); sys.exit(1 if bad else 0)
