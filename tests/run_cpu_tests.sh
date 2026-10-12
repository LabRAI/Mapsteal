#!/usr/bin/env bash
# CPU-only tests of src/mapsteal and src/baselines (no GPU, no network, no Teacher cost).
# Tests that need data files which are not distributed (Teacher-generated corpora and fixtures, EvoKD initial state) print SKIPPED and are counted separately from passes.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."; PY="${PYTHON:-python3}"; export PYTHONPATH="$PWD/src"
SKIPPED=0; RAN=0
skip() { echo "SKIPPED (input not available: $1)"; SKIPPED=$((SKIPPED+1)); }
have() { [ -e "$1" ]; }
FIX=tests/fixtures/medicine_trajectory_prefix.jsonl.gz; ROOTS=src/mapsteal/data/medicine/roots35.jsonl; D0=data/evokd/d0_root4_medicine.jsonl; CORP=data/frozen_corpora/mapsteal_medicine_B500.jsonl
echo "[1/9] every module compiles"; "$PY" -m compileall -q src/mapsteal src/baselines; RAN=$((RAN+1))
echo "[2/9] CLI help and dry plans"; "$PY" -m mapsteal --help >/dev/null; RAN=$((RAN+1))
if have $ROOTS; then "$PY" -m mapsteal acquire --domain medicine --budget 10 --output /tmp/mapsteal_dry | tail -1; "$PY" -m mapsteal acquire --domain cybersecurity --budget 10 --output /tmp/mapsteal_dry | tail -1; else skip "roots (acquire plan)"; fi
echo "[3/9] replay equivalence (archived Teacher responses, both domains)"; if have $FIX && have $ROOTS; then "$PY" tests/test_replay_equivalence.py | grep '^{'; RAN=$((RAN+1)); else skip "tests/fixtures, roots"; fi
echo "[4/9] corpus equivalence (archived B500 corpus prefix)"; if have $FIX && have $CORP; then "$PY" tests/test_corpus_equivalence.py | head -1; RAN=$((RAN+1)); else skip "tests/fixtures, data/frozen_corpora"; fi
echo "[5/9] corpus validator on the archived trajectory fixture"; if have $FIX && have src/mapsteal/data/medicine/seed40.jsonl; then "$PY" tests/check_corpus_validator.py; RAN=$((RAN+1)); else skip "tests/fixtures, seed40"; fi
echo "[6/9] cost guard"; if have $FIX && have $ROOTS; then "$PY" tests/test_cost_guard.py; RAN=$((RAN+1)); else skip "tests/fixtures, roots"; fi
echo "[7/9] baselines: CLI help, QEDKS acquire plan (no --execute), EvoKD dry run"; "$PY" -m baselines --help >/dev/null; "$PY" -m baselines qedks acquire --domain medicine --student 4B --seed 42 --output /tmp/qedks_dry | tail -1; RAN=$((RAN+1))
if have $D0; then "$PY" -m baselines evokd run --domain medicine --out /tmp/evokd_dry --dry-run >/dev/null && echo "evokd dry run ok"; else skip "data/evokd (EvoKD dry run)"; fi
echo "[8/9] baseline frozen-input equivalence (source map, QEDKS corpora + verify, EvoKD dry run vs archived records)"; "$PY" tests/test_baseline_equivalence.py; RAN=$((RAN+1))
echo "[9/9] formal settings vs archived run records and code constants"; "$PY" tests/test_formal_settings.py; RAN=$((RAN+1))
echo "CPU TESTS: no failure; $RAN groups executed, $SKIPPED skipped because their input data are not distributed"
