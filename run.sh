#!/usr/bin/env bash
# MapSteal Table 2 reproduction: unified entry point. See README.md.
#   bash run.sh --aggregate-only                 # Mode A: rebuild Table 2 from archived predictions (CPU, no Teacher)
#   bash run.sh --list                           # status matrix
#   bash run.sh --all --mode train --check       # verify code/corpora for Mode B (no execution)
#   bash run.sh --student 4B --domain medicine --method mapsteal --mode train            # prints plan
#   bash run.sh --student 4B --domain medicine --method mapsteal --mode train --execute  # runs after confirmation
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
cd "$HERE"
exec "$PYTHON_BIN" -m mapsteal_repro "$@"
