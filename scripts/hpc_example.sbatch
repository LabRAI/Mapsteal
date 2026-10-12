#!/bin/bash
# Example SLURM job for the GPU steps of MapSteal (adapt partition / constraint / environment to your cluster). One A100-class GPU, no Teacher calls.
#   sbatch --export=ALL,MAPSTEAL_REPO=$PWD,CORPUS=runs/smoke_medicine/corpus.jsonl,STUDENT=2B,PANELS=/path/to/panels,OUT=runs/smoke_medicine/student,LIMIT=100 scripts/hpc_example.sbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -Eeuo pipefail
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"; conda activate "${MAPSTEAL_ENV:-mapsteal}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$MAPSTEAL_REPO/src:${PYTHONPATH:-}"
cd "$MAPSTEAL_REPO"
if [ -z "${SKIP_TRAIN:-}" ]; then
  python -m mapsteal train --corpus "$CORPUS" --student "$STUDENT" --output "$OUT" --seed "${SEED:-42}"
  python -m mapsteal evaluate --domain "${DOMAIN:-medicine}" --checkpoint "$OUT/adapter" --student "$STUDENT" --panels "$PANELS" --output "$OUT/../evaluation_subset" ${LIMIT:+--limit $LIMIT}
fi
if [ -n "${REFERENCE_ADAPTER:-}" ]; then   # optional: evaluate an archived adapter with the new evaluator (to compare with the archived predictions)
  python -m mapsteal evaluate --domain "${DOMAIN:-medicine}" --checkpoint "$REFERENCE_ADAPTER" --student "$STUDENT" --panels "$PANELS" --output "$OUT/../evaluation_reference_adapter" ${LIMIT:+--limit $LIMIT}
fi
