# MapSteal

MapSteal acquires domain knowledge from a Teacher LLM without looking at the Student: it expands a small set of root questions into a tree of knowledge states (one Teacher request per QA record, fixed admission rules), trains a pristine Qwen3.5 Student on the resulting corpus with a fresh LoRA, and evaluates it zero-shot on multiple-choice benchmarks (Medicine: MedMCQA, MedQA, MedXpertQA-Text; Cybersecurity: CyberMetric, SecEval, CTI-Bench). This repository contains the implementation, the baselines of the paper's Table 1 (QEDKS-Adapted, EvoKD-Adapted, Pristine), the archived Student predictions, and the scripts to recompute Table 1. Method details: [docs/METHODS.md](docs/METHODS.md); full commands and settings: [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md).

## Installation
Python ≥ 3.9.
```bash
git clone git@github.com:LabRAI/Mapsteal.git && cd Mapsteal
pip install -e .                    # Table 1 recomputation needs only the standard library
pip install -e ".[acquire,gpu]"     # Teacher acquisition (openai) and GPU training/evaluation
pip install -e ".[qedks]"           # + QEDKS-Adapted dependencies      (".[evokd]" for EvoKD-Adapted)
export OPENAI_API_KEY=...           # for Teacher calls; or pass --env-file PATH
```
GPU runs use `torch==2.13.0`, `transformers==5.14.1`, `peft==0.19.1` (see `requirements.txt`) on one A100-class GPU. Students are read offline: `Qwen/Qwen3.5-4B` from the Hugging Face cache, 2B/9B from `$MAPSTEAL_MODELS_DIR/Qwen3.5-<size>-<revision>` (default `~/models`; revisions in `configs/formal_settings.json`).

## Data preparation
The benchmarks are downloaded from their own releases and converted to the frozen evaluation panels (each panel is checked against its recorded sha256):
```bash
python tools/prepare_benchmarks.py download external                           # MedQA, MedMCQA, MedXpertQA, CyberMetric, SecEval, CTI-Bench (pinned releases, ~140 MB)
python tools/prepare_benchmarks.py panels --source external --output panels    # evaluation panels  (needs pandas, pyarrow; transformers + Qwen3.5-4B tokenizer in the HF cache)
python tools/prepare_benchmarks.py gold --panels panels                        # gold labels -> data/gold/ (used by Table 1)
```
Check each benchmark's terms before use (`manifests/benchmark_sources.json` lists release and listed license). Sources, pinned versions and scoring conventions: [guide §1](docs/REPRODUCTION_GUIDE.md#1-data).

## Reproducing Table 1
```bash
bash run.sh --aggregate-only        # 180 prediction files in data/predictions/ + data/gold/ -> results/table1.tex, table1_summary.csv, table1_per_run.csv
python3 validation/validate.py      # independent recomputation of all 240 run-level accuracies
```
Table 1 = mean ± sample SD (seeds 42/43/44; Pristine one model) of 72 cells: 4 methods × 3 Students (2B/4B/9B) × 6 benchmarks. The output matches the archived Table 1 files exactly (180/180 files, 72/72 cells, 24/24 macro summaries). This recomputes the table from the recorded predictions; retraining the Students is described below.

## Running MapSteal
```bash
python -m mapsteal acquire  --domain medicine --budget 460 --output runs/med                 # plan and cost estimate only
python -m mapsteal acquire  --domain medicine --budget 460 --output runs/med --execute       # Teacher calls (about $5 for the full budget); resumable
python -m mapsteal train    --corpus runs/med/corpus.jsonl --student 4B --output runs/med/s42 --seed 42
python -m mapsteal evaluate --domain medicine --checkpoint runs/med/s42/adapter --student 4B --panels panels --output runs/med/s42/evaluation
```
Roots: the formal 35 (Medicine) / 40 (Cybersecurity) root questions are in `src/mapsteal/data/`; pass `--roots FILE` to use your own. Formal budget 500: Medicine 40 cold-start rows + 460 steps (the 40 cold-start rows are not distributed; the corpus then contains the acquisition rows only, or pass `--seed-corpus FILE`), Cybersecurity `--domain cybersecurity --budget 500`. Training and evaluation need one GPU. Status: each stage was run at small scale (B=40 / B=5, 2B Student); the full budget run was not repeated.

## Running QEDKS-Adapted
```bash
python -m baselines qedks acquire --domain medicine --student 4B --seed 42 --output runs/qedks/acq --execute   # Teacher + GPU, about $1 per run
python -m baselines qedks verify  --run runs/qedks/acq --domain medicine --student 4B --seed 42 --write
python -m baselines qedks train   --corpus runs/qedks/acq/training_corpus.jsonl --student 4B --output runs/qedks/train --seed 42
python -m mapsteal evaluate --domain medicine --checkpoint runs/qedks/train/adapter --student 4B --panels panels --output runs/qedks/train/evaluation
```
Status: argument parsing and the CPU checks (`verify`, corpus export on the archived runs) were run; the GPU acquisition and training were not run since the code was moved into this repository.

## Running EvoKD-Adapted
```bash
python -m baselines evokd run --domain medicine --student 4B --out runs/evokd/dry --d0 d0.jsonl --dry-run                # CPU schedule check
python -m baselines evokd run --domain medicine --student 4B --out runs/evokd/s42 --d0 d0.jsonl --seed 42 --execute      # Teacher + GPU; adapter in runs/evokd/s42/states/final/adapter
python -m mapsteal evaluate --domain medicine --checkpoint runs/evokd/s42/states/final/adapter --student 4B --panels panels --output runs/evokd/s42/evaluation
```
`--d0` is a JSONL file with the 4 initial question/reference-answer rows (keys `dataset`, `knowledge_object`, `question`, `reference_answer`, `source_id`); the formal rows are not distributed. 63 chat events, 3,159 steps. Status: the dry run was checked against the archived runs; the Teacher/GPU path was not run since the code was moved.

## Evaluating Pristine
```bash
python -m baselines pristine evaluate --domain medicine --student 4B --panels panels --output runs/pristine/4B/medicine
```
Status: run on GPU for 4B (100 items per benchmark), identical to the archived predictions.

## Repository layout
`src/mapsteal/` (method), `src/baselines/` (QEDKS-Adapted, EvoKD-Adapted, Pristine), `configs/formal_settings.json` (final hyper-parameters, seeds, budgets), `data/predictions/` (archived Student predictions: sample id + predicted option), `manifests/` (Table 1 manifest, panel hashes, benchmark releases), `provenance/` (archived training and acquisition run records), `results/` (Table 1 files), `tools/`, `tests/`, `validation/`, `docs/`.

Tests: `bash tests/run_cpu_tests.sh` and `python3 validation/validate_impl.py` (CPU; checks that need undistributed Teacher data print SKIPPED).

## Citation and license
Please cite the paper when it is available. No license has been chosen yet. Benchmarks, Qwen3.5 weights and GPT-4.1 outputs are subject to their own terms; the QEDKS-Adapted code derives from the authors' earlier research code.
