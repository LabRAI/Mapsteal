# MapSteal

MapSteal acquires domain knowledge from a Teacher LLM without looking at the Student: it expands a small set of root questions into a tree of knowledge states (one Teacher request per QA record, fixed admission rules), trains a pristine Qwen3.5 Student on the resulting corpus with a fresh LoRA, and evaluates it zero-shot on multiple-choice benchmarks (Medicine: MedMCQA, MedQA, MedXpertQA-Text; Cybersecurity: CyberMetric, SecEval, CTI-Bench). This repository contains the implementation, the baselines of the paper's Table 2 (QEDKS-Adapted, EvoKD-Adapted, Pristine), the archived Student predictions, and the scripts to recompute Table 2. Method details: [docs/METHODS.md](docs/METHODS.md); full commands and settings: [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md).

## Installation
Python ≥ 3.9.
```bash
git clone git@github.com:LabRAI/Mapsteal.git && cd Mapsteal
pip install -e .                    # Table 2 recomputation needs only the standard library
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
python tools/prepare_benchmarks.py gold --panels panels                        # gold labels -> data/gold/ (used by Table 2)
```
Check each benchmark's terms before use (`manifests/benchmark_sources.json` lists release and listed license). Sources, pinned versions and scoring conventions: [guide §1](docs/REPRODUCTION_GUIDE.md#1-data).

## Reproducing Table 2
```bash
bash run.sh --aggregate-only        # 180 prediction files in data/predictions/ + data/gold/ -> results/table2.tex, table2_summary.csv, table2_per_run.csv
python3 validation/validate.py      # independent recomputation of all 240 run-level accuracies
```
Table 2 = mean ± sample SD (seeds 42/43/44; Pristine one model) of 72 cells: 4 methods × 3 Students (2B/4B/9B) × 6 benchmarks. This recomputes the table from the recorded predictions (CPU only, no Teacher, no training); retraining the Students is described below.

**Output of `bash run.sh --aggregate-only`** (CPU, a few seconds; beginning and end of the terminal output; the full tables are in `results/`):
```text
MEDICINE  (accuracy %)

 Student      Method                  MedMCQA           MedQA        MedXpert
 ────────────────────────────────────────────────────────────────────────────
 Qwen3.5-2B   Pristine                  45.80           36.06           10.78
              MapSteal (Ours)   47.87 ± 1.03*   45.59 ± 0.16*    14.65 ± 0.19
              QEDKS-Adapted      43.27 ± 0.12    44.59 ± 0.81   14.88 ± 0.83*
              EvoKD-Adapted      40.53 ± 8.09    41.19 ± 8.47    12.39 ± 1.57

  ...
╭─ Verification Summary ───────────────────────────────────────────────────────────────────────╮
│ Prediction files              180/180 VERIFIED                                               │
│ Table 2 benchmark cells       72/72 COMPLETE                                                 │
│ Domain macro summaries        24/24 COMPLETE                                                 │
│ Aggregation integrity checks  PASS                                                           │
│ Last full validation          ALL CHECKS PASS  (report written 2026-10-16 20:26)             │
│ Output files                  results/  (table2.tex, table2_summary.csv, table2_per_run.csv) │
│                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
```
`python3 validation/validate.py` ends with `Overall: ALL CHECKS PASS`.

**Table 2** (accuracy %, generated into `results/table2.tex`; mean ± sample SD over seeds 42/43/44, Pristine is one model; CyberMetric excludes `cybermetric:1729`, so n = 2,499):

| Student | Method | MedMCQA | MedQA | MedXpertQA-Text | CyberMetric | SecEval | CTI-Bench |
|---|---|---:|---:|---:|---:|---:|---:|
| 2B | Pristine | 45.80 | 36.06 | 10.78 | 79.47 | 71.93 | 60.64 |
| 2B | **MapSteal (Ours)** | 47.87 ± 1.03 | 45.59 ± 0.16 | 14.65 ± 0.19 | 79.70 ± 0.14 | 78.68 ± 0.56 | 62.41 ± 1.07 |
| 2B | QEDKS-Adapted | 43.27 ± 0.12 | 44.59 ± 0.81 | 14.88 ± 0.83 | 79.49 ± 0.23 | 77.73 ± 1.46 | 65.48 ± 0.16 |
| 2B | EvoKD-Adapted | 40.53 ± 8.09 | 41.19 ± 8.47 | 12.39 ± 1.57 | 71.00 ± 6.19 | 69.49 ± 6.25 | 52.73 ± 9.42 |
| 4B | Pristine | 56.60 | 62.14 | 16.73 | 83.63 | 80.60 | 65.71 |
| 4B | **MapSteal (Ours)** | 62.27 ± 0.42 | 66.69 ± 0.44 | 17.63 ± 0.46 | 84.61 ± 0.16 | 83.19 ± 0.33 | 66.59 ± 0.71 |
| 4B | QEDKS-Adapted | 61.00 ± 0.53 | 65.54 ± 0.20 | 16.94 ± 0.45 | 84.21 ± 0.24 | 81.46 ± 0.96 | 67.30 ± 0.54 |
| 4B | EvoKD-Adapted | 58.47 ± 1.10 | 63.45 ± 0.66 | 16.22 ± 0.70 | 82.97 ± 0.13 | 80.28 ± 0.75 | 63.79 ± 0.93 |
| 9B | Pristine | 62.60 | 67.40 | 19.55 | 83.75 | 83.04 | 64.99 |
| 9B | **MapSteal (Ours)** | 65.60 ± 0.20 | 71.14 ± 0.43 | 20.18 ± 0.08 | 86.66 ± 0.12 | 85.32 ± 0.43 | 69.04 ± 0.77 |
| 9B | QEDKS-Adapted | 65.13 ± 0.90 | 71.20 ± 0.71 | 19.88 ± 0.50 | 86.75 ± 0.14 | 84.18 ± 0.40 | 70.52 ± 0.19 |
| 9B | EvoKD-Adapted | 59.00 ± 5.72 | 67.69 ± 5.64 | 19.85 ± 0.42 | 84.10 ± 1.00 | 81.00 ± 2.22 | 66.95 ± 1.73 |

MapSteal uses one acquisition trajectory per setting with three training seeds; QEDKS-Adapted and EvoKD-Adapted use three independent acquisition/training trajectories, so their standard deviations are not estimated the same way ([METHODS.md](docs/METHODS.md)).

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
`src/mapsteal/` (method), `src/baselines/` (QEDKS-Adapted, EvoKD-Adapted, Pristine), `configs/formal_settings.json` (final hyper-parameters, seeds, budgets), `data/predictions/` (archived Student predictions: sample id + predicted option), `manifests/` (Table 2 manifest, panel hashes, benchmark releases), `provenance/` (archived training and acquisition run records), `results/` (Table 2 files), `tools/`, `tests/`, `validation/`, `docs/`.

Tests: `bash tests/run_cpu_tests.sh` and `python3 validation/validate_impl.py` (CPU; checks that need undistributed Teacher data print SKIPPED).

## Citation and license
Please cite the paper when it is available. No license has been chosen yet. Benchmarks, Qwen3.5 weights and GPT-4.1 outputs are subject to their own terms; the QEDKS-Adapted code derives from the authors' earlier research code.
