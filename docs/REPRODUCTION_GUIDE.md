# Reproduction guide

Method descriptions and baseline details: [METHODS.md](METHODS.md). Final settings: [`configs/formal_settings.json`](../configs/formal_settings.json) (checked against the archived runs by `tests/test_formal_settings.py`).

## 1. Data

### Benchmarks (needed for evaluation and Table 1)
`tools/prepare_benchmarks.py` downloads each benchmark from its own release, rebuilds the six frozen panels and checks every panel against its recorded sha256 (`manifests/panel_hashes.json`); `manifests/benchmark_sources.json` pins the exact release, file hash and the license shown on the source page when checked. Read each benchmark's terms yourself.

| benchmark | release used | source |
|---|---|---|
| MedQA (USMLE test, 1,273 items, 5 options) | `data_clean/questions/US/test.jsonl` of the MedQA data archive | Google Drive archive linked from github.com/jind11/MedQA |
| MedMCQA (500 validation items) | validation split, Hugging Face release | `openlifescienceai/medmcqa` (pinned revision) |
| MedXpertQA-Text (2,450) | `Text/test.jsonl` | `TsinghuaC3I/MedXpertQA` (pinned revision) |
| CyberMetric (2,500 of CyberMetric-10000; 2,499 scored) | commit `205262cd…` (2024-12-19) of `CyberMetric-10000-v1.json` | github.com/cybermetric/CyberMetric |
| SecEval (1,521) | `questions.json` | `XuanwuAI/SecEval` (pinned revision) |
| CTI-Bench MCQ (2,482) | `cti-mcq.tsv` | `AI4Sec/cti-bench` (pinned revision) |

```bash
python tools/prepare_benchmarks.py download external                           # needs internet; ~140 MB (MedQA archive)
python tools/prepare_benchmarks.py panels --source external --output panels    # needs pandas+pyarrow, and transformers with the Qwen3.5-4B tokenizer in the HF cache
python tools/prepare_benchmarks.py gold --panels panels                        # writes data/gold/*.json (git-ignored)
```
Notes: the item subsets are those named by `data/predictions/` (sample ids and order), so the panels carry the same items the Students were evaluated on. If the MedQA download fails (Google Drive may require a manual step), download the archive from the MedQA project page and extract `data_clean/questions/US/test.jsonl` to `external/medqa_US_test.jsonl`. The CyberMetric repository has since been updated (2026-05-27); the pinned commit is the version that reproduces the frozen panel byte for byte, so do not substitute the current file. Processing conventions inherited from the frozen panels: SecEval items with several correct letters (284 of 1,521) or none (7) are scored against the first listed letter ('A' if none); five CTI-Bench items with fewer than four options show the option text `nan` in the prompt; CyberMetric item 1729 is excluded for all models (a MapSteal Cybersecurity acquisition question re-worded it with the same answer target; it changes CyberMetric accuracy by about 0.03 points).

### Root questions (included)
`src/mapsteal/data/medicine/roots35.jsonl` (35) and `src/mapsteal/data/cybersecurity/roots40.jsonl` (40) are the frozen roots of the formal acquisitions, unchanged; use `--roots FILE` for your own. Recorded provenance: the 40 Cybersecurity roots are hand-written questions tied to CyBOK v1.2 section headings (`source`, `location` fields; no model was used and no CyBOK text is copied). The Medicine roots are of two kinds: 5 (`ROOT-05/10/18/27/34`) equal Seed40 questions from an earlier lineage, and 30 (`REFRESHM-01..30`) were proposed by the pipeline's REFRESH step; the records do not name the model that produced them, so they should not be read as hand-written.

### Teacher data and starting rows (not distributed)
The GPT-4.1 training corpora, acquisition transcripts, the 40 Medicine cold-start rows, the 4-row EvoKD initial states and the archived Teacher responses used by some tests are not part of this repository (their redistribution has not been cleared); `manifests/restricted_inputs.json` lists them with size and sha256. Consequences:
- **Fresh runs do not need them** except the starting rows: MapSteal Medicine builds its corpus from the acquisition rows only unless you pass `--seed-corpus FILE` (JSONL with `instruction`/`output`/`input`); the formal corpora contained 40 extra cold-start rows. EvoKD-Adapted needs 4 initial rows per domain: pass `--d0 FILE` (JSONL with keys `dataset`, `knowledge_object`, `question`, `reference_answer`, `source_id`).
- **Retraining on the archived corpora** (level B) and the tests that replay archived Teacher responses need the files themselves (available from the authors; `python tools/restore_restricted.py BUNDLE_DIR` restores them with sha256 checks).

## 2. Reproduce Table 1
```bash
python tools/prepare_benchmarks.py download external && python tools/prepare_benchmarks.py panels --source external --output panels && python tools/prepare_benchmarks.py gold --panels panels
bash run.sh --aggregate-only                      # reads manifests/table1_manifest.json, data/predictions/ (180 files), data/gold/; writes results/table1.tex, table1_summary.csv, table1_per_run.csv
python3 validation/validate.py                    # independent recomputation of the 240 run-level accuracies, structure checks, tamper test
```
`data/predictions/<method>/<size>/<domain>/<seed>/<benchmark>.jsonl` holds one `{"item_id", "pred"}` row per item in the original order (the archived files also stored labels and scores; those fields were dropped and the original sha256 is kept in the manifest). Accuracy = correct / scored items, mean ± sample SD (ddof 1) over seeds 42/43/44; Pristine is a single model. Executed with the workflow above: 180/180 prediction files verified, 72/72 cells and 24/24 macro summaries complete, outputs byte-identical to the archived Table 1 files. This recomputes the table from the recorded predictions; it does not retrain anything.

## 3. Experiments
| id | experiment | paper artifact | how |
|---|---|---|---|
| E1 | Table 1 from archived predictions | Table 1 | §2 |
| E2 | Pristine, 3 sizes × 2 domains | Table 1 | `baselines pristine evaluate` |
| E3 | MapSteal: acquisition (budget 500) per domain, training seeds 42/43/44 × 3 sizes × 2 domains | Table 1 | §4 |
| E4 | QEDKS-Adapted: 18 acquisitions + 18 trainings | Table 1 | §5 |
| E5 | EvoKD-Adapted: 18 online runs | Table 1 | §6 |
The paper source is not part of this repository, so other tables and figures cannot be mapped and no plotting script is included.

## 4. MapSteal
```bash
python -m mapsteal acquire  --domain medicine --budget 460 --output runs/mapsteal/med                  # plan + cost estimate
python -m mapsteal acquire  --domain medicine --budget 460 --output runs/mapsteal/med --execute        # GPT-4.1 calls; needs OPENAI_API_KEY (or --env-file); resumable; --spend-ceiling (default 5 USD)
python -m mapsteal corpus   --run runs/mapsteal/med --output runs/mapsteal/med/corpus.jsonl --validate   # acquire --execute also writes corpus.jsonl
python -m mapsteal train    --corpus runs/mapsteal/med/corpus.jsonl --student 4B --output runs/mapsteal/med/s42 --seed 42
python -m mapsteal evaluate --domain medicine --checkpoint runs/mapsteal/med/s42/adapter --student 4B --panels panels --output runs/mapsteal/med/s42/evaluation
```
Cybersecurity: `--domain cybersecurity --budget 500`. Formal budget 500: Medicine = 40 cold-start rows + 460 executed steps, Cybersecurity = 500 steps; a full acquisition costs about $5. `B` counts executed steps, not API requests. Training: one GPU, 15–25 min per adapter in the archived runs. Evaluation: one GPU. `bash run.sh --student 4B --domain medicine --method mapsteal --mode full` prints the formal plan (add `--execute` to run).

## 5. QEDKS-Adapted
```bash
python -m baselines qedks acquire --domain medicine --student 4B --seed 42 --output runs/qedks/acq --execute   # Teacher + GPU (about $1 and 1.4–2.8 h per run in the archived runs); drop --execute for a plan
python -m baselines qedks verify  --run runs/qedks/acq --domain medicine --student 4B --seed 42 --write         # checks the run and writes training_corpus.jsonl
python -m baselines qedks train   --corpus runs/qedks/acq/training_corpus.jsonl --student 4B --output runs/qedks/train --seed 42
python -m mapsteal evaluate --domain medicine --checkpoint runs/qedks/train/adapter --student 4B --panels panels --output runs/qedks/train/evaluation
```
Install `pip install -e ".[qedks]"`. Formal pairing: acquisition seed = training seed ∈ {42, 43, 44}.

## 6. EvoKD-Adapted
```bash
python -m baselines evokd run --domain medicine --student 4B --out runs/evokd/d --d0 d0_medicine.jsonl --dry-run                 # CPU schedule check with fake Teacher/trainer
python -m baselines evokd run --domain medicine --student 4B --out runs/evokd/med_s42 --d0 d0_medicine.jsonl --events 63 --seed 42 --execute
python -m mapsteal evaluate --domain medicine --checkpoint runs/evokd/med_s42/states/final/adapter --student 4B --panels panels --output runs/evokd/med_s42/evaluation
```
63 chat events, 3,159 training steps, 3 Teacher requests per event; one GPU for the online training. Use your own 4-row initial state with `--d0` (see §1). Install `pip install -e ".[evokd]"`.

## 7. Pristine
```bash
python -m baselines pristine evaluate --domain medicine --student 4B --panels panels --output runs/pristine/4B/medicine
```

## 8. Setup and status
Students: `Qwen/Qwen3.5-4B` from the Hugging Face cache; 2B/9B at `$MAPSTEAL_MODELS_DIR/Qwen3.5-<size>-<revision>` (default `~/models`); revisions in `configs/formal_settings.json`; jobs run offline. GPU environment: `torch==2.13.0`, `transformers==5.14.1`, `peft==0.19.1`, one A100-class GPU. `scripts/hpc_example.sbatch` is a SLURM example.

| workflow | status |
|---|---|
| benchmark preparation → Table 1 (§1–2) | executed end to end |
| `mapsteal acquire/corpus/train/evaluate` | each stage executed at small scale (Medicine B=40, Cybersecurity B=5, 2B Student, $0.46); replay of archived Teacher responses and corpus reconstruction verified; full-scale run not repeated |
| `baselines pristine evaluate` | executed on GPU for 4B (100 items per benchmark; 600/600 predictions identical to the archived ones) |
| QEDKS-Adapted acquire/train, EvoKD-Adapted `--execute` | argument parsing and CPU checks only; GPU/Teacher paths not run since the code was moved into this repository |
| EvoKD-Adapted `--dry-run`, QEDKS `verify`/`corpus` | executed on the archived runs |

CPU tests: `bash tests/run_cpu_tests.sh` (tests that need undistributed data print SKIPPED), `python3 validation/validate_impl.py`.

## 9. Notes on interpreting results
- Retraining is not bit-reproducible and Teacher outputs may differ between calls; expect differences of about the seed SD, and a new acquisition is a new trajectory.
- The training configuration (MapSteal, QEDKS-Adapted) was selected on Medicine 4B and applied unchanged to all other settings; the evaluation data used for that selection are not documented here.
- SDs are not estimated alike: MapSteal's reflect training seeds on one acquisition trajectory per domain; the baselines' include independent acquisitions. Methods also differ in API cost per record (METHODS.md). QEDKS-Adapted Medicine prompts name benchmark formats. Four EvoKD-Adapted runs have incomplete Teacher-call records (180, 10, 183, 24 of 189 planned).
- The archived predictions were produced by the original evaluation scripts of several runs; the shared evaluator was checked against MapSteal adapters and a Pristine subset.
