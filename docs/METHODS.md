# Methods

## MapSteal (`src/mapsteal`)
MapSteal acquires domain knowledge from a Teacher without looking at the Student: a **structured, recursive, Teacher-driven** expansion of a small set of frozen root questions, followed by a **fresh LoRA** adaptation of the pristine Student on the acquired QA. Nothing about the Student's uncertainty, weaknesses, downstream scores or semantic similarity is used anywhere.

### 1. Acquisition (`mapsteal/acquisition`, driver `branch.py`)
**State** — a node `(O, F, C, T, D, E)` = Object, Facet, Condition, TargetSpec (the knowledge to ask for), Decision type (`identify|choose|explain|compare|assess|calculate|predict|define|recall`), Evidence, plus `state_id`, `source_root_id`, `parent_state_id`, `generation_depth`, `child_type`. The **frontier** starts with the frozen roots (Medicine: 35, Cybersecurity: 40; `data/<domain>/roots*.jsonl`).
**One executed step (= one QA record = one unit of the budget B)**
1. *Select* the next state from the frontier with the **fair scheduler** (`scheduler.py`): among roots that still have a pending state, the root with the fewest executed non-root states; ties by the oldest pending item.
2. *Realize the question*: a root uses its frozen question; every deeper state gets a deterministic question built from `T` and `D` by fixed templates (`<domain>/realizer.py`; guards for context, template artefacts, duplicates, answer leakage). A failed realization is recorded (`REALIZATION_FAIL`) and costs no Teacher call.
3. *Ask the Teacher* (`teacher.py`, GPT-4.1 `gpt-4.1-2025-04-14`, temperature 0, strict JSON schema, **one request per executed step**, one counted retry after an invalid response). The prompt is the frozen contract + opportunity/compactness instructions (`<domain>/contract.py`, `PROMPT_C`) + the per-root **direction registry** (`graph.py`) + the current state. The reply contains `answer`, `explanation`, `reusable_knowledge` and four typed arrays of proposed children (`new_objects`, `new_facets`, `applicability_conditions`, `new_targets`), each with its own target, decision, origin and a **direction** declaration (an existing direction id or a new named direction).
4. *Validate* the reply (`schema.py`) and *admit children* deterministically (`engine.py`): a proposed child becomes a new frontier state unless a fixed rule rejects it (empty target, invalid decision, restatement of the parent or an ancestor target, target contained in the answer, identity already visited, D–T incompatibility, duplicate inside the call). Admitted children are attached to their direction in the root's graph (`DirectionGraph`, pure bookkeeping, exact-name merging only).
5. *Record* one trajectory row (`trajectory.jsonl`): state, realized question, answer, explanation, proposals, admitted children, graph events, prompt hash, per-attempt usage and cost.
**Budget and termination** — `B` counts executed (successful) steps, **not** API requests (retries, failed steps and reused ledger rows are additional/free). Acquisition stops when `B` steps have been executed (or on a hard cost halt, a kill rule, an exhausted frontier). The Teacher ledger makes identical requests free on resume.

### 2. Corpus (`mapsteal/corpus`)
Training corpus = [Medicine only: the frozen Seed40 cold-start QA] + the first `N` executed trajectory rows after the frozen QC (drop empty answer/explanation; drop a row whose normalised question was already kept, no top-up). Row: `{"input": "", "instruction": <question>, "output": "<answer>\n\nExplanation: <explanation>"}`. The formal exact-B500 corpora are `B_total = 500` (Medicine: 40 seed + 460 steps → 495 rows; Cybersecurity: 500 steps → 500 rows).

### 3. Student adaptation (`mapsteal/training`)
Fresh LoRA from the pristine Student (Qwen3.5-2B/4B/9B, exact revisions in `models.py`): **selected training configuration** — lr 3e-4, 2 epochs, rank 16, α 32, dropout 0.05, MLP-only targets (gate/up/down), linear schedule, warmup 0, effective batch 8 (1×8), max length 1024, gradient checkpointing, bf16, response-only loss, seeds 42/43/44 in the paper. No warm start.

### 4. Evaluation (`mapsteal/evaluation`)
Frozen panels, zero-shot lettered prompt, next-token option-letter argmax, batch size 1. Medicine: MedMCQA (500), MedQA (1,273), MedXpertQA-Text (2,450). Cybersecurity: CyberMetric (2,500; `cybermetric:1729` excluded from reported accuracy), SecEval (1,521), CTI-Bench (2,482). The panels are not distributed.

### What is not in MapSteal
No Student uncertainty or weakness detection, no downstream feedback, no semantic-similarity edges or knowledge-graph optimisation, no extra agents or filters beyond the frozen rules above.

# Baselines

Both adapted baselines are **task-specific adaptations** of earlier methods to the setting of this paper (Qwen3.5 Students, GPT-4.1 Teacher, a fixed budget of about 500 QA records, Medicine and Cybersecurity). They are not exact reproductions of the original methods, which do not specify every detail needed for this setting. This document describes what the code in `src/baselines/` does; the archived formal runs (`provenance/`) were produced with this code. Settings are collected in `configs/formal_settings.json` and checked against the archived run records by `tests/test_formal_settings.py`. The files under `src/baselines/qedks/qsrc/` and `src/baselines/evokd/` keep the module and identifier names of the formal runs (for example `pilot_b100_v2.py`) so that their hashes can be compared with `manifests/baseline_source_map.json`.

## EvoKD-Adapted

**Idea (original method).** A Teacher LLM repeatedly analyses where a small Student fails and generates new training data for it; the Student is trained online on that data. **Adaptation here:** the Student is a decoder LLM adapted with one LoRA adapter, and the "which answers are wrong" signal comes from the Student's likelihood of a Teacher reference answer instead of a calibrated classifier confidence.

**Initial state.** The Student starts from an identity adapter (LoRA r16, α32, dropout 0.05 on q/k/v/o/gate/up/down projections, B = 0, so the adapted model equals the pristine Student). The initial data D0 has **4 question/reference-answer rows per domain** (`data/evokd/`; not distributed, supply your own with `--d0`). Before training, the Student identifies D0.

**Weakness identification (Identify).** For a row with question *q* and Teacher reference answer *a*, the score is the mean token log-probability of *a* given the Student prompt for *q* under the current adapter (reference tokens only, end-of-sequence token excluded, evaluation mode). The row is labelled *correct* if the score is ≥ τ = −0.5494510531425476 and *wrong* otherwise; there is no third class. τ was fixed beforehand from an offline calibration (maximum balanced accuracy) on earlier trajectories; the calibration data are not part of this repository. The Student's own greedy answer (≤ 96 new tokens) is generated for every row and shown to the Teacher together with the reference answer. The Student prompt is "Answer the following {medical|cybersecurity} question using only your current knowledge. Give a concise, direct answer." followed by the question.

**One chat event (3 Teacher requests, `gpt-4.1`, temperature 0).**
1. *Weakness analysis:* the user prompt states the task ("a language model that answers {domain} knowledge questions with a short, direct answer"), lists the rows of the previous batch labelled incorrect and correct (question, reference answer, model answer) and asks for a summary of the patterns of questions prone to be answered incorrectly (free text, ≤ 1,024 output tokens).
2. *Generation (same conversation):* the Teacher is asked for *n*_hard questions the model would answer incorrectly and *n*_easy it would answer correctly (self-contained, single clearly correct answer); structured JSON output.
3. *Labeling (new conversation, no Student information):* the questions are answered concisely and directly; structured JSON output. Rows with empty answers are discarded; no other filter, deduplication or admission rule exists.
Each event produces 8 rows (4 hard + 4 easy); after the new rows are identified with the current Student they are used for one training step. The last event has 4 rows (2 + 2), so the budget is 62 × 8 + 4 = **500 candidate rows** from **63 chat events** (planned 189 Teacher requests).

**Online schedule (training steps, not events).** Step *t* is a *review* step if *t* mod 50 = 0 (batch: 8 rows sampled uniformly from the history of D0 and all generated rows), else a *chat* step if *t* mod 40 = 0 (new event as above, then one training step on its rows; a chat step that coincides with a review step is a review step), else a *repeat* step (train again on the current batch; before the first event, on rows sampled from D0). The 63rd chat event is at step 3,120 and the run ends 39 steps later: **3,159 steps** = 63 chat + 63 review + 3,033 repeat.

**Training.** One adapter and one AdamW optimizer for the whole run: learning rate 2e-5 (constant), weight decay 0, gradient-norm clip 2.0, batch ≤ 8 rows (micro-batches of 4, or 2 for 2B/9B with gradient checkpointing for 9B, memory only), maximum length 1024, bf16 autocast with fp32 adapter weights, loss = mean over rows of the per-row mean cross-entropy on the answer tokens. The final model is the adapter after the last step; there is no retraining. Formal runs: 3 Students × 2 domains × seeds 42/43/44 (18 online trajectories; the seed fixes the adapter initialisation, the review/repeat sampling and the training seed).

**Domain adaptation.** The Medicine and Cybersecurity packages (`paper_faithful/`, `cyber/`) differ only in domain words of the prompts and in the Student prompt.

**Implementation choices the original method leaves open here:** batch size 8 per event with a 4/4 hard–easy split; the likelihood-based Identify step and its threshold; an initial state of 4 rows; sampling of D0 rows for repeat steps before the first event; structured JSON outputs for generation and labeling; LoRA with a single persistent optimizer; and the learning rate. The records do not contain a justification for choosing these values over alternatives.

**Known limitations.** The Identify step labels a row from the likelihood of one reference string, so it is sensitive to answer formatting and phrasing (for example about 21% of references carry a list-number prefix) and can label equivalent answers "wrong" (an internal analysis of the archived trajectories found 541 of 4,971 wrong-labelled records whose model answer is identical to the reference). The Teacher is told these answers are incorrect. A same-input replay of one 9B run diverged from step 1,281 and changed the Medical3 average by up to about 3 points, so the large seed variance of some cells cannot be attributed to the acquisition trajectory alone. Teacher accounting is incomplete for four runs (see below).

**Teacher-call accounting.** The schedule plans 189 Teacher requests per trajectory (3 × 63). The archived `train_summary.json` of 14 runs records 189. Four runs record fewer: `med_9B_s42` 180, `med_9B_s43` 10, `med_9B_s44` 183 and `cyb_s44` 24. The planned count must not be read as verified consumption for these four: the complete per-request ledger is not part of this repository, so their recorded Teacher totals (calls and cost) are incomplete. Recorded costs of the other runs are at most $0.84 per trajectory.

## QEDKS-Adapted

**Idea (original method).** The Student asks the questions: entities are extracted from Teacher answers, the Student generates questions about them, the question the Student is least sure about is sent to the Teacher, and follow-up questions and corrections expand the frontier. **Adaptation here (code: `src/baselines/qedks/`):**
- *Roots:* 30 root entities generated by the Teacher (temperature 0).
- *Per step:* choose among template and Student-refined questions of queued entities by the perplexity of the Student's own greedy answer (≤ 192 new tokens, chat template with thinking disabled; undefined perplexity ranks last); a step-seeded 50% mix of entity and question candidates; Teacher answer (`gpt-4.1`, chat completions, temperature 0, ≤ 1,024 tokens); if the Student's answer is judged bad (sentence-embedding similarity threshold 0.5) a Teacher correction is added; entity expansion (≤ 5 per answer); chain-of-questions follow-ups (broadness threshold 0.38, **minimum answer length 100 words instead of the original 220**, ≤ 3 follow-ups); k-means diversification of the frontier every 25 iterations (3 lowest-coverage clusters), a mechanism added in this adaptation.
- *Budget:* the loop stops when 500 QA rows (main + correction rows) exist, with a safety cap of 1,500 iterations; training rows are the 500 rows after dropping empty answers.
- *Training:* the same fresh-LoRA configuration as MapSteal (`configs/formal_settings.json`: lr 3e-4, 2 epochs, rank 16, α 32, dropout 0.05, MLP projections only, linear schedule); the original QEDKS fine-tuning (lr 2e-4, 1 epoch, 512 tokens, fp16) is not used. Acquisition seed = training seed ∈ {42, 43, 44}, i.e. three independent trajectories per setting.
- *Other changes to the original code (see the `[QEDKS-ADAPTED …]` comments):* Qwen3.5 chat template and LoRA target mapping; the Teacher answer cache replaced by a request ledger; step-seeded random numbers instead of global state; per-(prompt, kind) deduplication of training rows; a corrected regular expression and removal of one question-rejection rule in the question refinement; the Cybersecurity prompts rewritten from the Medicine ones.

**Benchmark-aware Medicine prompts.** The Medicine prompts of the original code were kept and mention benchmark formats: the Teacher answer style asks for "MedQA-style" and "PubMedQA-style" answers, the root-entity prompt asks for entities that "would plausibly appear in MedQA, PubMedQA, MedMCQA, or clinical exam-style questions", and the follow-up prompts ask for "benchmark-oriented" questions. The Cybersecurity prompts use generic wording and name no benchmark. QEDKS-Adapted Medicine therefore receives benchmark-format hints that MapSteal and EvoKD-Adapted do not; this may advantage it relative to the other methods.

`verify` checks an acquisition run (500 rows, seed/domain, code hashes in `manifests/qedks_code_hashes.json`, quality filter, and — when the raw logs are present — request uniqueness, model, temperature, cost cap).

## Pristine

The unadapted Student, evaluated with the shared evaluator (`python -m baselines pristine evaluate`, equivalent to `python -m mapsteal evaluate --checkpoint none`). The archived Pristine predictions of Table 1 were produced by the original evaluation scripts of several earlier runs; the shared evaluator reproduced all 600 archived predictions in a 4B spot check (100 items per benchmark, `validation/baseline_gpu_check/`).

## Budgets, API cost and variance across methods

All methods are allotted about **500 Teacher QA records** per trajectory, but they differ in what a record costs and how the training is scheduled:

| | MapSteal | QEDKS-Adapted | EvoKD-Adapted |
|---|---|---|---|
| Teacher QA records | 500 (Medicine: 40 cold-start + 460 trajectory) | 500 (main + correction rows) | 500 candidate rows (63 events) |
| Teacher requests per trajectory | 1 per executed step (+ retries); 501 attempts recorded for Cybersecurity | 841–903 in the archived runs (answers, corrections, entity extraction, roots) | 189 planned (3 per event); recorded 189 for 14 runs, partial for 4 |
| API cost (recorded) | about $4.9 for Cybersecurity (1 retry; includes a short earlier phase of the same trajectory); Medicine not recorded here | $0.87–1.34 | ≤ $0.84 (incomplete for 4 runs) |
| Student training | fresh LoRA, 2 epochs over the corpus (124 optimizer steps for 495 rows, 126 for 500) | same as MapSteal | online, 3,159 steps of ≤ 8 rows, lr 2e-5, no retraining |
| Trajectories behind a cell's SD | **one** acquisition trajectory per domain, three training seeds | three independent acquisitions, each trained once | three independent online runs |

MapSteal's per-QA token usage is higher because each request carries the state, contract and direction registry. A comparison by API cost or by training compute is therefore not implied by the common 500-record budget.

**Variance.** The standard deviations in Table 1 are not estimated the same way: for MapSteal they reflect training-seed variation on a fixed corpus; for QEDKS-Adapted and EvoKD-Adapted they include acquisition variation (different Teacher/Student trajectories). The MapSteal SDs are therefore not comparable with the baseline SDs, and acquisition variance of MapSteal is not estimated.

**Provenance of starting data.** The Medicine corpus begins with the 40 cold-start QA rows ("Seed40"), recorded in the corpus manifest as Teacher-answered rows of an earlier lineage outside the trajectory; the repository does not record which Teacher model produced them. Their questions are not distributed. The EvoKD-Adapted initial state is 4 rows per domain; the Medicine rows record their source dataset (MedQA, MedMCQA, MMLU-Med, PubMedQA) and the Cybersecurity rows record a cybersecurity topic area; the data splits they come from are not recorded in this repository.
