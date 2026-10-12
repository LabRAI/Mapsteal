"""Final paper-faithful EvoKD run: Algorithm 1 executed literally (see
alg1.py for the step semantics). All heavy dependencies are injected so the
whole loop is testable without GPU or network:

  trainer_factory(adapter_dir, optimizer_state) -> object with
      train_step(rows) / generate(prompt) / save_state(dir)
  teacher: final_teacher.TeacherChatClient (or a fake with .call/.prompt_provider/.totals)

Per chat event i (paper lines 12-18): weakness analysis (conversation turn 1)
-> generation of ceil(b/2) hard + floor(b/2) easy questions (turn 2, same
conversation) -> labeling (NEW conversation) -> Identify(model^{i-1}, D^i)
-> one training step on D^i. No admission/dedup/format filter of any kind.

Resume: a pre-chat state (adapter + optimizer + history + Identify results)
is written BEFORE each chat event's Teacher calls; the chat record is written
right after them, so a restart never re-spends (CallLedger IDs are one-shot).
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import alg1, final_prompts as fp, identify_v2, pilot_b100_v2 as pilot
from .student_backend import student_answer_prompt


@dataclass
class RunConfig:
    root: Path
    d0_rows: list[dict[str, Any]]
    s0_adapter_dir: Path
    n_chat_events: int = 13
    chat: int = alg1.CHAT
    review: int = alg1.REVIEW
    student_batch: int = alg1.STUDENT_BATCH
    stop_after_steps: int | None = None  # tests only
    mode: Any = None  # None -> FreeTextMode (the literal free-text baseline)
    keep_states: tuple[int, ...] | None = None  # None = keep every pre_chat state; else keep these event indices + the newest two


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")


def to_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"prompt": student_answer_prompt(r["question"]), "output": r["reference_answer"]} for r in rows]


def _judge_factory(teacher, request_id: str) -> Callable[[int], Any]:
    return lambda n: teacher.prompt_provider(
        request_id=request_id, schema_name=identify_v2.SEMANTIC_JUDGE_SCHEMA_NAME,
        schema=identify_v2.semantic_judge_response_schema(n))


def _identify(trainer, teacher, request_id: str, rows: list[dict[str, Any]]):
    identify_fn, diagnostics = pilot.build_identify_fn_with_diagnostics(_judge_factory(teacher, request_id))
    result = identify_fn(student_answer_fn=lambda q: trainer.generate(student_answer_prompt(q)), batch=rows)
    return result, diagnostics


def acquire_batch(*, i: int, wrong: list, correct: list, teacher, n_hard: int, n_easy: int) -> dict[str, Any]:
    """Paper 3.1: weakness analysis + input text generation in ONE conversation,
    labeling in a new one. Returns the raw material plus the parsed rows."""
    weak_prompt = fp.weakness_user_prompt(wrong, correct)
    weak_text, weak_meta = teacher.call(
        request_id=f"ef-e{i}-weakness", messages=[{"role": "user", "content": weak_prompt}],
        max_output_tokens=fp.WEAKNESS_TURN_MAX_OUTPUT_TOKENS)
    weak_text = weak_text.strip()
    if not weak_text:
        raise RuntimeError(f"event {i}: empty weakness analysis")
    gen_messages = fp.conversation_for_generation(weak_prompt, weak_text, n_hard, n_easy)
    gen_raw, gen_meta = teacher.call(request_id=f"ef-e{i}-gen", messages=gen_messages,
                                     schema_name="evokd-final-generation", schema=fp.generation_schema(n_hard, n_easy))
    parsed = json.loads(gen_raw)
    hard_qs, easy_qs = parsed["incorrect_questions"], parsed["correct_questions"]
    if len(hard_qs) != n_hard or len(easy_qs) != n_easy or not all(isinstance(q, str) and q.strip() for q in hard_qs + easy_qs):
        raise RuntimeError(f"event {i}: malformed generation output: {gen_raw!r}")
    questions = [(q.strip(), "hard") for q in hard_qs] + [(q.strip(), "easy") for q in easy_qs]
    label_prompt = fp.labeling_user_prompt([q for q, _ in questions])
    label_raw, label_meta = teacher.call(request_id=f"ef-e{i}-label", messages=[{"role": "user", "content": label_prompt}],
                                         schema_name="evokd-final-labeling", schema=fp.labeling_schema(len(questions)))
    answers = json.loads(label_raw)["answers"]
    if len(answers) != len(questions):
        raise RuntimeError(f"event {i}: labeling returned {len(answers)} answers for {len(questions)} questions")
    rows, unusable = [], []
    for (q, diff), a in zip(questions, answers):
        (rows if isinstance(a, str) and a.strip() else unusable).append(
            {"question": q, "reference_answer": (a or "").strip(), "difficulty": diff})
    return {"weakness_prompt": weak_prompt, "weakness_text": weak_text, "generation_messages": gen_messages,
            "generation_raw": gen_raw, "label_prompt": label_prompt, "label_raw": label_raw,
            "rows": rows, "unusable_rows": unusable,
            "usage": {"weakness": weak_meta.get("usage"), "generation": gen_meta.get("usage"),
                      "labeling": label_meta.get("usage")}}


class FreeTextMode:
    """The literal free-text baseline: behaviour of the original run_algorithm."""
    name = "free_text"

    def baseline_identify(self, trainer, teacher, d0_rows):
        return _identify(trainer, teacher, "ef-e0-judge", d0_rows)

    def training_rows(self, rows):
        return to_training_rows(rows)

    def acquire(self, *, i, wrong, correct, trainer, teacher, n_hard, n_easy):
        acq = acquire_batch(i=i, wrong=wrong, correct=correct, teacher=teacher, n_hard=n_hard, n_easy=n_easy)
        result, diag = _identify(trainer, teacher, f"ef-e{i}-judge", acq["rows"])
        return {**acq, "wrong": result["wrong"], "correct": result["correct"], "uncertain": result["uncertain"],
                "identify_diagnostics": diag}


def run_algorithm(cfg: RunConfig, *, trainer_factory: Callable[..., Any], teacher) -> dict[str, Any]:
    root = Path(cfg.root)
    events_dir, states_dir = root / "events", root / "states"
    step_log = root / "step_log.jsonl"
    for d in (events_dir, states_dir):
        d.mkdir(parents=True, exist_ok=True)
    mode = cfg.mode or FreeTextMode()
    sched = alg1.plan(cfg.n_chat_events, cfg.chat, cfg.review)
    n_hard = -(-cfg.student_batch // 2)   # ceil(b/2) hard
    n_easy = cfg.student_batch // 2       # floor(b/2) easy
    num_steps = sched["num_steps"] if cfg.stop_after_steps is None else cfg.stop_after_steps
    final_meta_path = states_dir / "final" / "meta.json"
    extending = False
    if final_meta_path.is_file():
        fmeta = json.loads(final_meta_path.read_text())
        if num_steps <= fmeta["step"]:
            return json.loads((root / "train_summary.json").read_text())
        extending = True   # a shorter finished run: continue the SAME trajectory (same schedule prefix)

    # ------------------------------------------------ resume or fresh start
    resume_dirs = sorted((p for p in states_dir.glob("pre_chat_*") if (p / "meta.json").is_file()),
                         key=lambda p: int(p.name.rsplit("_", 1)[1]))   # numeric: pre_chat_100 > pre_chat_13
    later_pre = [p for p in resume_dirs if extending and json.loads((p / "meta.json").read_text())["step"] > fmeta["step"]]
    if extending and not later_pre:
        # rebuild everything the loop needs from the finished run's own records
        trainer = trainer_factory(states_dir / "final" / "adapter", states_dir / "final" / "optimizer.pt")
        step, i = fmeta["step"], fmeta["i"]
        chat_recs = [json.loads((events_dir / f"event_{k:02d}_chat.json").read_text()) for k in range(1, i + 1)]
        history = alg1.HistoryCache(cfg.d0_rows)
        for k, r in enumerate(chat_recs, start=1):
            history.add(k, r["rows"])
        current_rows = chat_recs[-1]["rows"]
        wrong_prev, correct_prev = chat_recs[-1]["wrong"], chat_recs[-1]["correct"]
        consecutive_uncertain = 0
        # keep the old final state under its own name; a new "final" is written at the end
        (states_dir / f"final_at_step_{fmeta['step']:04d}").mkdir(exist_ok=True)
        for f in ("adapter", "optimizer.pt", "meta.json"):
            src = states_dir / "final" / f
            dst = states_dir / f"final_at_step_{fmeta['step']:04d}" / f
            if not dst.exists():
                src.rename(dst)
        if (root / "train_summary.json").is_file():
            (root / f"train_summary_at_step_{fmeta['step']:04d}.json").write_text((root / "train_summary.json").read_text())
    elif resume_dirs:
        if extending:
            resume_dirs = later_pre
        latest = resume_dirs[-1]
        meta = json.loads((latest / "meta.json").read_text())
        trainer = trainer_factory(latest / "adapter", latest / "optimizer.pt")
        step = meta["step"] - 1                      # the chat step itself is re-executed
        i = meta["i"] - 1
        wrong_prev, correct_prev = meta["wrong_prev"], meta["correct_prev"]
        history = alg1.HistoryCache([])
        history.rows = meta["history_rows"]
        current_rows = meta["current_rows"]
        kept = [json.loads(l) for l in step_log.read_text().splitlines() if l.strip()] if step_log.is_file() else []
        kept = [r for r in kept if r["step"] < meta["step"]]
        step_log.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept), encoding="utf-8")
        consecutive_uncertain = meta.get("consecutive_all_uncertain", 0)
    else:
        trainer = trainer_factory(cfg.s0_adapter_dir, None)
        base_result, base_diag = mode.baseline_identify(trainer, teacher, cfg.d0_rows)
        _write_json(root / "baseline_d0_identify.json", {
            "model": "S0 (frozen model^0)", "wrong": base_result["wrong"], "correct": base_result["correct"],
            "uncertain": base_result["uncertain"], "diagnostics": base_diag})
        step, i = 0, 0
        wrong_prev, correct_prev = base_result["wrong"], base_result["correct"]
        history = alg1.HistoryCache(cfg.d0_rows)
        current_rows: list[dict[str, Any]] = []
        consecutive_uncertain = 0
        if step_log.is_file():
            step_log.unlink()

    def log_step(action: str, event: int, rows: list[dict[str, Any]], metrics: dict[str, Any], t0: float, extra=None):
        rec = {"step": step, "action": action, "event": event, "n_rows": len(rows),
               "loss": metrics["loss"], "grad_norm": metrics["grad_norm"],
               "n_label_tokens": metrics["n_label_tokens"], "seconds": time.time() - t0, **(extra or {})}
        with step_log.open("a", encoding="utf-8") as h:
            h.write(json.dumps(rec, sort_keys=True) + "\n")

    # ------------------------------------------------------------- the loop
    while step < num_steps:
        step += 1
        t0 = time.time()
        act = alg1.action_for_step(step, cfg.chat, cfg.review)
        if act == "review":
            rows = history.sample(cfg.student_batch, f"evokd-final-v1-review-{step}")
            m = trainer.train_step(mode.training_rows(rows))
            log_step("review", i, rows, m, t0, {"row_sources": [r["source_event"] for r in rows]})
        elif act == "chat":
            i += 1
            pre = states_dir / f"pre_chat_{i:02d}"
            if not (pre / "meta.json").is_file():
                info = trainer.save_state(pre)
                _write_json(pre / "meta.json", {
                    "step": step, "i": i, "wrong_prev": wrong_prev, "correct_prev": correct_prev,
                    "history_rows": history.rows, "current_rows": current_rows,
                    "consecutive_all_uncertain": consecutive_uncertain, **info})
            chat_path = events_dir / f"event_{i:02d}_chat.json"
            if chat_path.is_file():
                rec = json.loads(chat_path.read_text())
            else:
                acq = mode.acquire(i=i, wrong=wrong_prev, correct=correct_prev, trainer=trainer, teacher=teacher,
                                   n_hard=n_hard, n_easy=n_easy)
                rec = {**acq, "iteration": i, "step": step, "n_wrong_input": len(wrong_prev),
                       "n_correct_input": len(correct_prev), "teacher_totals_after": teacher.totals()}
                _write_json(chat_path, rec)
            consecutive_uncertain = pilot.check_semantic_judge_not_systematically_failing(
                diagnostics=rec["identify_diagnostics"], consecutive_all_uncertain=consecutive_uncertain)
            if not rec["rows"]:
                raise pilot.PilotStopped(f"event {i}: no usable rows after labeling")
            current_rows = rec["rows"]
            history.add(i, current_rows)
            wrong_prev, correct_prev = rec["wrong"], rec["correct"]
            m = trainer.train_step(mode.training_rows(current_rows))
            log_step("chat", i, current_rows, m, t0)
            if cfg.keep_states is not None:   # disk: each state is ~260 MB (adapter + optimizer)
                for d in states_dir.glob("pre_chat_*"):
                    k = int(d.name.rsplit("_", 1)[1])
                    if k not in cfg.keep_states and k < i - 1:
                        shutil.rmtree(d)
        else:
            rows = alg1.repeat_batch(step, i, current_rows, cfg.d0_rows, cfg.student_batch)
            m = trainer.train_step(mode.training_rows(rows))
            log_step("repeat", i, rows, m, t0)

    final = trainer.save_state(states_dir / "final")
    _write_json(states_dir / "final" / "meta.json", {"step": step, "i": i, **final})
    steps = [json.loads(l) for l in step_log.read_text().splitlines() if l.strip()]
    summary = {"schedule": sched, "num_steps_run": step, "chat_events": i,
               "actions": {a: sum(1 for s in steps if s["action"] == a) for a in ("chat", "review", "repeat")},
               "history_size": len(history), "teacher_totals": teacher.totals(), "final": final}
    _write_json(root / "train_summary.json", summary)
    return summary
