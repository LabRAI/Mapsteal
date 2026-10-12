"""Algorithm 1 (EvoKD.pdf, page 5) driver -- the paper-faithful chat event:
weakness analysis -> ONE batch-generation call -> ONE batch-labeling call ->
Identify -> train. No per-root loop, no admission/rejection filter of any
kind (see DESIGN_v1.md section 5). Repeat-Batch and Review-History steps
(the paper's free, no-LLM-call training passes) are separate functions
below, not folded into this chat-event function.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..kf.ledger import CallLedger
from ..kf.schemas import ProviderResponse, TeacherRequest

from . import identify as identify_mod
from . import prompts
from .teacher_provider import usage_to_fields


class AlgorithmError(RuntimeError):
    pass


def _content_hash(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reserve_complete(*, ledger: CallLedger, request_id: str, prompt: str, provider_call: Callable[[], tuple[str, dict]],
                      trace_path: Path, request_type: str) -> tuple[str, dict[str, Any]]:
    req = TeacherRequest(request_id=request_id, prompt=prompt, content_hash=_content_hash(prompt),
                        provider="evokd-pf-teacher", hidden_retries=0)
    ledger.reserve(req)
    try:
        raw, meta = provider_call()
    except Exception as exc:
        ledger.fail(req, error=f"{type(exc).__name__}: {exc}")
        raise
    ledger.complete(req, ProviderResponse(request_id, raw, meta))
    fields = usage_to_fields(meta.get("usage"))
    trace_row = {
        "request_type": request_type, "request_id": request_id, "response_id": meta.get("response_id"),
        **fields,
    }
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace_row, sort_keys=True) + "\n")
    return raw, meta


def run_chat_event(*, iteration: int, wrong: list[dict[str, Any]], correct: list[dict[str, Any]],
                   n_easy: int, n_hard: int, weakness_provider, batch_gen_provider, batch_label_provider,
                   student_answer_fn: Callable[[str], str], ledger_path: Path, trace_path: Path,
                   identify_fn: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """One full chat event (Algorithm 1 line 14, expanded into the paper's
    3-call structure per Section 3.1): weakness -> batch-generate -> batch-
    label -> Identify on the fresh batch. Returns the new batch D^i (list of
    {"question","reference_answer","difficulty"}), the wrong/correct split
    on it, and raw call metadata for cost accounting. Makes NO admission/
    rejection decision beyond dropping items the Teacher itself marked
    "abstained" (recorded transparently, never retried).

    ``identify_fn``: optional override for the Identify() call, defaulting
    to identify.identify (byte-identical previous behavior when omitted).
    Must accept (student_answer_fn=..., batch=...) and return a dict with
    at least "wrong"/"correct" keys; an "uncertain" key, if present, is
    passed through unchanged (identify.identify never returns one, so the
    default path's returned dict is unaffected)."""
    ledger = CallLedger(ledger_path)

    # ---- call 1: weakness analysis
    weakness_prompt = prompts.render_weakness_prompt(wrong=wrong, correct=correct)
    raw_w, meta_w = _reserve_complete(
        ledger=ledger, request_id=f"pf-i{iteration}-weakness", prompt=weakness_prompt,
        provider_call=lambda: weakness_provider.request(prompt=weakness_prompt),
        trace_path=trace_path, request_type="weakness")
    parsed_w = prompts.parse_json_response(raw_w)
    if not parsed_w or not parsed_w.get("weaknesses"):
        raise AlgorithmError(f"iteration {iteration}: weakness response invalid or empty: {raw_w!r}")
    weaknesses = parsed_w["weaknesses"]
    generation_constraints = parsed_w.get("generation_constraints", [])
    avoid = parsed_w.get("avoid", [])

    # ---- call 2: ONE batch-generation call for the whole batch
    gen_prompt = prompts.render_batch_gen_prompt(
        weaknesses=weaknesses, generation_constraints=generation_constraints, avoid=avoid,
        n_easy=n_easy, n_hard=n_hard)
    raw_g, meta_g = _reserve_complete(
        ledger=ledger, request_id=f"pf-i{iteration}-batchgen", prompt=gen_prompt,
        provider_call=lambda: batch_gen_provider.request(prompt=gen_prompt),
        trace_path=trace_path, request_type="batch_generation")
    parsed_g = prompts.parse_json_response(raw_g)
    if not parsed_g or "easy_questions" not in parsed_g or "hard_questions" not in parsed_g:
        raise AlgorithmError(f"iteration {iteration}: batch-generation response invalid: {raw_g!r}")
    easy_qs, hard_qs = parsed_g["easy_questions"], parsed_g["hard_questions"]
    all_questions = [(q, "easy") for q in easy_qs] + [(q, "hard") for q in hard_qs]

    # ---- call 3: ONE batch-labeling call for the whole batch
    label_prompt = prompts.render_batch_label_prompt([q for q, _ in all_questions])
    raw_l, meta_l = _reserve_complete(
        ledger=ledger, request_id=f"pf-i{iteration}-batchlabel", prompt=label_prompt,
        provider_call=lambda: batch_label_provider.request(prompt=label_prompt),
        trace_path=trace_path, request_type="batch_labeling")
    parsed_l = prompts.parse_json_response(raw_l)
    if not parsed_l or "answers" not in parsed_l:
        raise AlgorithmError(f"iteration {iteration}: batch-labeling response invalid: {raw_l!r}")

    answers_by_index = {a["index"]: a for a in parsed_l["answers"]}
    batch: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    for idx, (question, difficulty) in enumerate(all_questions):
        answer_entry = answers_by_index.get(idx)
        if answer_entry is None:
            abstained.append({"index": idx, "question": question, "reason": "missing_from_response"})
            continue
        if answer_entry["answer_status"] == "abstained" or not answer_entry["answer"].strip():
            abstained.append({"index": idx, "question": question, "reason": "teacher_abstained"})
            continue
        batch.append({"question": question, "reference_answer": answer_entry["answer"].strip(),
                      "difficulty": difficulty})

    resolved_identify_fn = identify_fn or identify_mod.identify
    identify_result = resolved_identify_fn(student_answer_fn=student_answer_fn, batch=batch)

    return {
        "iteration": iteration, "batch": batch, "abstained": abstained,
        "wrong": identify_result["wrong"], "correct": identify_result["correct"],
        "uncertain": identify_result.get("uncertain", []),
        "weaknesses": weaknesses, "generation_constraints": generation_constraints, "avoid": avoid,
        "calls": {"weakness": meta_w, "batch_generation": meta_g, "batch_labeling": meta_l},
        "identify_diagnostics": {k: v for k, v in identify_result.items()
                                if k not in ("wrong", "correct", "uncertain")},
    }
