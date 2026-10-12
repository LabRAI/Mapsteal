"""Identify v2: a two-stage correctness criterion, replacing identify.py's
exact-normalized-string match (see ../IDENTIFY_VALIDITY_AUDIT_v1.md for why
-- 76% of exact-match "wrong" items across every Identify() call made so
far were semantic false negatives).

Stage 1 (this module, pure Python, no LLM call): deterministic canonical
matching only --
  - leading/trailing punctuation, whitespace, and case normalization;
  - whole-word-span containment (the normalized reference appears as a
    complete, contiguous sequence of words inside the normalized student
    answer, or vice versa) -- catches "Prostate cancer" vs "Prostate
    cancer.", "EBV" vs "Epstein-Barr virus (EBV).", "Pancreas" vs "The
    pancreas produces insulin.".
No fuzzy/edit-distance heuristic and no learned threshold anywhere in this
stage. Short polarity tokens ("yes"/"no"/"true"/"false") are excluded from
the containment shortcut -- a long paragraph can incidentally contain the
word "no" without agreeing with it -- and always deferred to stage 2.

Stage 2 (one batched Teacher call, only for items stage 1 leaves
unresolved): a semantic judge classifies each remaining item as
semantically_correct / genuinely_incorrect / uncertain. `uncertain` items
are excluded from BOTH the wrong and correct lists -- never forced into
either bucket, and therefore never enter a downstream weakness-analysis
prompt.

This module makes NO change to identify.py (kept, unmodified, for
comparison) and is not wired into algorithm.run_chat_event -- it is used
standalone by scripts/run_identify_v2_replay_v1.py to validate the new
criterion against the same frozen batches before any acquisition code is
touched.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

SEMANTIC_JUDGE_SCHEMA_NAME = "evokd-pf-identify-v2-semantic-judge"
_POLARITY_TOKENS = {"yes", "no", "true", "false"}


def normalize_answer_v2(text: str) -> str:
    t = str(text).strip().casefold()
    t = re.sub(r"[^\w\s]", " ", t)  # punctuation -> space, never deleted outright
    return " ".join(t.split())


def _contains_span(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def stage1_match(reference_answer: str, student_answer: str) -> bool | None:
    """True if stage 1 can deterministically confirm correctness, else
    None (never False -- stage 1 only ever early-accepts; every item it
    cannot confirm is deferred to stage 2, not declared wrong)."""
    ref = normalize_answer_v2(reference_answer)
    ans = normalize_answer_v2(student_answer)
    if ref == ans:
        return True
    if ref in _POLARITY_TOKENS:
        return None
    ref_tokens, ans_tokens = ref.split(), ans.split()
    if _contains_span(ans_tokens, ref_tokens) or _contains_span(ref_tokens, ans_tokens):
        return True
    return None


def semantic_judge_response_schema(n_total: int) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["judgments"],
        "properties": {
            "judgments": {
                "type": "array", "minItems": n_total, "maxItems": n_total,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["index", "verdict"],
                    "properties": {
                        "index": {"type": "integer"},
                        "verdict": {"type": "string",
                                   "enum": ["semantically_correct", "genuinely_incorrect", "uncertain"]},
                    },
                },
            },
        },
    }


def render_semantic_judge_prompt(items: list[dict[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(items):
        lines.append(f"{i}. question: {item['question']}\n   reference_answer: {item['reference_answer']}\n   student_answer: {item['student_answer']}")
    numbered = "\n".join(lines)
    return (
        "You are judging whether a medical-QA student model's answer is "
        "semantically correct compared to a short reference answer, even "
        "if the wording, verbosity, or level of detail differs.\n\n"
        f"{numbered}\n\n"
        "For each item, decide:\n"
        "- \"semantically_correct\" if the student's answer conveys the "
        "same essential medical fact as the reference, regardless of "
        "phrasing, verbosity, or minor differences in specificity.\n"
        "- \"genuinely_incorrect\" if the student's answer is factually "
        "wrong or contradicts the reference.\n"
        "- \"uncertain\" if you cannot confidently decide either way (for "
        "example, the reference itself looks unclear, garbled, or does not "
        "appear to match the question asked).\n\n"
        "Return JSON only matching the required schema: one entry per "
        "item, using the exact 0-based index shown above."
    )


def parse_semantic_judge_response(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except Exception:
        return None


def classify_records(records: list[dict[str, Any]], *, semantic_judge_provider) -> dict[str, Any]:
    """``records``: list of {"question", "reference_answer", "student_answer", ...}
    (already-generated Student answers -- this function makes no student
    inference call, only the optional stage-2 Teacher call). Returns
    {"wrong", "correct", "uncertain", "stage1_resolved", "stage2_calls",
    "stage2_meta", "unresolved_count"}."""
    correct: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for record in records:
        verdict = stage1_match(record["reference_answer"], record["student_answer"])
        (correct if verdict is True else unresolved).append(record)

    wrong: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    stage2_calls = 0
    stage2_meta = None
    if unresolved:
        prompt = render_semantic_judge_prompt(unresolved)
        raw, meta = semantic_judge_provider.request(prompt=prompt)
        stage2_calls = 1
        stage2_meta = meta
        parsed = parse_semantic_judge_response(raw)
        judgments_by_index = {j["index"]: j for j in (parsed or {}).get("judgments", [])}
        for idx, record in enumerate(unresolved):
            verdict = judgments_by_index.get(idx, {}).get("verdict", "uncertain")
            if verdict == "semantically_correct":
                correct.append(record)
            elif verdict == "genuinely_incorrect":
                wrong.append(record)
            else:
                uncertain.append(record)

    return {
        "wrong": wrong, "correct": correct, "uncertain": uncertain,
        "stage1_resolved": len(records) - len(unresolved),
        "unresolved_count": len(unresolved),
        "stage2_calls": stage2_calls, "stage2_meta": stage2_meta,
    }


def identify_v2(*, student_answer_fn: Callable[[str], str], batch: list[dict[str, Any]],
                semantic_judge_provider) -> dict[str, Any]:
    """Same call shape as identify.identify(), for use as a drop-in in a
    real (not-yet-authorized) pilot: runs Student inference itself, then
    classify_records()."""
    records = [{**row, "student_answer": student_answer_fn(row["question"])} for row in batch]
    return classify_records(records, semantic_judge_provider=semantic_judge_provider)
