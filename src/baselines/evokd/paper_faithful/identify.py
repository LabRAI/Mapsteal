"""Identify(model, D) -- Algorithm 1 lines 2/17: evaluate a Student
checkpoint on a batch and split it into wrong/correct.

Correctness criterion (disclosed adaptation, DESIGN_v1.md section 9): the
paper's classifier setting uses exact predicted-label == gold-label (with a
0.95 confidence threshold to hand-pick the clearest wrong/correct cases --
Section 4.1). Our Student is a decoder LM producing free text, not a
calibrated classifier, so no analogous confidence score exists (the SAME
limitation the prior ChemProt reproduction's evaluate_student.py module
already documented: "confidence is always null -- the production inference
path exposes no calibrated probability"). The smallest transparent
adaptation, reusing a technique ALREADY used elsewhere in this exact
codebase for the identical problem (run_manifest.py's own
`pre_update_eval` step): exact, casefold+whitespace-normalized string
match between the Student's free-text answer and the reference answer.
No semantic-similarity judge, no LLM-as-judge, no partial credit --
deliberately the simplest possible binary criterion, not a complex
uncertainty model.
"""
from __future__ import annotations

from typing import Any, Callable


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def is_correct(student_answer: str, reference_answer: str) -> bool:
    return normalize_answer(student_answer) == normalize_answer(reference_answer)


def identify(*, student_answer_fn: Callable[[str], str], batch: list[dict[str, Any]]) -> dict[str, list[dict]]:
    """``batch``: list of {"question": str, "reference_answer": str, ...}.
    Returns {"wrong": [...], "correct": [...]}, each entry augmented with
    the Student's own raw output (`student_answer`) for downstream display
    in the next weakness-analysis prompt."""
    wrong, correct = [], []
    for row in batch:
        raw = student_answer_fn(row["question"])
        record = {**row, "student_answer": raw}
        (correct if is_correct(raw, row["reference_answer"]) else wrong).append(record)
    return {"wrong": wrong, "correct": correct}
