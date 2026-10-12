"""Frozen EvoKD-paper-faithful prompts (Algorithm 1 / Section 3.1): exactly
three request types -- weakness analysis, ONE batch generation call, ONE
batch labeling call. No per-item/per-root loop anywhere in this module.

Deliberately absent from this module: any duplicate check, any word-count
rule, any "must start with X" rule, any single-sentence rule, any numeric
filter, any admission/rejection step of any kind. Per user instruction,
none of studies/evokd_chemprot_reproduction_v1's filter machinery
(direct_student_questioning_v2.normalize_and_validate_question,
rejects_paper_local_numeric) is imported or reimplemented here.
"""
from __future__ import annotations

import json
from typing import Any

WEAKNESS_SCHEMA_NAME = "evokd-pf-weakness-v1"
BATCH_GEN_SCHEMA_NAME = "evokd-pf-batch-gen-v1"
BATCH_LABEL_SCHEMA_NAME = "evokd-pf-batch-label-v1"

WEAKNESS_RESPONSE_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["weaknesses", "generation_constraints", "avoid"],
    "properties": {
        "weaknesses": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "generation_constraints": {"type": "array", "items": {"type": "string"}},
        "avoid": {"type": "array", "items": {"type": "string"}},
    },
}


def _bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


def _render_examples(records: list[dict[str, Any]], *, include_gold: bool) -> str:
    if not records:
        return "(none)"
    lines = []
    for i, r in enumerate(records, 1):
        if include_gold:
            lines.append(f"{i}. question: {r['question']}\n   reference_answer: {r['reference_answer']}\n   student_answer: {r['student_answer']}")
        else:
            lines.append(f"{i}. question: {r['question']}\n   student_answer: {r['student_answer']}")
    return "\n".join(lines)


def render_weakness_prompt(*, wrong: list[dict[str, Any]], correct: list[dict[str, Any]]) -> str:
    return (
        "You are analyzing a medical-knowledge QA student model's performance.\n\n"
        "Below are examples the student answered incorrectly:\n"
        f"{_render_examples(wrong, include_gold=True)}\n\n"
        "Below are examples the student answered correctly:\n"
        f"{_render_examples(correct, include_gold=False)}\n\n"
        "Infer the concrete medical knowledge, reasoning pattern, or question type "
        "associated with the student's failures. Do not invent a weakness that is "
        "unsupported by the examples. Return JSON only matching the required schema."
    )


def batch_gen_response_schema(n_easy: int, n_hard: int) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["easy_questions", "hard_questions"],
        "properties": {
            "easy_questions": {"type": "array", "items": {"type": "string"}, "minItems": n_easy, "maxItems": n_easy},
            "hard_questions": {"type": "array", "items": {"type": "string"}, "minItems": n_hard, "maxItems": n_hard},
        },
    }


def render_batch_gen_prompt(*, weaknesses: list[str], generation_constraints: list[str],
                            avoid: list[str], n_easy: int, n_hard: int) -> str:
    return (
        "You are generating new medical-knowledge questions to teach a student model, "
        "based on a weakness analysis.\n\n"
        f"Diagnosed weaknesses:\n{_bullet(weaknesses)}\n\n"
        f"Generation constraints:\n{_bullet(generation_constraints)}\n\n"
        f"Avoid:\n{_bullet(avoid)}\n\n"
        f"Generate exactly {n_easy} EASY questions and exactly {n_hard} HARD questions "
        "in a single batch.\n"
        "- EASY questions should cover related medical knowledge without the diagnosed "
        "failure pattern (this prevents catastrophic forgetting of what the student "
        "already knows).\n"
        "- HARD questions should specifically target the diagnosed weakness.\n\n"
        "Each question must be self-contained, answerable from medical knowledge, and "
        "in English. Do not provide answers or explanations. Return JSON only matching "
        "the required schema."
    )


def batch_label_response_schema(n_total: int) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["answers"],
        "properties": {
            "answers": {
                "type": "array", "minItems": n_total, "maxItems": n_total,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["index", "answer_status", "answer"],
                    "properties": {
                        "index": {"type": "integer"},
                        "answer_status": {"type": "string", "enum": ["answered", "abstained"]},
                        "answer": {"type": "string"},
                    },
                },
            },
        },
    }


def render_batch_label_prompt(questions: list[str]) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
    return (
        "Answer each of the following cybersecurity questions accurately and concisely.\n\n"
        f"{numbered}\n\n"
        "Return JSON only matching the required schema: one entry per question, using "
        "the exact 0-based index shown above. If a question cannot be answered, set "
        "answer_status to \"abstained\" and answer to an empty string."
    )


def parse_json_response(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except Exception:
        return None
