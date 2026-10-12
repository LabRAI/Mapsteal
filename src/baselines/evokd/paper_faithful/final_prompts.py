"""Prompts for the final paper-faithful EvoKD baseline. Structure follows
EvoKD.pdf Figure 5 (sentiment example) adapted to medical QA:

  conversation A (weakness -> generation, multi-turn, same conversation):
    turn 1  user: task + INCORRECT list + CORRECT list + "summarize the patterns
            of questions that are prone to be incorrectly answered"
    turn 1  assistant: free-text pattern string (the "weakness")
    turn 2  user: "Based on your summary, please assume n_hard questions that
            the model would answer incorrectly and n_easy that it would answer
            correctly ... only reply with the questions"
  conversation B (labeling, NEW conversation): questions only, no student info.

Implementation choices (format only, not content): turn 2 and the labeling
turn use JSON-schema structured output for reliable parsing instead of the
paper's plain-text list format. No de-duplication, length, form or content
filter exists anywhere in this module or its callers (the paper has none).
The extra `generation_constraints` / `avoid` fields and the "abstain" option
of the earlier v1 pilot were our additions and are NOT used here.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

TASK = "There is a language model that answers medical knowledge questions with a short, direct answer."
WEAKNESS_TURN_MAX_OUTPUT_TOKENS = 1024


def _render_items(records: Sequence[dict[str, Any]]) -> str:
    if not records:
        return "(none)"
    lines = []
    for i, r in enumerate(records, 1):
        lines.append(f'{i}. "question": {json.dumps(r["question"], ensure_ascii=False)}; '
                     f'"reference answer": {json.dumps(r["reference_answer"], ensure_ascii=False)}; '
                     f'"model answer": {json.dumps(r["student_answer"], ensure_ascii=False)}')
    return "\n".join(lines)


def weakness_user_prompt(wrong: Sequence[dict[str, Any]], correct: Sequence[dict[str, Any]]) -> str:
    return (
        f"{TASK} I have some questions that the model answered correctly, and some other "
        "questions that the model answered incorrectly.\n"
        "The model incorrectly answered:\n"
        f"{_render_items(wrong)}\n"
        "The model correctly answered:\n"
        f"{_render_items(correct)}\n"
        "I want you to summarize the patterns of questions that are prone to be incorrectly "
        "answered by the model, based on the questions I provided."
    )


def generation_user_prompt(n_hard: int, n_easy: int) -> str:
    return (
        f"Based on your summary, please assume {n_hard} medical questions that the model would "
        f"answer incorrectly and {n_easy} medical questions that the model would answer correctly.\n"
        "Each question must be self-contained and have a single, clearly correct answer.\n"
        "You should only reply with the questions. Don't explain the answers or other information."
    )


def generation_schema(n_hard: int, n_easy: int) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["incorrect_questions", "correct_questions"],
        "properties": {
            "incorrect_questions": {"type": "array", "items": {"type": "string"},
                                    "minItems": n_hard, "maxItems": n_hard},
            "correct_questions": {"type": "array", "items": {"type": "string"},
                                  "minItems": n_easy, "maxItems": n_easy},
        },
    }


def conversation_for_generation(weakness_prompt: str, weakness_text: str, n_hard: int, n_easy: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": weakness_prompt},
        {"role": "assistant", "content": weakness_text},
        {"role": "user", "content": generation_user_prompt(n_hard, n_easy)},
    ]


def labeling_user_prompt(questions: Sequence[str]) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return (
        "I have some medical questions:\n"
        f"{numbered}\n"
        "I want you to answer each question accurately and concisely, with a short, direct answer. "
        "Don't explain other things."
    )


def labeling_schema(n: int) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False, "required": ["answers"],
        "properties": {"answers": {"type": "array", "items": {"type": "string"},
                                   "minItems": n, "maxItems": n}},
    }
