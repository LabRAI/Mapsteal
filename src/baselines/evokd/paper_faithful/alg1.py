"""Pure control flow of EvoKD Algorithm 1 (EvoKD.pdf p.5), no torch/network.

`step` counts TRAINING steps (one optimizer step on a batch of <= 8 rows, the
paper's student batch size), NOT chat events:

    while step < num_steps:
        step += 1
        if step % review == 0:   train on history            (Review History)
        elif step % chat == 0:   new LLM batch D^i + 1 step   (chat)
        else:                    train on D^i                 (Repeat Batch)

Note the precedence: a step divisible by both (e.g. 200 for chat=40,
review=50) is a review step; the chat event that would have fallen there is
skipped, exactly as the pseudocode reads.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

CHAT = 40
REVIEW = 50
STUDENT_BATCH = 8


def action_for_step(step: int, chat: int = CHAT, review: int = REVIEW) -> str:
    if step < 1:
        raise ValueError("steps are 1-based")
    if step % review == 0:
        return "review"
    if step % chat == 0:
        return "chat"
    return "repeat"


def plan(n_chat_events: int, chat: int = CHAT, review: int = REVIEW) -> dict[str, Any]:
    """Simulate only the control flow until ``n_chat_events`` chat events have
    happened, then let that last batch receive its ``chat - 1`` follow-up
    steps (so it is trained on as much as every earlier batch)."""
    if n_chat_events < 1:
        raise ValueError("n_chat_events must be >= 1")
    chat_steps: list[int] = []
    review_steps: list[int] = []
    skipped_chat_steps: list[int] = []
    step = 0
    while len(chat_steps) < n_chat_events:
        step += 1
        act = action_for_step(step, chat, review)
        if act == "chat":
            chat_steps.append(step)
        elif act == "review":
            review_steps.append(step)
            if step % chat == 0:
                skipped_chat_steps.append(step)
    num_steps = chat_steps[-1] + chat - 1
    review_steps = [s for s in range(1, num_steps + 1) if s % review == 0]
    skipped_chat_steps = [s for s in review_steps if s % chat == 0]
    return {"chat_steps": chat_steps, "num_steps": num_steps, "review_steps": review_steps,
            "skipped_chat_steps": skipped_chat_steps,
            "n_repeat_steps": num_steps - len(chat_steps) - len(review_steps)}


class HistoryCache:
    """The paper's "global cache": every generated sample, starting from D^0
    (`history <- {D^0}`, then `Add D^i to history`)."""

    def __init__(self, d0_rows: Sequence[dict[str, Any]]):
        self.rows: list[dict[str, Any]] = [dict(r, source_event=0) for r in d0_rows]

    def add(self, event: int, rows: Sequence[dict[str, Any]]) -> None:
        self.rows.extend(dict(r, source_event=event) for r in rows)

    def __len__(self) -> int:
        return len(self.rows)

    def sample(self, k: int, rng_label: str) -> list[dict[str, Any]]:
        """One fetched batch (uniform, without replacement, deterministic per label)."""
        take = min(k, len(self.rows))
        rng = random.Random(rng_label)
        return [self.rows[i] for i in rng.sample(range(len(self.rows)), take)]


def repeat_batch(step: int, event_index: int, current_rows: Sequence[dict[str, Any]],
                 d0_rows: Sequence[dict[str, Any]], k: int = STUDENT_BATCH) -> list[dict[str, Any]]:
    """Batch for a Repeat-Batch step: the whole current D^i (<= k rows) once i >= 1;
    before the first chat (i = 0) D^0 is larger than a batch, so a seeded
    batch of k rows is drawn from it (underspecified in the paper)."""
    if event_index >= 1:
        return [dict(r) for r in current_rows]
    rng = random.Random(f"evokd-final-v1-repeat-d0-{step}")
    take = min(k, len(d0_rows))
    return [dict(d0_rows[i]) for i in rng.sample(range(len(d0_rows)), take)]
