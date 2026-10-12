"""B100 pilot (13 chat events, batch_size=8, ~104 generated examples)
using Identify v2 + validated adapter continuation. Core, testable
orchestration logic lives here; scripts/run_pilot_b100_v2.py is the thin
CLI wrapper (argparse + single-GPU bootstrap gate + real provider/backend
wiring).

Does not modify identify.py, identify_v2.py's matching rules, the
semantic-judge prompt, prompts.py's three acquisition prompts, or
training.py's fresh-LoRA default path. Every event MUST use adapter
continuation (init_adapter_dir=<previous event's real adapter>) --
Algorithm 1 line 18 read literally: model^i <- train(model^{i-1}, D^i).
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Callable

from . import algorithm, identify_v2


class PilotStopped(RuntimeError):
    """Raised for every hard-stop condition. Never caught silently --
    scripts/run_pilot_b100_v2.py lets this propagate to a non-zero exit."""


class _NeverCalledProvider:
    """Placeholder passed to classify_records() when stage 1 resolves
    everything -- .request() must never actually be invoked in that case."""
    def request(self, *, prompt):
        raise AssertionError("semantic judge called with zero unresolved items -- should not happen")


def build_identify_fn_with_diagnostics(judge_provider_factory: Callable[[int], Any]) -> tuple[Callable, dict]:
    """Wraps identify_v2.identify_v2 so the caller can also recover, per
    record, whether stage 1 or stage 2 resolved it (identify_v2's own
    return value only has the three final buckets) -- needed to persist
    "Stage-1 decisions, Stage-2 judge outputs" per event, without touching
    identify_v2.py's validated internals.

    ``judge_provider_factory(n_unresolved) -> provider``: the semantic-
    judge provider's structured-output schema is sized to the EXACT number
    of items in its prompt (identify_v2.semantic_judge_response_schema's
    minItems==maxItems==n_total), which is only known after stage 1 runs
    -- so the provider must be built lazily, per call, not passed in
    pre-constructed for a guessed size."""
    diagnostics: dict[str, Any] = {}

    def identify_fn(*, student_answer_fn: Callable[[str], str], batch: list[dict[str, Any]]) -> dict[str, Any]:
        records = [{**row, "student_answer": student_answer_fn(row["question"])} for row in batch]
        per_item_stage1 = [
            {"question": r["question"], "stage1_verdict": identify_v2.stage1_match(r["reference_answer"], r["student_answer"])}
            for r in records
        ]
        n_unresolved = sum(1 for item in per_item_stage1 if item["stage1_verdict"] is None)
        semantic_judge_provider = judge_provider_factory(n_unresolved) if n_unresolved else _NeverCalledProvider()
        result = identify_v2.classify_records(records, semantic_judge_provider=semantic_judge_provider)
        unresolved_count = result["unresolved_count"]
        diagnostics.update({
            "per_item_stage1": per_item_stage1,
            "stage1_resolved": result["stage1_resolved"],
            "unresolved_count": unresolved_count,
            "stage2_calls": result["stage2_calls"],
            "stage2_usage": (result.get("stage2_meta") or {}).get("usage"),
            "_all_uncertain_this_event": bool(unresolved_count) and len(result["uncertain"]) == unresolved_count,
        })
        return result

    return identify_fn, diagnostics


def check_wrong_list_not_empty(wrong: list[dict[str, Any]], *, context: str) -> None:
    """Requirement 8: never let an empty wrong list silently reach a
    weakness-analysis call (the frozen WEAKNESS_RESPONSE_JSON_SCHEMA
    requires minItems=1 on "weaknesses", which would force the Teacher to
    fabricate one). Stop and report instead of inventing a fallback."""
    if not wrong:
        raise PilotStopped(
            f"{context}: wrong list is empty -- the current implementation/spec has no "
            "defined behavior for a weakness-analysis call with zero wrong examples "
            "(WEAKNESS_RESPONSE_JSON_SCHEMA requires at least one weakness, which would "
            "force fabrication). Stopping per instruction rather than inventing one.")


def verify_adapter_continuation(manifest: dict[str, Any], *, expected_prev_adapter_dir: Path) -> None:
    """Hard-stop check: adapter continuation provenance must be present,
    correct, and byte-verified against the real prior adapter on disk."""
    if manifest.get("initialization") != "continued_from_frozen_adapter":
        raise PilotStopped(
            f"adapter continuation broke: manifest.initialization={manifest.get('initialization')!r}, "
            "expected 'continued_from_frozen_adapter' -- refusing to silently accept a fresh-LoRA fallback")
    prov = manifest.get("init_adapter_provenance")
    if not prov:
        raise PilotStopped("adapter continuation broke: manifest has no init_adapter_provenance block")
    if Path(prov["init_adapter_dir"]).resolve() != expected_prev_adapter_dir.resolve():
        raise PilotStopped(
            f"adapter continuation broke: provenance points at {prov['init_adapter_dir']!r}, "
            f"expected {expected_prev_adapter_dir}")
    import hashlib
    actual_hash = hashlib.sha256((expected_prev_adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()
    if prov["init_adapter_weights_sha256"] != actual_hash:
        raise PilotStopped("adapter continuation broke: provenance weight hash does not match the real prior adapter file on disk")


def check_semantic_judge_not_systematically_failing(*, diagnostics: dict[str, Any],
                                                     consecutive_all_uncertain: int) -> int:
    """Hard-stop condition: if the semantic judge's output cannot be parsed
    at all, every unresolved item silently defaults to "uncertain"
    (identify_v2.classify_records' own fail-safe). That is safe for one
    occasional occurrence, but if EVERY unresolved item is uncertain for
    two events running, treat it as the judge systematically failing
    (e.g. malformed JSON on every call) rather than a real ambiguity
    pattern, and stop. Returns the updated consecutive-failure counter."""
    unresolved = diagnostics.get("unresolved_count", 0)
    if diagnostics.get("stage2_calls", 0) == 0 or unresolved == 0:
        return 0  # nothing was deferred to the judge this event
    all_uncertain_this_event = diagnostics.get("_all_uncertain_this_event", False)
    if all_uncertain_this_event:
        consecutive_all_uncertain += 1
    else:
        consecutive_all_uncertain = 0
    if consecutive_all_uncertain >= 2:
        raise PilotStopped(
            "semantic judge appears to be systematically failing: every stage-1-unresolved "
            "item was classified 'uncertain' for 2 consecutive events (consistent with "
            "repeated unparseable judge output, not genuine ambiguity)")
    return consecutive_all_uncertain


def classify_weakness_novelty(current_weaknesses: list[str], seen_weaknesses: set[str]) -> dict[str, list[str]]:
    """Simple, disclosed-as-heuristic bookkeeping (case-insensitive,
    whitespace-normalized exact match; no semantic clustering) for
    "repeated vs newly-appearing" weakness themes across events."""
    def norm(s: str) -> str:
        return " ".join(s.strip().casefold().split())
    normed_current = {norm(w): w for w in current_weaknesses}
    repeated = [w for key, w in normed_current.items() if key in seen_weaknesses]
    new = [w for key, w in normed_current.items() if key not in seen_weaknesses]
    return {"repeated": repeated, "new": new, "normed_keys": list(normed_current.keys())}
