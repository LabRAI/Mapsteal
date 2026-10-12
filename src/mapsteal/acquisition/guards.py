"""Small deterministic contracts; no model calls or semantic inference.

These helpers distinguish validated store structure from semantic information
novelty.  They deliberately do not select, rank, deduplicate, or repair
acquisition requests.
"""
from __future__ import annotations

import hashlib
import json
import re


OPEN_ENDED_QUERY_CONTRACT = "open_ended_acquisition_query_v1"
JOINT_OBJECT_CONTRACT = "evidence_supported_joint_object_v1"

# These constructions conventionally refer to omitted answer options.  The
# interface has none, so fail closed rather than inventing options or retrying.
OPTION_DEPENDENT_PATTERNS = (
    r"\bwhich\s+of\s+the\s+following\b",
    r"\ball\s+of\s+the\s+following\b",
    r"\bwhich\s+of\s+these\b",
    r"\bexcept\b",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_open_ended_query(question: str) -> tuple[bool, str | None]:
    """Reject only explicit option-dependent forms; no semantic repair."""
    normalized = " ".join(question.lower().split())
    if not normalized:
        return False, "empty_question"
    for pattern in OPTION_DEPENDENT_PATTERNS:
        if re.search(pattern, normalized):
            return False, f"option_dependent_form:{pattern}"
    return True, None


def empty_map_state() -> dict:
    """A single-topic Map state with a passive, nonselectable joint store."""
    return {"nodes": {}, "object_facets": [], "joint_objects": []}


def attach_supported_object_facet(state: dict, topic: str, facet: str, *, provenance: str) -> bool:
    """Attach exactly one updater-supported ObjectFacet; never cross-join tags."""
    if not topic or not facet or not provenance:
        raise ValueError("topic, facet, and provenance are required")
    node = state["nodes"].setdefault(topic, {"canonical_topic": topic, "tags": [], "provenance": []})
    if provenance not in node["provenance"]:
        node["provenance"].append(provenance)
    if facet in node["tags"]:
        return False
    node["tags"].append(facet)
    state["object_facets"].append((topic, facet))
    return True


def create_evidence_supported_joint_object(state: dict, member_topics: list[str], *,
                                           supporting_evidence_ids: list[str], provenance: str) -> dict:
    """Materialize a passive joint only when explicit external evidence is cited.

    The caller/updater is responsible for claiming explicit support from the
    immutable receipt. This function enforces only structural provenance: a
    joint cannot be created from a topic pair alone, nor can it carry facets.
    """
    members = tuple(sorted(set(member_topics)))
    evidence = tuple(sorted(set(supporting_evidence_ids)))
    if len(members) < 2:
        raise ValueError("joint object requires at least two distinct member topics")
    if not evidence:
        raise ValueError("joint object requires supporting external evidence IDs")
    if not provenance:
        raise ValueError("joint object requires provenance")
    object_id = stable_id({"contract": JOINT_OBJECT_CONTRACT, "member_topics": members,
                           "supporting_evidence_ids": evidence})
    for existing in state["joint_objects"]:
        if existing["object_id"] == object_id:
            return existing
    joint = {"object_id": object_id, "member_topics": list(members),
             "supporting_evidence_ids": list(evidence), "provenance": provenance}
    state["joint_objects"].append(joint)
    return joint


def selectable_coordinates(state: dict) -> list[tuple[str, str]]:
    """Selection reads only explicitly materialized ObjectFacets.

    Passive joint objects have no facets under v1 and therefore cannot become
    selectable merely by existing.
    """
    return list(state["object_facets"])
