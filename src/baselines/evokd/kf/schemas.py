"""Strict records crossing scientific or persistence boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, TypeVar

from . import SCHEMA_VERSION

SOURCE_FAMILIES = ("medmcqa", "pubmedqa", "bioasq", "chemprot", "evidence_inference")
GAP_TYPES = ("factual_gap", "relation_gap", "mechanism_gap", "evidence_gap", "boundary_or_counterexample_gap")
GAP_ORIGINS = ("mock", "human_curated_seed", "student_observation")
TARGET_TYPES = ("fact", "relation", "mechanism", "evidence", "boundary")
T = TypeVar("T", bound="StrictRecord")


class SchemaError(ValueError):
    pass


class StrictRecord:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], value: dict[str, Any]) -> T:
        if not isinstance(value, dict):
            raise SchemaError(f"{cls.__name__} must be an object")
        extra = value.keys() - {item.name for item in fields(cls)}
        if extra:
            raise SchemaError(f"{cls.__name__} extra fields: {sorted(extra)}")
        try:
            return cls(**value)
        except TypeError as exc:
            raise SchemaError(str(exc)) from exc


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be non-empty text")


@dataclass(frozen=True)
class KnowledgeTarget(StrictRecord):
    target_id: str
    target_type: str
    subject: str
    relation_or_property: str
    unknown_slot: str
    scope_or_boundary: str

    def __post_init__(self) -> None:
        _text("target_id", self.target_id)
        if self.target_type not in TARGET_TYPES:
            raise SchemaError(f"unsupported target_type: {self.target_type}")
        _text("subject", self.subject)
        _text("unknown_slot", self.unknown_slot)
        for name in ("relation_or_property", "scope_or_boundary"):
            if not isinstance(getattr(self, name), str):
                raise SchemaError(f"{name} must be text, optionally empty")


@dataclass(frozen=True)
class GapRecord(StrictRecord):
    schema_version: str
    gap_id: str
    realization: int
    root_id: str
    parent_row_id: str
    gap_origin: str
    observation_ids: list[str]
    gap_type: str
    knowledge_target: KnowledgeTarget
    known_facts: list[str]
    missing_information: str
    why_unresolved: str
    source_family: str
    provenance_hashes: list[str]
    student_state_evidence: dict[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported gap schema_version")
        for name in ("gap_id", "root_id", "parent_row_id", "missing_information", "why_unresolved", "created_at"):
            _text(name, getattr(self, name))
        if self.gap_origin not in GAP_ORIGINS:
            raise SchemaError(f"unsupported gap_origin: {self.gap_origin}")
        if not self.observation_ids or any(not str(value).strip() for value in self.observation_ids):
            raise SchemaError("gap requires observation/provenance IDs")
        if self.gap_type not in GAP_TYPES:
            raise SchemaError(f"unsupported gap_type: {self.gap_type}")
        if self.source_family not in SOURCE_FAMILIES:
            raise SchemaError(f"unsupported source_family: {self.source_family}")
        if not self.known_facts or not self.provenance_hashes:
            raise SchemaError("gap requires known facts and provenance hashes")
        if self.missing_information.strip().lower() in {fact.strip().lower() for fact in self.known_facts}:
            raise SchemaError("known fact cannot equal missing information")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GapRecord":
        converted = dict(value)
        converted["knowledge_target"] = KnowledgeTarget.from_dict(value["knowledge_target"])
        extra = converted.keys() - {item.name for item in fields(cls)}
        if extra:
            raise SchemaError(f"GapRecord extra fields: {sorted(extra)}")
        try:
            return cls(**converted)
        except TypeError as exc:
            raise SchemaError(str(exc)) from exc


@dataclass(frozen=True)
class CandidateProposal(StrictRecord):
    question: str
    target_id: str
    target: KnowledgeTarget

    def __post_init__(self) -> None:
        _text("question", self.question)
        _text("target_id", self.target_id)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateProposal":
        if not isinstance(value, dict):
            raise SchemaError("CandidateProposal must be an object")
        extra = value.keys() - {"question", "target_id", "target"}
        if extra:
            raise SchemaError(f"CandidateProposal extra fields: {sorted(extra)}")
        try:
            target = dict(value["target"])
            target.setdefault("target_id", value["target_id"])
            return cls(value["question"], value["target_id"], KnowledgeTarget.from_dict(target))
        except (KeyError, TypeError) as exc:
            raise SchemaError(str(exc)) from exc


@dataclass
class QueryCandidate(StrictRecord):
    selection_id: str
    candidate_index: int
    raw_generation: str
    parsed_question: str
    declared_target_id: str
    raw_target_output: dict[str, Any]
    gap_id: str
    generation_seed: int
    hard_rejections: list[str] = field(default_factory=list)
    admission_evidence: dict[str, Any] = field(default_factory=dict)
    realization_preference: dict[str, Any] = field(default_factory=dict)
    tie_break: dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    selected: bool = False
    selection_kind: str = "question_realization_not_utility"


@dataclass(frozen=True)
class ProposalResult(StrictRecord):
    provider_kind: str
    raw_model_response: str
    parsed_candidates: list[CandidateProposal]
    parse_failures: list[str]
    attempt_count: int = 1
    hidden_retries: int = 0


@dataclass(frozen=True)
class RendererOutcome(StrictRecord):
    status: Literal["rendered", "renderer_rejected", "teacher_abstained"]
    reason: str
    item: "RenderedItem | None"


@dataclass(frozen=True)
class SupportingFact(StrictRecord):
    fact_id: str
    subject: str
    relation: str
    object: str
    polarity: Literal["positive", "negative", "qualified"]
    certainty: str

    def __post_init__(self) -> None:
        for name in ("fact_id", "subject", "relation", "object", "certainty"):
            _text(name, getattr(self, name))


@dataclass(frozen=True)
class TeacherKnowledge(StrictRecord):
    schema_version: str
    direct_answer: str
    supporting_facts: list[SupportingFact]
    explanation: str
    uncertainty: str
    limitations: list[str]
    source_scope: str
    abstain: bool
    abstain_reason: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("unsupported knowledge schema_version")
        if self.abstain:
            _text("abstain_reason", self.abstain_reason)
        else:
            _text("direct_answer", self.direct_answer)
            if not self.supporting_facts:
                raise SchemaError("non-abstained knowledge requires supporting facts")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherKnowledge":
        if not isinstance(value, dict):
            raise SchemaError("TeacherKnowledge must be an object")
        extra = value.keys() - {item.name for item in fields(cls)}
        if extra:
            raise SchemaError(f"TeacherKnowledge extra fields: {sorted(extra)}")
        converted = dict(value)
        converted["supporting_facts"] = [SupportingFact.from_dict(item) for item in value.get("supporting_facts", [])]
        try:
            return cls(**converted)
        except TypeError as exc:
            raise SchemaError(str(exc)) from exc


@dataclass(frozen=True)
class TeacherRequest(StrictRecord):
    request_id: str
    prompt: str
    content_hash: str
    provider: str
    hidden_retries: int


@dataclass(frozen=True)
class ProviderResponse(StrictRecord):
    request_id: str
    raw_response: str
    provider_metadata: dict[str, Any]


@dataclass
class RenderedItem(StrictRecord):
    source_family: str
    canonical_input: str
    canonical_output: str
    knowledge_object_id: str
    supporting_fact_ids: list[str]
    renderer_version: str
    internal_status: Literal["accepted", "unknown", "rejected"]
    verifier_results: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Verification(StrictRecord):
    name: str
    version: str
    passed: bool
    reason_codes: list[str]
    evidence: dict[str, Any]
