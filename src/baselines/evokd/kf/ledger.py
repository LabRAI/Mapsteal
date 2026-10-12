"""Append-only call ledger with conservative recovery semantics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import ProviderResponse, TeacherRequest


class AmbiguousReservationError(RuntimeError):
    pass


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class CallLedger:
    path: Path

    def events(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path)

    def _events_for(self, request_id: str) -> list[dict[str, Any]]:
        return [event for event in self.events() if event["request_id"] == request_id]

    def request_count(self) -> int:
        return sum(event["status"] == "reserved" for event in self.events())

    def reserve(self, request: TeacherRequest) -> None:
        if self._events_for(request.request_id):
            raise ValueError(f"request already present: {request.request_id}")
        append_jsonl(self.path, {"request_id": request.request_id, "content_hash": request.content_hash,
                                 "status": "reserved", "provider": request.provider,
                                 "hidden_retries": request.hidden_retries})

    def complete(self, request: TeacherRequest, response: ProviderResponse) -> None:
        self._require_open(request)
        append_jsonl(self.path, {"request_id": request.request_id, "content_hash": request.content_hash,
                                 "status": "completed", "raw_response": response.raw_response,
                                 "provider_metadata": response.provider_metadata})

    def fail(self, request: TeacherRequest, error: str) -> None:
        self._require_open(request)
        append_jsonl(self.path, {"request_id": request.request_id, "content_hash": request.content_hash,
                                 "status": "failed", "error": error})

    def _require_open(self, request: TeacherRequest) -> None:
        events = self._events_for(request.request_id)
        if len(events) != 1 or events[0].get("status") != "reserved" or events[0].get("content_hash") != request.content_hash:
            raise AmbiguousReservationError(f"request is not one matching open reservation: {request.request_id}")

    def recover(self, request: TeacherRequest) -> ProviderResponse | None:
        events = self._events_for(request.request_id)
        if not events:
            return None
        if any(event.get("content_hash") != request.content_hash for event in events):
            raise AmbiguousReservationError("request content hash mismatch")
        statuses = [event.get("status") for event in events]
        if statuses == ["reserved", "completed"]:
            event = events[-1]
            return ProviderResponse(request.request_id, event["raw_response"], event.get("provider_metadata", {}))
        if statuses == ["reserved"]:
            raise AmbiguousReservationError("reservation has ambiguous external-call status")
        if statuses == ["reserved", "failed"]:
            raise AmbiguousReservationError("failed call is spent and cannot be retried")
        raise AmbiguousReservationError(f"invalid ledger history: {statuses}")

