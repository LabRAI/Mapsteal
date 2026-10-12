"""Teacher client for the final EvoKD baseline: multi-turn (message-list)
Responses-API calls, reserve/complete/fail ledger (one-shot request IDs, no
hidden retries), FULL recording of every prompt and raw output (the earlier
pilot stored only content hashes), token accounting and a hard cost guard.

Pricing used for cost estimates: gpt-4.1 public list price ($2 / 1M input,
$8 / 1M output tokens) -- disclosed, not read from any price table in the repo.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from ..kf.ledger import CallLedger
from ..kf.schemas import ProviderResponse, TeacherRequest

MODEL = "gpt-4.1"
PROVIDER = "evokd-final-teacher"
PRICE_IN_PER_M, PRICE_OUT_PER_M = 2.0, 8.0


class BudgetExceeded(RuntimeError):
    pass


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * PRICE_IN_PER_M / 1e6 + output_tokens * PRICE_OUT_PER_M / 1e6


class TeacherChatClient:
    def __init__(self, *, ledger_path: Path, io_log_path: Path, budget_usd: float,
                 create_fn: Callable[..., Any] | None = None, model: str = MODEL,
                 timeout_seconds: float = 180, default_max_output_tokens: int = 4096,
                 prior_spend_usd: float = 0.0):
        self.ledger = CallLedger(Path(ledger_path))
        self.io_log_path = Path(io_log_path)
        self.model = model
        self.budget_usd = budget_usd
        self.default_max_output_tokens = default_max_output_tokens
        self.spend_usd = prior_spend_usd
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_calls = 0
        if create_fn is None:
            from openai import OpenAI  # lazy: only real construction touches the network library
            create_fn = OpenAI(max_retries=0, timeout=timeout_seconds).responses.create
        self._create = create_fn

    def call(self, *, request_id: str, messages: list[dict[str, str]], schema_name: str | None = None,
             schema: dict[str, Any] | None = None, max_output_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
        if self.spend_usd >= self.budget_usd:
            raise BudgetExceeded(f"spend ${self.spend_usd:.4f} already >= budget ${self.budget_usd:.2f}")
        prompt_text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        req = TeacherRequest(request_id=request_id, prompt=prompt_text,
                             content_hash=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                             provider=PROVIDER, hidden_retries=0)
        self.ledger.reserve(req)
        kwargs: dict[str, Any] = dict(model=self.model, input=messages, temperature=0,
                                      max_output_tokens=max_output_tokens or self.default_max_output_tokens, store=False)
        if schema is not None:
            kwargs["text"] = {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}}
        try:
            response = self._create(**kwargs)
        except Exception as exc:
            self.ledger.fail(req, error=f"{type(exc).__name__}: {exc}")
            raise
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        raw = response.output_text
        meta = {"provider": PROVIDER, "model": self.model, "response_id": getattr(response, "id", None),
                "max_retries": 0, "store": False, "usage": usage}
        self.ledger.complete(req, ProviderResponse(request_id, raw, meta))
        in_tok = (usage or {}).get("input_tokens") or 0
        out_tok = (usage or {}).get("output_tokens") or 0
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.n_calls += 1
        self.spend_usd += cost_usd(in_tok, out_tok)
        self.io_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.io_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "request_id": request_id, "timestamp": time.time(), "model": self.model, "temperature": 0,
                "schema_name": schema_name, "messages": messages, "output_text": raw,
                "usage": usage, "response_id": meta["response_id"],
                "call_cost_usd": cost_usd(in_tok, out_tok)}, ensure_ascii=False, sort_keys=True) + "\n")
        return raw, meta

    def prompt_provider(self, *, request_id: str, schema_name: str, schema: dict[str, Any]) -> "_PromptProvider":
        """Adapter with the `.request(prompt=...)` shape identify_v2 expects."""
        return _PromptProvider(self, request_id, schema_name, schema)

    def totals(self) -> dict[str, Any]:
        return {"n_calls": self.n_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cost_usd": cost_usd(self.input_tokens, self.output_tokens)}


class _PromptProvider:
    def __init__(self, client: TeacherChatClient, request_id: str, schema_name: str, schema: dict[str, Any]):
        self._client, self._request_id, self._schema_name, self._schema = client, request_id, schema_name, schema

    def request(self, *, prompt: str) -> tuple[str, dict[str, Any]]:
        return self._client.call(request_id=self._request_id, messages=[{"role": "user", "content": prompt}],
                                 schema_name=self._schema_name, schema=self._schema)
