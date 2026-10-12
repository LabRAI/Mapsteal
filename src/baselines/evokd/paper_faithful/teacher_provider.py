"""OpenAIDirectV3Provider-shaped Teacher client. Copied (not cross-imported)
per this repo's own convention (see evokd_chemprot_reproduction_v1/evokd/
teacher_provider.py, which does the same thing) -- generalized to the three
schemas this study needs (weakness / batch-generation / batch-labeling).

Hard rule: the real, network-calling class here must never be imported by
any dry-run/design-only code path. Its constructor lazily imports `openai`.
"""
from __future__ import annotations

from typing import Any

MODEL = "gpt-4.1"
PROVIDER_NAME = "openai-responses"


def provider_parameters(*, timeout_seconds: float, max_output_tokens: int,
                        structured_output_schema: str) -> dict[str, Any]:
    if timeout_seconds <= 0 or max_output_tokens <= 0:
        raise ValueError("timeout and output limit must be positive")
    return {
        "max_retries": 0, "timeout_seconds": timeout_seconds,
        "max_output_tokens": max_output_tokens, "temperature": 0, "store": False,
        "structured_output_schema": structured_output_schema,
    }


class PaperFaithfulTeacherProvider:
    name, model, max_retries = PROVIDER_NAME, MODEL, 0

    def __init__(self, *, structured_output_schema: str, response_json_schema: dict[str, Any],
                timeout_seconds: float = 180, max_output_tokens: int = 4096) -> None:
        self.structured_output_schema = structured_output_schema
        self.response_json_schema = response_json_schema
        self.parameters = provider_parameters(
            timeout_seconds=timeout_seconds, max_output_tokens=max_output_tokens,
            structured_output_schema=structured_output_schema)
        from openai import OpenAI  # lazy: only touched on real construction
        self._client = OpenAI(max_retries=0, timeout=timeout_seconds)

    def request(self, *, prompt: str) -> tuple[str, dict[str, Any]]:
        response = self._client.responses.create(
            model=self.model, input=prompt, temperature=0,
            max_output_tokens=self.parameters["max_output_tokens"], store=False,
            text={"format": {"type": "json_schema", "name": self.structured_output_schema,
                             "schema": self.response_json_schema, "strict": True}},
        )
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return response.output_text, {
            "provider": self.name, "model": self.model, "response_id": response.id,
            "max_retries": 0, "store": False, "usage": usage,
        }


def usage_to_fields(usage: dict[str, Any] | None) -> dict[str, int | None]:
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
