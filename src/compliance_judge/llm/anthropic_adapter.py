"""Claude adapter (optional). Install with `uv sync --extra anthropic`.

Uses structured outputs (`output_config.format`) so responses are schema-valid JSON, and the
server-side refusal fallback on models that support it. Any failure raises `LLMError`; the
pipeline turns that into `unclear` verdicts rather than guessing.
"""

from __future__ import annotations

import time
from typing import Any

from compliance_judge.llm.base import LLMError, LLMRefusal, LLMRequest, LLMResponse

# Models that accept the server-side `fallbacks: "default"` refusal routing.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, effort: str = "medium", timeout: float = 120.0) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMError("install the anthropic extra: uv sync --extra anthropic") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=2)
        self.effort = effort

    def complete(self, request: LLMRequest) -> LLMResponse:
        anthropic = self._anthropic
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": request.json_schema},
            },
        }
        started = time.perf_counter()
        try:
            if request.model in FALLBACK_MODELS:
                response = self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                response = self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("connection error") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if response.stop_reason == "refusal":
            raise LLMRefusal("model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("no text block in response")
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
