"""OpenAI adapter (optional). Install with `uv sync --extra openai`.

Set JUDGE_EXTRACTION_MODEL and JUDGE_ASSESSMENT_MODEL to OpenAI model ids when using it.
"""

from __future__ import annotations

import time

from compliance_judge.llm.base import LLMError, LLMRefusal, LLMRequest, LLMResponse


class OpenAILLM:
    name = "openai"

    def __init__(self, timeout: float = 120.0) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMError("install the openai extra: uv sync --extra openai") from exc
        self._openai = openai
        self._client = openai.OpenAI(timeout=timeout, max_retries=2)

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.task,
                        "schema": request.json_schema,
                        "strict": True,
                    },
                },
            )
        except self._openai.OpenAIError as exc:
            raise LLMError(f"OpenAI error: {exc}") from exc
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMRefusal("model declined the request")
        if not choice.message.content:
            raise LLMError("empty response")
        usage = response.usage
        return LLMResponse(
            text=choice.message.content,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
