"""LLM adapters behind one interface. `fake` is the default and needs no network."""

from __future__ import annotations

from compliance_judge.config import Settings
from compliance_judge.llm.base import LLMClient, LLMError, LLMRefusal, LLMRequest, LLMResponse
from compliance_judge.llm.fake import FakeLLM

__all__ = ["LLMClient", "LLMError", "LLMRefusal", "LLMRequest", "LLMResponse", "make_client"]


def make_client(settings: Settings) -> LLMClient:
    if settings.provider == "fake":
        return FakeLLM(quote_drift=settings.fake_quote_drift)
    if settings.provider == "anthropic":
        from compliance_judge.llm.anthropic_adapter import AnthropicLLM

        return AnthropicLLM(effort=settings.effort)
    if settings.provider == "openai":
        from compliance_judge.llm.openai_adapter import OpenAILLM

        return OpenAILLM()
    raise ValueError(f"unknown JUDGE_LLM_PROVIDER '{settings.provider}' (fake|anthropic|openai)")
