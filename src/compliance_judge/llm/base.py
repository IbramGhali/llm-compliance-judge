"""Provider-agnostic LLM interface.

The pipeline only sees `LLMClient`. A request carries the rendered prompt (what a real
model reads) and the structured payload it was rendered from (what the deterministic fake
adapter reads), so tests and offline evaluation exercise the exact same pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Task = Literal["extraction", "assessment"]


@dataclass(frozen=True)
class LLMRequest:
    task: Task
    model: str
    system: str
    user: str
    json_schema: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 16000


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMError(RuntimeError):
    """The provider failed or returned something unusable. The pipeline fails safe."""


class LLMRefusal(LLMError):
    """The provider declined the request."""


class LLMClient(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token) for offline cost reporting."""
    return max(1, len(text) // 4)
