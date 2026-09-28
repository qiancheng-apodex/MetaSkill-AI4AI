"""Tool, task, budget and result contracts shared with the modular runtime."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from ..token_count import approximate_token_count
@dataclass(frozen=True, slots=True)
class ToolResult:
    """One bounded adapter observation."""

    ok: bool
    observation: Any
    error_type: str | None = None
    infrastructure_failure: bool = False
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if self.terminal_reason is not None and (
            not isinstance(self.terminal_reason, str)
            or not self.terminal_reason
            or len(self.terminal_reason) > 80
        ):
            raise ValueError("terminal_reason must be a non-empty string of at most 80 characters")


ToolHandler = Callable[[Mapping[str, Any]], ToolResult]


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("tool name must be a non-empty string")
        if self.name == "finish":
            raise ValueError("finish is reserved by the fixed runner")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")

    def api_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True, slots=True)
class TargetTask:
    instance_id: str
    prompt: str
    initial_observation: Any
    public_metadata: Mapping[str, Any] = field(default_factory=dict)
    input_projection: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be non-empty")


@dataclass(frozen=True, slots=True)
class TargetBudget:
    max_steps: int
    max_tool_calls: int
    max_accounted_tokens: int
    max_completion_tokens_per_call: int
    wall_time_seconds: float
    max_observation_chars: int = 30000

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_steps,
            self.max_tool_calls,
            self.max_accounted_tokens,
            self.max_completion_tokens_per_call,
            self.max_observation_chars,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integer_fields):
            raise ValueError("all integer budget fields must be positive integers")
        if isinstance(self.wall_time_seconds, bool) or not isinstance(self.wall_time_seconds, (int, float)) or self.wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be positive")


@dataclass(frozen=True, slots=True)
class TargetRunResult:
    instance_id: str
    model: str
    harness_id: str
    harness_sha256: str
    seed: int | None
    stop_reason: str
    final_answer: str | None
    steps: int
    tool_calls: int
    model_calls: int
    accounted_tokens: int
    token_accounting_complete: bool
    tool_error_counts: Mapping[str, int]
    infrastructure_retry_count: int
    wall_time_seconds: float
    events: tuple[Mapping[str, Any], ...]
    transcript: tuple[Mapping[str, Any], ...]
    public_final_artifacts: tuple[Mapping[str, Any], ...] = ()


def _safe_json(value: Any, max_chars: int) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        rendered = json.dumps({"repr": repr(value)}, ensure_ascii=False)
    if len(rendered) <= max_chars:
        return rendered
    return json.dumps(
        {
            "truncated": True,
            "original_characters": len(rendered),
            "prefix": rendered[:max_chars],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _metadata_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _metadata_infrastructure_retry_count(value: Mapping[str, Any]) -> int:
    attempts = value.get("attempts")
    if not isinstance(attempts, (list, tuple)):
        return 0
    return sum(
        isinstance(attempt, Mapping)
        and attempt.get("retry_delay_seconds") is not None
        for attempt in attempts
    )


def _usage_total(completion: Any, messages: Sequence[Mapping[str, Any]]) -> tuple[int, bool]:
    metadata = getattr(completion, "metadata", None)
    usage = getattr(metadata, "usage", None)
    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total, True
    message = getattr(completion, "message", {})
    approximation = approximate_token_count(
        json.dumps(list(messages), ensure_ascii=False, default=repr)
        + json.dumps(message, ensure_ascii=False, default=repr)
    )
    return approximation, False


