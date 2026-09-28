"""Compile a modular bundle against its adapter's fixed primitive capabilities."""

from __future__ import annotations

from collections.abc import Collection

from ..harness.compiler import CompiledHarness, FIXED_RUNNER_BOUNDARY
from .runtime import ModularSession, WORKBENCH_NAMES
from .schema import HarnessBundle


def compile_bundle(
    bundle: HarnessBundle,
    *,
    harness_id: str,
    allowed_tools: Collection[str],
) -> CompiledHarness:
    """Validate source/callback interfaces without executing policy functions."""
    if not isinstance(bundle, HarnessBundle):
        raise TypeError("bundle must be a HarnessBundle")
    if not isinstance(harness_id, str) or not harness_id.strip():
        raise ValueError("harness_id must be a non-empty string")
    if (
        not isinstance(allowed_tools, Collection)
        or isinstance(allowed_tools, (str, bytes))
        or any(type(name) is not str or not name for name in allowed_tools)
    ):
        raise ValueError("allowed_tools must be a collection of non-empty tool names")
    normalized_tools = tuple(sorted(set(allowed_tools)))
    if set(normalized_tools) & (set(WORKBENCH_NAMES) | {"finish"}):
        raise ValueError("adapter primitives cannot use reserved modular tool names")
    validated = HarnessBundle.from_dict(
        bundle.to_dict(), primitive_tools=(*normalized_tools, *WORKBENCH_NAMES)
    )
    # Constructing a session checks every declared program and required arity.
    # It does not call setup or any other Builder function or adapter tool.
    session = ModularSession(validated)
    try:
        return CompiledHarness(
            harness_id=harness_id.strip(),
            harness_sha256=validated.sha256,
            system_prompt=validated.instruction + "\n\n" + FIXED_RUNNER_BOUNDARY,
            reflection_interval=0,
            reflection_instruction="",
            tool_names=normalized_tools,
            bundle=validated,
        )
    finally:
        session.close()


__all__ = ["compile_bundle"]
