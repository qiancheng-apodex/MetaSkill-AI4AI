"""Execute generated policies through fixed, counted public capabilities.

No generated program receives a provider, adapter, filesystem object or private
task handle. The host executes every requested primitive and records its result.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from ..api import (
    AuthenticationFailure,
    ChatCompletionError,
    InfrastructureFailure,
    ModelContentFailure,
    RequestCompatibilityFailure,
)
from ..harness.compiler import FIXED_RUNNER_BOUNDARY
from ..harness.target_runner import (
    TargetBudget,
    TargetRunResult,
    TargetTask,
    Tool,
    ToolResult,
    _metadata_dict,
    _metadata_infrastructure_retry_count,
    _safe_json,
    _usage_total,
)
from ..token_count import approximate_token_count
from .diagnostics import CALLBACKS, ERROR_CODES, public_module_failure
from .interpreter import PolicyExecutionError, PolicyProgram, PolicyValidationError
from .schema import HarnessBundle
from .state import (
    MemoryStore,
    ModuleContractError,
    RuntimeLimits,
    Workbench,
    context_groups,
    describe_context_groups,
    json_copy,
    project_context,
)


WORKBENCH_NAMES = ("workbench_write", "workbench_read", "workbench_list")
_MODULE_ERRORS = (ModuleContractError, PolicyExecutionError, PolicyValidationError)


@dataclass(frozen=True)
class _FailureBinding:
    """Host-created identity of this failing invocation, never session history."""

    module: str
    callback: str
    tool_index: int | None = None


def _public_failure_diagnostic(error: Exception) -> dict[str, Any]:
    if isinstance(error, PolicyExecutionError):
        code = error.error_code
        if (
            type(code) is not str
            or code not in ERROR_CODES
            or code in {"module_contract_error", "policy_validation_error"}
        ):
            code = "policy_execution_error"
        diagnostic: dict[str, Any] = {
            "schema_version": 1,
            "error_code": code,
            "error_type": "PolicyExecutionError",
            "operand_types": error.operand_types,
        }
    elif isinstance(error, PolicyValidationError):
        diagnostic = {
            "schema_version": 1,
            "error_code": "policy_validation_error",
            "error_type": "PolicyValidationError",
        }
    else:
        diagnostic = {
            "schema_version": 1,
            "error_code": "module_contract_error",
            "error_type": "ModuleContractError",
        }
    binding = getattr(error, "_modular_failure_binding", None)
    if type(binding) is _FailureBinding:
        diagnostic.update(module=binding.module, callback=binding.callback)
        if binding.module == "tools":
            diagnostic["tool_index"] = binding.tool_index
        if isinstance(error, PolicyExecutionError):
            diagnostic["source_location"] = error.source_location
    projected = public_module_failure(
        {"type": "module_failure", "public_diagnostic": diagnostic}
    )
    assert projected is not None  # All required values above are host constants.
    return projected["public_diagnostic"]


def validate_arguments(
    value: Any, schema: Mapping[str, Any] | bool, path: str = "arguments"
) -> None:
    """Validate the tool-schema subset used by public benchmark capabilities."""
    if schema is True:
        return
    if schema is False:
        raise ModuleContractError(f"{path}: value is forbidden by schema")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ModuleContractError(f"{path}: value differs from const")
    if "not" in schema:
        try:
            validate_arguments(value, schema["not"], path)
        except ModuleContractError:
            pass
        else:
            raise ModuleContractError(f"{path}: forbidden schema matched")
    for key in ("anyOf", "oneOf"):
        if key not in schema:
            continue
        matches = 0
        for alternative in schema[key]:
            try:
                validate_arguments(value, alternative, path)
                matches += 1
            except ModuleContractError:
                pass
        if not matches or key == "oneOf" and matches != 1:
            raise ModuleContractError(f"{path}: schema alternatives do not match")
    for requirement in schema.get("allOf", []):
        validate_arguments(value, requirement, path)
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    checks = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) is int or type(value) is float and math.isfinite(value),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if kinds is not None and not any(checks.get(kind, False) for kind in kinds):
        raise ModuleContractError(f"{path}: wrong argument type")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        raise ModuleContractError(f"{path}: value is outside enum")
    if isinstance(value, dict):
        if len(value) < schema.get("minProperties", 0) or len(value) > schema.get(
            "maxProperties", math.inf
        ):
            raise ModuleContractError(f"{path}: object size outside bounds")
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - set(value):
            raise ModuleContractError(f"{path}: missing required arguments")
        extra_schema = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in properties:
                validate_arguments(item, properties[name], f"{path}.{name}")
            elif extra_schema is False:
                raise ModuleContractError(f"{path}: unexpected argument {name}")
            elif isinstance(extra_schema, Mapping):
                validate_arguments(item, extra_schema, f"{path}.{name}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", math.inf
        ):
            raise ModuleContractError(f"{path}: array length outside bounds")
        for item in value:
            validate_arguments(item, schema.get("items", {}), path + "[]")
        if schema.get("uniqueItems") and any(
            _json_equal(item, prior)
            for index, item in enumerate(value)
            for prior in value[:index]
        ):
            raise ModuleContractError(f"{path}: array contains duplicate items")
    if isinstance(value, str) and (
        len(value) < schema.get("minLength", 0)
        or len(value) > schema.get("maxLength", math.inf)
    ):
        raise ModuleContractError(f"{path}: string length outside bounds")
    if type(value) in (int, float):
        if value < schema.get("minimum", -math.inf) or value > schema.get(
            "maximum", math.inf
        ):
            raise ModuleContractError(f"{path}: numeric argument outside bounds")
        if value <= schema.get("exclusiveMinimum", -math.inf) or value >= schema.get(
            "exclusiveMaximum", math.inf
        ):
            raise ModuleContractError(
                f"{path}: numeric argument outside exclusive bounds"
            )
        if "multipleOf" in schema:
            # Decimal representations avoid binary-float modulo surprises.
            from fractions import Fraction

            if Fraction(str(value)) % Fraction(str(schema["multipleOf"])):
                raise ModuleContractError(
                    f"{path}: numeric argument is not a permitted multiple"
                )


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality distinguishes booleans from numbers, unlike Python ==."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _shape(
    value: Any, allowed: set[str], required: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ModuleContractError(f"invalid {label} return fields")
    return value


class ModularSession:
    """One episode's programs, memory, scratch workspace and controller state.

    Harness-Bench explicitly reuses this object across its dependent rounds.
    A new task always receives a new session, even when bundle bytes are shared.
    """

    def __init__(
        self,
        bundle: HarnessBundle,
        *,
        limits: RuntimeLimits | None = None,
        workbench_factory: Callable[[RuntimeLimits], Workbench] | None = None,
    ) -> None:
        self.bundle = bundle
        self.limits = limits or RuntimeLimits()
        self.programs: dict[str, PolicyProgram] = {}
        requirements = {
            "memory": {"write": 2, "retrieve": 2},
            "context": {"select_context": 2},
            "controller": {"on_event": 2},
            "verification": {"before_finish": 2},
            "workspace": {"setup": 1},
        }
        for name, required in requirements.items():
            component = getattr(bundle, name)
            if component is not None:
                self._add_program(name, component.source, required)
        for tool in bundle.tools:
            self._add_program("tools." + tool.name, tool.source, {"next_action": 3})
        verification = self.programs.get("verification")
        if (
            verification
            and "on_failure" in verification.function_names
            and verification.arity("on_failure") != 2
        ):
            raise ModuleContractError(
                "verification.on_failure must accept two arguments"
            )
        self.memory = MemoryStore(self.limits)
        # Only host-side admission/tests can inject a memory-backed workbench.
        # Generated source receives JSON, never this factory or its object.
        self.workbench = (workbench_factory or Workbench)(self.limits)
        self.state: dict[str, Any] = {}
        self.phase = ""
        self.instruction = ""
        self.visible_tools: list[str] | None = None
        self.public_events: list[dict[str, Any]] = []
        self.policy_calls = 0
        self.initialized = False
        self.task_id: str | None = None

    def _add_program(self, name: str, source: str, required: dict[str, int]) -> None:
        program = PolicyProgram(source)
        for function, arity in required.items():
            if (
                function not in program.function_names
                or program.arity(function) != arity
            ):
                raise ModuleContractError(
                    f"{name}.{function} must accept {arity} arguments"
                )
        self.programs[name] = program

    def call(self, module: str, function: str, *args: Any) -> Any:
        self.policy_calls += 1
        if self.policy_calls > self.limits.max_policy_calls:
            raise ModuleContractError("episode policy-call allowance exceeded")
        program = self.programs[module]
        try:
            return program.call(function, list(args))
        except _MODULE_ERRORS as error:
            # Bind only a real candidate callback. A subsequent host contract
            # failure or exhausted episode allowance must not inherit this.
            if function in program.function_names:
                if function in CALLBACKS.get(module, ()):
                    error._modular_failure_binding = _FailureBinding(module, function)
                elif function == "next_action":
                    for index, tool in enumerate(self.bundle.tools):
                        if module == "tools." + tool.name:
                            error._modular_failure_binding = _FailureBinding(
                                "tools", function, index
                            )
                            break
            raise

    def snapshot(self, remaining: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "controller": copy.deepcopy(self.state),
            "phase": self.phase,
            "memory": self.memory.records,
            "recent_events": copy.deepcopy(
                self.public_events[-self.limits.event_tail :]
            ),
            "remaining": dict(remaining),
        }

    def observe(
        self,
        event: dict[str, Any],
        remaining: Mapping[str, Any],
        log: list[dict[str, Any]],
        tool_names: set[str],
    ) -> None:
        event = json_copy(event)
        self.public_events.append(event)
        # Only the bounded tail is policy state; the run's permanent event log
        # retains the complete history separately.
        self.public_events = self.public_events[-self.limits.event_tail :]
        while (
            len(self.public_events) > 1
            and len(json.dumps(self.public_events, ensure_ascii=False)) > 32768
        ):
            self.public_events.pop(0)
        log.append(event)
        if "memory" in self.programs:
            decision = self.call("memory", "write", event, self.memory.records)
            self.memory.apply(decision)
            log.append(
                {
                    "type": "memory_write",
                    "decision": decision,
                    "record_count": len(self.memory.records),
                }
            )
        if "controller" in self.programs:
            decision = self.call(
                "controller", "on_event", event, self.snapshot(remaining)
            )
            _shape(
                decision,
                {"state", "phase", "visible_tools", "instruction"},
                set(),
                "controller",
            )
            if "state" in decision:
                if not isinstance(decision["state"], dict):
                    raise ModuleContractError("controller state must be an object")
                self.state = json_copy(
                    decision["state"], max_chars=self.limits.max_state_chars
                )
            if "phase" in decision:
                if (
                    not isinstance(decision["phase"], str)
                    or len(decision["phase"]) > 128
                ):
                    raise ModuleContractError("controller phase must be a short string")
                self.phase = decision["phase"]
            if "instruction" in decision:
                if (
                    not isinstance(decision["instruction"], str)
                    or len(decision["instruction"]) > self.limits.max_instruction_chars
                ):
                    raise ModuleContractError(
                        "controller instruction exceeds its limit"
                    )
                self.instruction = decision["instruction"]
            if "visible_tools" in decision:
                visible = decision["visible_tools"]
                if visible is not None and (
                    not isinstance(visible, list)
                    or any(
                        not isinstance(x, str) or x not in tool_names for x in visible
                    )
                    or len(set(visible)) != len(visible)
                ):
                    raise ModuleContractError(
                        "controller selected unknown or duplicate tools"
                    )
                self.visible_tools = visible
            log.append({"type": "controller_decision", "decision": decision})

    def model_input(
        self,
        transcript: list[dict[str, Any]],
        task: TargetTask,
        log: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # An adapter may define a common public view before any Builder-owned
        # context selection. The permanent transcript remains unchanged.
        public_transcript = (
            task.input_projection(transcript)
            if task.input_projection is not None else transcript
        )
        selected = copy.deepcopy(public_transcript)
        retrieved: list[dict[str, Any]] = []
        if "memory" in self.programs:
            query = {
                "task": task.prompt,
                "phase": self.phase,
                "latest_event": self.public_events[-1] if self.public_events else None,
            }
            identifiers = self.call("memory", "retrieve", query, self.memory.records)
            retrieved = self.memory.retrieve(identifiers)
            log.append(
                {"type": "memory_retrieve", "ids": identifiers, "records": retrieved}
            )
        if "context" in self.programs:
            groups = context_groups(public_transcript)
            keep = self.call(
                "context", "select_context", describe_context_groups(groups), retrieved
            )
            selected = project_context(public_transcript, groups, keep)
            log.append(
                {
                    "type": "context_selection",
                    "kept_groups": keep,
                    "total_groups": len(groups),
                }
            )
        if retrieved or self.instruction or self.phase:
            selected.append(
                {
                    "role": "user",
                    "content": _safe_json(
                        {
                            "harness_memory": retrieved,
                            "harness_phase": self.phase,
                            "harness_instruction": self.instruction,
                        },
                        self.limits.max_memory_chars
                        + self.limits.max_instruction_chars,
                    ),
                }
            )
        return selected

    def close(self) -> None:
        self.workbench.close()


class _StopRun(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class ModularTargetRunner:
    def __init__(
        self, provider: Any, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if not callable(getattr(provider, "create", None)):
            raise TypeError("provider must expose create")
        self.provider = provider
        self.clock = clock

    def run(
        self,
        *,
        task: TargetTask,
        bundle: HarnessBundle,
        harness_id: str,
        tools: Sequence[Tool],
        budget: TargetBudget,
        model: str,
        reasoning_effort: str | None,
        seed: int | None,
        temperature: float | None = 0.0,
        session: ModularSession | None = None,
    ) -> TargetRunResult:
        started = self.clock()
        owned_session = session is None
        events: list[dict[str, Any]] = []
        transcript: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": bundle.instruction + "\n\n" + FIXED_RUNNER_BOUNDARY,
            },
            {
                "role": "user",
                "content": _safe_json(
                    {
                        "task": task.prompt,
                        "initial_observation": task.initial_observation,
                        "public_metadata": dict(task.public_metadata),
                    },
                    budget.max_observation_chars,
                ),
            },
        ]
        errors: Counter[str] = Counter()
        model_calls = tool_calls = accounted_tokens = retries = 0
        consecutive_empty_responses = 0
        complete_accounting = True
        final_answer: str | None = None
        stop_reason = "max_steps"

        def remaining() -> dict[str, Any]:
            return {
                "model_calls": budget.max_steps - model_calls,
                "tool_calls": budget.max_tool_calls - tool_calls,
                "tokens": budget.max_accounted_tokens - accounted_tokens,
                "seconds": max(
                    0.0, budget.wall_time_seconds - (self.clock() - started)
                ),
            }

        def check_budget(*, tool: bool = False) -> None:
            if self.clock() - started >= budget.wall_time_seconds:
                raise _StopRun("wall_time_exhausted")
            if accounted_tokens >= budget.max_accounted_tokens:
                raise _StopRun("token_budget_exhausted")
            if tool and tool_calls >= budget.max_tool_calls:
                raise _StopRun("tool_budget_exhausted")

        try:
            if session is None:
                session = ModularSession(bundle)
            if session.bundle.sha256 != bundle.sha256:
                raise ModuleContractError(
                    "session bundle differs from requested harness"
                )
            if session.task_id not in (None, task.instance_id):
                raise ModuleContractError(
                    "a modular session cannot be shared across tasks"
                )
            session.task_id = task.instance_id
            primitive: dict[str, Tool] = {}
            for tool in tools:
                if tool.name in primitive or tool.name in WORKBENCH_NAMES:
                    raise ModuleContractError("duplicate or reserved primitive tool")
                primitive[tool.name] = tool

            object_schema = {"type": "object", "additionalProperties": False}
            workbench_tools = [
                Tool(
                    "workbench_write",
                    "Write a text file in task-local scratch space, not the official deliverable workspace.",
                    {
                        **object_schema,
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                    lambda args: ToolResult(
                        True, session.workbench.write(args["path"], args["content"])
                    ),
                ),
                Tool(
                    "workbench_read",
                    "Read a task-local scratch file.",
                    {
                        **object_schema,
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                    lambda args: ToolResult(True, session.workbench.read(args["path"])),
                ),
                Tool(
                    "workbench_list",
                    "List task-local scratch files.",
                    {**object_schema, "properties": {}},
                    lambda args: ToolResult(True, session.workbench.listing()),
                ),
            ]
            primitive.update({tool.name: tool for tool in workbench_tools})
            composed = {tool.name: tool for tool in bundle.tools}
            if len(composed) != len(bundle.tools) or set(composed) & (
                set(primitive) | {"finish"}
            ):
                raise ModuleContractError(
                    "composed tool name collides with a fixed capability"
                )
            tool_names = set(primitive) | set(composed)

            def observe(event: dict[str, Any]) -> None:
                session.observe(event, remaining(), events, tool_names)

            def execute_primitive(
                name: Any, arguments: Any, *, origin: str
            ) -> ToolResult:
                nonlocal tool_calls
                check_budget(tool=True)
                tool_calls += 1
                if not isinstance(name, str) or name not in primitive:
                    result = ToolResult(
                        False, {"error": "unknown primitive tool"}, "unknown_tool"
                    )
                else:
                    try:
                        arguments = json_copy(arguments)
                        validate_arguments(arguments, primitive[name].parameters)
                    except ModuleContractError as error:
                        # The raw model argument string is already archived in
                        # the transcript. Do not propagate nonfinite JSON into
                        # policy inputs or the structured event log.
                        arguments = {"rejected_invalid_arguments": True}
                        result = ToolResult(
                            False, {"error": str(error)}, "invalid_tool_arguments"
                        )
                    else:
                        try:
                            result = primitive[name].handler(arguments)
                            if not isinstance(result, ToolResult):
                                raise TypeError("tool handler must return ToolResult")
                        except ModuleContractError as error:
                            result = ToolResult(
                                False, {"error": str(error)}, "workspace_error"
                            )
                        except Exception as error:
                            result = ToolResult(
                                False,
                                {"error": type(error).__name__},
                                "tool_handler_exception",
                                infrastructure_failure=True,
                            )
                if not result.ok:
                    errors[result.error_type or "tool_error"] += 1
                public_result = {
                    "ok": result.ok,
                    "observation": json.loads(
                        _safe_json(result.observation, budget.max_observation_chars)
                    ),
                    "error_type": result.error_type,
                    "infrastructure_failure": result.infrastructure_failure,
                    "terminal_reason": result.terminal_reason,
                }
                event = {
                    "type": "primitive_result",
                    "tool": name,
                    "arguments": arguments,
                    "origin": origin,
                    "result": public_result,
                }
                events.append(
                    {
                        "type": "tool_result",
                        "step": model_calls,
                        "tool": name,
                        "ok": result.ok,
                        "error_type": result.error_type,
                        "infrastructure_failure": result.infrastructure_failure,
                        "terminal_reason": result.terminal_reason,
                    }
                )
                # Infrastructure or official terminal status wins over generated
                # recovery. Neither may be disguised as an ordinary tool error.
                if result.infrastructure_failure:
                    events.append(event)
                    raise _StopRun("infrastructure_failure")
                if result.terminal_reason:
                    events.append(event)
                    raise _StopRun(result.terminal_reason)
                observe(event)
                check_budget()
                return ToolResult(
                    result.ok, public_result["observation"], result.error_type
                )

            def actions(value: Any, *, origin: str) -> list[dict[str, Any]]:
                if (
                    not isinstance(value, list)
                    or len(value) > session.limits.max_composed_actions
                ):
                    raise ModuleContractError("policy action list exceeds its limit")
                results = []
                for action in value:
                    _shape(
                        action,
                        {"tool", "arguments"},
                        {"tool", "arguments"},
                        "primitive request",
                    )
                    result = execute_primitive(
                        action["tool"], action["arguments"], origin=origin
                    )
                    results.append(asdict(result))
                return results

            def invoke_composed(name: str, arguments: dict[str, Any]) -> ToolResult:
                nonlocal tool_calls
                spec = composed[name]
                try:
                    arguments = json_copy(arguments)
                    validate_arguments(arguments, spec.to_dict()["parameters"])
                except ModuleContractError as error:
                    check_budget(tool=True)
                    tool_calls += 1
                    errors["invalid_tool_arguments"] += 1
                    return ToolResult(
                        False, {"error": str(error)}, "invalid_tool_arguments"
                    )
                state: Any = {}
                previous: Any = None
                before = tool_calls
                for _ in range(session.limits.max_composed_actions + 1):
                    check_budget()
                    decision = session.call(
                        "tools." + name, "next_action", arguments, state, previous
                    )
                    _shape(
                        decision,
                        {"state", "call", "return"},
                        {"state"},
                        "composed tool",
                    )
                    if ("call" in decision) == ("return" in decision):
                        raise ModuleContractError(
                            "composed tool must either call a primitive or return"
                        )
                    state = json_copy(
                        decision["state"], max_chars=session.limits.max_state_chars
                    )
                    if "return" in decision:
                        primitive_actions = tool_calls - before
                        if tool_calls == before:
                            # Pure computation still consumes one tool action.
                            check_budget(tool=True)
                            tool_calls += 1
                        events.append(
                            {
                                "type": "composed_return",
                                "tool": name,
                                "primitive_actions": primitive_actions,
                                "charged_actions": tool_calls - before,
                                "result": decision["return"],
                            }
                        )
                        return ToolResult(True, decision["return"])
                    if tool_calls - before >= session.limits.max_composed_actions:
                        raise ModuleContractError(
                            "composed tool primitive allowance exceeded"
                        )
                    action = _shape(
                        decision["call"],
                        {"tool", "arguments"},
                        {"tool", "arguments"},
                        "composed primitive request",
                    )
                    previous = asdict(
                        execute_primitive(
                            action["tool"],
                            action["arguments"],
                            origin="composed:" + name,
                        )
                    )
                raise ModuleContractError("composed tool did not return")

            def observe_submission(allowed: bool, feedback: str) -> None:
                # Completion is not a primitive action, but its result is public
                # feedback needed by stateful retry and one-time gate policies.
                observe({
                    "type": "submission_result",
                    "tool": "finish",
                    "result": asdict(ToolResult(
                        allowed, {"feedback": feedback},
                        None if allowed else "verification_rejected",
                    )),
                })

            def verify(answer: str) -> tuple[bool, str]:
                if "verification" not in session.programs:
                    return True, ""
                decision = session.call(
                    "verification",
                    "before_finish",
                    answer,
                    session.snapshot(remaining()),
                )
                _shape(
                    decision,
                    {"allow", "feedback", "actions"},
                    {"allow", "feedback"},
                    "verification",
                )
                if type(decision["allow"]) is not bool or not isinstance(
                    decision["feedback"], str
                ):
                    raise ModuleContractError(
                        "verification needs boolean allow and text feedback"
                    )
                if decision["allow"] and decision.get("actions"):
                    raise ModuleContractError(
                        "verification cannot accept before requested checks execute"
                    )
                checks = actions(decision.get("actions", []), origin="verification")
                events.append(
                    {
                        "type": "verification_decision",
                        "allow": decision["allow"],
                        "feedback": decision["feedback"],
                        "check_results": checks,
                    }
                )
                feedback = _safe_json(
                    {"verification": decision["feedback"], "checks": checks},
                    budget.max_observation_chars,
                )
                return decision["allow"], feedback

            observe(
                {
                    "type": "round_start",
                    "task": {
                        "prompt": task.prompt,
                        "initial_observation": task.initial_observation,
                        "public_metadata": dict(task.public_metadata),
                    },
                }
            )
            if not session.initialized:
                session.initialized = True
                if "workspace" in session.programs:
                    setup = session.call(
                        "workspace",
                        "setup",
                        {
                            "prompt": task.prompt,
                            "initial_observation": task.initial_observation,
                            "public_metadata": dict(task.public_metadata),
                        },
                    )
                    _shape(setup, {"files", "actions"}, set(), "workspace setup")
                    files = setup.get("files", [])
                    if (
                        not isinstance(files, list)
                        or len(files) > session.limits.max_workspace_files
                    ):
                        raise ModuleContractError("invalid workspace setup files")
                    for file in files:
                        _shape(
                            file,
                            {"path", "content"},
                            {"path", "content"},
                            "workspace file",
                        )
                        execute_primitive(
                            "workbench_write", file, origin="workspace_setup"
                        )
                    setup_results = actions(
                        setup.get("actions", []), origin="workspace_setup"
                    )
                    transcript.append(
                        {
                            "role": "user",
                            "content": _safe_json(
                                {
                                    "workbench_files": session.workbench.listing(),
                                    "setup_results": setup_results,
                                },
                                budget.max_observation_chars,
                            ),
                        }
                    )

            finish_definition = {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Submit the answer through the harness completion check.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            }
            definitions = {
                name: tool.api_definition() for name, tool in primitive.items()
            }
            definitions.update(
                {
                    name: {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.description,
                            "parameters": tool.to_dict()["parameters"],
                        },
                    }
                    for name, tool in composed.items()
                }
            )

            for step in range(1, budget.max_steps + 1):
                check_budget()
                observe(
                    {"type": "before_model", "step": step, "remaining": remaining()}
                )
                messages = session.model_input(transcript, task, events)
                # Public resource counters are available to every condition.
                # This does not decide whether the Target should continue or submit.
                messages[0]["content"] += "\n\n" + json.dumps({
                    "remaining_budget_before_this_request": {k: v for k, v in remaining().items() if k != "seconds"},
                    "token_accounting": "Input and output tokens of this and later requests count toward the remaining total."
                }, ensure_ascii=False)
                visible = (
                    tool_names
                    if session.visible_tools is None
                    else set(session.visible_tools)
                )
                api_tools = [definitions[name] for name in sorted(visible)] + [
                    finish_definition
                ]
                prompt_tokens = approximate_token_count(
                    json.dumps(
                        {"messages": messages, "tools": api_tools}, ensure_ascii=False
                    )
                )
                output_limit = min(
                    budget.max_completion_tokens_per_call,
                    budget.max_accounted_tokens - accounted_tokens - prompt_tokens,
                )
                if output_limit <= 0:
                    raise _StopRun("token_budget_exhausted")
                # Actual model-visible projections remain reconstructable even
                # when context selection omits older transcript exchanges.
                events.append(
                    {
                        "type": "model_request",
                        "step": step,
                        "messages": copy.deepcopy(messages),
                        "visible_tools": sorted(visible),
                        "max_completion_tokens": output_limit,
                    }
                )
                check_budget()
                try:
                    completion = self.provider.create(
                        model=model,
                        messages=copy.deepcopy(messages),
                        max_completion_tokens=output_limit,
                        reasoning_effort=reasoning_effort,
                        seed=seed,
                        temperature=temperature,
                        additional_fields={"tools": api_tools, "tool_choice": "auto"},
                    )
                except (
                    InfrastructureFailure,
                    AuthenticationFailure,
                    RequestCompatibilityFailure,
                    ModelContentFailure,
                    ChatCompletionError,
                ) as error:
                    metadata = _metadata_dict(error.request_metadata)
                    retries += _metadata_infrastructure_retry_count(metadata)
                    reason = (
                        "infrastructure_failure"
                        if isinstance(error, InfrastructureFailure)
                        else (
                            "configuration_failure"
                            if isinstance(
                                error,
                                (AuthenticationFailure, RequestCompatibilityFailure),
                            )
                            else "model_failure"
                        )
                    )
                    events.append({"type": reason, "step": step, "metadata": metadata})
                    raise _StopRun(reason) from error
                model_calls += 1
                used, complete = _usage_total(completion, messages)
                if not complete:
                    used += approximate_token_count(
                        json.dumps(api_tools, ensure_ascii=False)
                    )
                accounted_tokens += used
                complete_accounting = complete_accounting and complete
                metadata = _metadata_dict(getattr(completion, "metadata", {}))
                retries += _metadata_infrastructure_retry_count(
                    metadata.get("request", {})
                )
                events.append(
                    {"type": "model_response", "step": step, "metadata": metadata}
                )
                if getattr(completion, "is_model_content_failure", False):
                    content = getattr(completion, "message", None)
                    events.append(
                        {
                            "type": "model_content_failure",
                            "message": (
                                copy.deepcopy(content)
                                if isinstance(content, Mapping)
                                else None
                            ),
                        }
                    )
                    raise _StopRun("model_failure")
                if accounted_tokens > budget.max_accounted_tokens:
                    raise _StopRun("token_budget_exhausted")
                if self.clock() - started >= budget.wall_time_seconds:
                    raise _StopRun("wall_time_exhausted")
                message = getattr(completion, "message", None)
                if (
                    not isinstance(message, Mapping)
                    or message.get("role") != "assistant"
                ):
                    raise _StopRun("model_failure")
                transcript.append(copy.deepcopy(dict(message)))
                calls = message.get("tool_calls")
                if not calls:
                    answer = message.get("content")
                    if not isinstance(answer, str) or not answer.strip():
                        # Preserve the original response in the request archive;
                        # never infer tool actions from reasoning-only text.
                        transcript.pop()
                        consecutive_empty_responses += 1
                        observe({"type": "model_output_error", "step": model_calls,
                            "error": "empty_action_and_answer",
                            "consecutive_empty_responses": consecutive_empty_responses})
                        if consecutive_empty_responses > 2:
                            raise _StopRun("model_failure")
                        transcript.append({"role": "user", "content":
                            "No executable tool call or final answer was returned. "
                            "Return a valid tool call or a nonempty final answer. "
                            "This correction uses the remaining task budget."})
                        continue
                    consecutive_empty_responses = 0
                    allowed, feedback = verify(answer)
                    observe_submission(allowed, feedback)
                    if allowed:
                        final_answer, stop_reason = answer, "assistant_final"
                        events.append({"type": "assistant_final", "step": model_calls})
                        break
                    transcript.append({"role": "user", "content": feedback})
                    continue
                if not isinstance(calls, list) or any(
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not call["id"]
                    or not isinstance(call.get("function"), dict)
                    or not isinstance(call["function"].get("name"), str)
                    for call in calls
                ):
                    raise _StopRun("model_failure")
                if len({call["id"] for call in calls}) != len(calls):
                    raise _StopRun("model_failure")
                consecutive_empty_responses = 0
                for call in calls:
                    name = call["function"].get("name")
                    raw_arguments = call["function"].get("arguments")
                    try:
                        arguments = (
                            json.loads(raw_arguments)
                            if isinstance(raw_arguments, str)
                            else None
                        )
                        if not isinstance(arguments, dict):
                            raise ValueError("arguments must be an object")
                    except (ValueError, RecursionError):
                        check_budget(tool=True)
                        tool_calls += 1
                        errors["invalid_tool_arguments"] += 1
                        result = ToolResult(
                            False,
                            {"error": "invalid tool arguments"},
                            "invalid_tool_arguments",
                        )
                    else:
                        if name == "finish":
                            check_budget(tool=True)
                            tool_calls += 1
                            if len(calls) != 1:
                                result = ToolResult(
                                    False,
                                    {
                                        "error": "finish must be the only call in its turn"
                                    },
                                    "invalid_finish",
                                )
                                errors["invalid_finish"] += 1
                            elif set(arguments) != {"answer"} or not isinstance(
                                arguments["answer"], str
                            ):
                                result = ToolResult(
                                    False,
                                    {"error": "finish requires a text answer"},
                                    "invalid_finish",
                                )
                                errors["invalid_finish"] += 1
                            else:
                                allowed, feedback = verify(arguments["answer"])
                                observe_submission(allowed, feedback)
                                result = ToolResult(
                                    allowed,
                                    {"verification": feedback},
                                    None if allowed else "verification_rejected",
                                )
                                if allowed:
                                    final_answer, stop_reason = (
                                        arguments["answer"],
                                        "target_finish",
                                    )
                                    events.append(
                                        {"type": "finish", "step": model_calls}
                                    )
                        elif not isinstance(name, str) or name not in visible:
                            check_budget(tool=True)
                            tool_calls += 1
                            errors["tool_not_exposed"] += 1
                            result = ToolResult(
                                False,
                                {"error": "tool not exposed in this phase"},
                                "tool_not_exposed",
                            )
                        elif name in composed:
                            result = invoke_composed(name, arguments)
                        else:
                            result = execute_primitive(name, arguments, origin="target")
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name if isinstance(name, str) else "invalid",
                            "content": _safe_json(
                                asdict(result), budget.max_observation_chars
                            ),
                        }
                    )
                    if (
                        not result.ok
                        and "verification" in session.programs
                        and "on_failure"
                        in session.programs["verification"].function_names
                    ):
                        recovery = session.call(
                            "verification",
                            "on_failure",
                            {"tool": name, "result": asdict(result)},
                            session.snapshot(remaining()),
                        )
                        _shape(
                            recovery,
                            {"actions", "feedback"},
                            {"actions", "feedback"},
                            "recovery",
                        )
                        if not isinstance(recovery["feedback"], str):
                            raise ModuleContractError("recovery feedback must be text")
                        recovered = actions(recovery["actions"], origin="recovery")
                        # Append after the entire native call group, below.
                        events.append(
                            {
                                "type": "recovery_decision",
                                "feedback": recovery["feedback"],
                                "results": recovered,
                            }
                        )
                if final_answer is not None:
                    break
                recovery_events = [
                    event
                    for event in events
                    if event.get("type") == "recovery_decision"
                    and not event.get("delivered")
                    and (event.get("feedback") or event.get("results"))
                ]
                if recovery_events:
                    transcript.append(
                        {
                            "role": "user",
                            "content": _safe_json(
                                {"recovery": copy.deepcopy(recovery_events)},
                                budget.max_observation_chars,
                            ),
                        }
                    )
                    # Delivery bookkeeping is separate from the public policy
                    # event stream; immutable archival happens on return.
                    for event in recovery_events:
                        event["delivered"] = True
        except _StopRun as stop:
            stop_reason = stop.reason
        except _MODULE_ERRORS as error:
            errors["module_failure"] += 1
            stop_reason = "module_failure"
            events.append(
                {
                    "type": "module_failure",
                    "error_class": type(error).__name__,
                    "message": str(error),
                    "public_diagnostic": _public_failure_diagnostic(error),
                }
            )
        finally:
            # A terminal budget/environment event can interrupt a native batch.
            # Preserve API pairing without inventing a successful observation.
            pending: list[dict[str, Any]] = []
            for message in transcript:
                if message.get("role") == "assistant":
                    raw_calls = message.get("tool_calls")
                    pending = (
                        [
                            call
                            for call in raw_calls
                            if isinstance(call, dict)
                            and isinstance(call.get("id"), str)
                        ]
                        if isinstance(raw_calls, list)
                        else []
                    )
                elif message.get("role") == "tool":
                    pending = [
                        call
                        for call in pending
                        if call["id"] != message.get("tool_call_id")
                    ]
            for call in pending:
                function = call.get("function", {})
                name = function.get("name") if isinstance(function, dict) else None
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name if isinstance(name, str) else "invalid",
                        "content": _safe_json(
                            {
                                "runner_terminated": stop_reason,
                                "not_a_tool_observation": True,
                                "details_in_primitive_events": True,
                            },
                            budget.max_observation_chars,
                        ),
                    }
                )
            if session is not None:
                events.append(
                    {
                        "type": "module_accounting",
                        "policy_calls_cumulative": session.policy_calls,
                        "memory_writes_cumulative": session.memory.writes,
                        "memory_reads_cumulative": session.memory.reads,
                        "workbench_files": session.workbench.listing(),
                    }
                )
                if owned_session:
                    session.close()

        return TargetRunResult(
            instance_id=task.instance_id,
            model=model,
            harness_id=harness_id,
            harness_sha256=bundle.sha256,
            seed=seed,
            stop_reason=stop_reason,
            final_answer=final_answer,
            steps=model_calls,
            tool_calls=tool_calls,
            model_calls=model_calls,
            accounted_tokens=accounted_tokens,
            token_accounting_complete=complete_accounting,
            tool_error_counts=dict(errors),
            infrastructure_retry_count=retries,
            wall_time_seconds=max(0.0, self.clock() - started),
            events=tuple(copy.deepcopy(events)),
            transcript=tuple(copy.deepcopy(transcript)),
        )
