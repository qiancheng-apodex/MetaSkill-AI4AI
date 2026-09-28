"""Bounded wire records for modular-v2.0 harnesses and support skills.

Structure and evidence membership are checked here. The interpreter validates
and runs generated source without Python eval or exec.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


INTERFACE_VERSION = "modular-v2.0"
MAX_JSON_BYTES = 524_288
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 20_000
MAX_SOURCE_BYTES = 65_536
MAX_INSTRUCTION_BYTES = 16_384
MAX_TOOLS = 32
MAX_SKILL_BYTES = 24_576
MAX_SKILL_TEXT_BYTES = 4_096
MAX_SKILL_EVIDENCE = 512
COMPONENT_MODULES = ("memory", "context", "controller", "verification", "workspace")
RESERVED_TOOL_NAMES = frozenset({"finish"})
_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_SKILL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_BUNDLE_FIELDS = {"interface_version", "instruction", "tools", *COMPONENT_MODULES}
_SUPPORT_SKILL_FIELDS = {
    "skill_id",
    "interface_version",
    "when",
    "provide",
    "use",
    "evidence_ids",
}
_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)
_SCHEMA_KEYWORDS = frozenset(
    {
        "type",
        "description",
        "title",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "default",
        "examples",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "anyOf",
        "allOf",
        "oneOf",
        "not",
    }
)


class BundleValidationError(ValueError):
    """A path-qualified error in a bundle, component, schema or module skill."""


def _fail(path: str, message: str) -> None:
    raise BundleValidationError(f"{path}: {message}")


def _text(value: Any, path: str, limit: int, *, empty: bool = False) -> str:
    if type(value) is not str:
        _fail(path, "expected a string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        _fail(path, "invalid Unicode surrogate")
    if size > limit:
        _fail(path, f"exceeds {limit} UTF-8 bytes")
    if not empty and not value.strip():
        _fail(path, "must not be empty or whitespace")
    return value


def _object(value: Any, fields: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(path, "expected a JSON object")
    if any(type(key) is not str for key in value):
        _fail(path, "object keys must be strings")
    missing, extra = fields - value.keys(), value.keys() - fields
    if missing:
        _fail(path, f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        _fail(path, f"unknown fields: {', '.join(sorted(extra))}")
    return value


def _check_json(value: Any, *, limit: int = MAX_JSON_BYTES) -> None:
    nodes = 0
    ancestors: set[int] = set()

    def visit(item: Any, depth: int, path: str) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES:
            _fail(path, "too many JSON values")
        if depth > MAX_JSON_DEPTH:
            _fail(path, "JSON nesting too deep")
        kind = type(item)
        if item is None or kind is bool:
            return
        if kind is str:
            _text(item, path, limit, empty=True)
            return
        if kind is int:
            if item.bit_length() > 512:
                _fail(path, "integer exceeds 512 bits")
            return
        if kind is float:
            if not math.isfinite(item):
                _fail(path, "nonfinite numbers are not JSON")
            return
        if kind not in (dict, list):
            _fail(path, "expected only JSON objects, arrays and scalar values")
        if id(item) in ancestors:
            _fail(path, "cyclic JSON value")
        ancestors.add(id(item))
        if kind is dict:
            for key, child in item.items():
                _text(key, path, limit, empty=True)
                visit(child, depth + 1, f"{path}.{key}")
        else:
            for index, child in enumerate(item):
                visit(child, depth + 1, f"{path}[{index}]")
        ancestors.remove(id(item))

    visit(value, 0, "$")
    if len(_canonical(value).encode("utf-8")) > limit:
        _fail("$", f"document exceeds {limit} UTF-8 bytes")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_json(source: str, *, limit: int = MAX_JSON_BYTES) -> Any:
    _text(source, "$", limit)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail("$", f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        _fail("$", f"nonfinite JSON literal: {value}")

    try:
        value = json.loads(source, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, BundleValidationError):
            raise
        raise BundleValidationError(f"$: invalid JSON: {exc}") from exc
    _check_json(value, limit=limit)
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _version(value: Any) -> None:
    if value != INTERFACE_VERSION or type(value) is not str:
        _fail("interface_version", f"must equal {INTERFACE_VERSION!r}")


def _strings(
    value: Any,
    path: str,
    *,
    maximum: int,
    item_bytes: int = 128,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        _fail(path, "expected a list of strings")
    if len(value) > maximum or (not value and not allow_empty):
        _fail(path, f"expected {'0' if allow_empty else '1'}..{maximum} entries")
    result = tuple(
        _text(item, f"{path}[{i}]", item_bytes) for i, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        _fail(path, "duplicate entries")
    return result


def _tool_name(name: Any, path: str = "tools.name") -> str:
    _text(name, path, 64)
    if not _TOOL_NAME.fullmatch(name):
        _fail(path, "expected a letter followed by letters, digits or underscores")
    if name in RESERVED_TOOL_NAMES:
        _fail(path, "reserved runner tool name")
    return name


def _allowlist(value: Any, path: str) -> frozenset[str]:
    if not isinstance(value, Collection) or isinstance(value, (str, bytes)):
        _fail(path, "expected a collection of strings, not a string")
    if any(type(item) is not str for item in value):
        _fail(path, "expected only string entries")
    return frozenset(value)


def _parameter_schema(
    schema: Any, path: str = "parameters", *, root: bool = True
) -> None:
    if type(schema) is bool and not root:
        return
    if type(schema) is not dict:
        _fail(path, "expected a JSON Schema object")
    unknown = schema.keys() - _SCHEMA_KEYWORDS
    if unknown:
        _fail(path, f"unsupported JSON Schema keywords: {', '.join(sorted(unknown))}")
    if root and schema.get("type") != "object":
        _fail(path, "tool parameters must have type object")
    if "type" in schema:
        types = schema["type"]
        if type(types) is str:
            types = [types]
        if (
            type(types) is not list
            or not types
            or any(type(t) is not str or t not in _SCHEMA_TYPES for t in types)
        ):
            _fail(path + ".type", "invalid JSON Schema type")
        if len(set(types)) != len(types):
            _fail(path + ".type", "duplicate schema types")
    for name in ("description", "title"):
        if name in schema:
            _text(schema[name], f"{path}.{name}", MAX_SKILL_TEXT_BYTES, empty=True)
    properties = schema.get("properties", {})
    if type(properties) is not dict:
        _fail(path + ".properties", "expected an object")
    for name, subschema in properties.items():
        _parameter_schema(subschema, f"{path}.properties.{name}", root=False)
    if "required" in schema:
        names = _strings(
            schema["required"], path + ".required", maximum=256, allow_empty=True
        )
        if not set(names) <= properties.keys():
            _fail(path + ".required", "required names must be declared in properties")
    for name in ("additionalProperties", "items", "not"):
        if name in schema:
            _parameter_schema(schema[name], f"{path}.{name}", root=False)
    for name in ("anyOf", "allOf", "oneOf"):
        if name in schema:
            choices = schema[name]
            if type(choices) is not list or not choices:
                _fail(f"{path}.{name}", "expected a nonempty schema array")
            for index, choice in enumerate(choices):
                _parameter_schema(choice, f"{path}.{name}[{index}]", root=False)
    if "enum" in schema:
        choices = schema["enum"]
        if type(choices) is not list or not choices:
            _fail(path + ".enum", "expected a nonempty array")
        if len({_canonical(item) for item in choices}) != len(choices):
            _fail(path + ".enum", "duplicate enum values")
    if "examples" in schema and type(schema["examples"]) is not list:
        _fail(path + ".examples", "expected an array")
    if "uniqueItems" in schema and type(schema["uniqueItems"]) is not bool:
        _fail(path + ".uniqueItems", "expected a boolean")
    for name in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ):
        if name in schema and (type(schema[name]) is not int or schema[name] < 0):
            _fail(f"{path}.{name}", "expected a nonnegative integer")
    for name in (
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ):
        if name in schema and type(schema[name]) not in (int, float):
            _fail(f"{path}.{name}", "expected a number")
    if "multipleOf" in schema and schema["multipleOf"] <= 0:
        _fail(path + ".multipleOf", "must be positive")


@dataclass(frozen=True, slots=True)
class Component:
    source: str

    def __post_init__(self) -> None:
        _text(self.source, "source", MAX_SOURCE_BYTES)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Component":
        _check_json(data)
        return cls(**_object(data, {"source"}, "component"))

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source}


@dataclass(frozen=True, slots=True)
class ComposedTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    source: str

    def __post_init__(self) -> None:
        _tool_name(self.name)
        _text(self.description, "description", MAX_SKILL_TEXT_BYTES)
        _text(self.source, "source", MAX_SOURCE_BYTES)
        # Permit dataclasses.replace on an existing immutable tool as well as
        # a new plain JSON dictionary. Wire parsing remains JSON-only.
        parameters = (
            _thaw(self.parameters)
            if isinstance(self.parameters, MappingProxyType)
            else self.parameters
        )
        _check_json(parameters)
        _parameter_schema(parameters)
        object.__setattr__(self, "parameters", _freeze(parameters))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComposedTool":
        _check_json(data)
        return cls(
            **_object(data, {"name", "description", "parameters", "source"}, "tool")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _thaw(self.parameters),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class HarnessBundle:
    interface_version: str
    instruction: str
    memory: Component | None
    tools: tuple[ComposedTool, ...]
    context: Component | None
    controller: Component | None
    verification: Component | None
    workspace: Component | None

    def __post_init__(self) -> None:
        _version(self.interface_version)
        _text(self.instruction, "instruction", MAX_INSTRUCTION_BYTES)
        for name in COMPONENT_MODULES:
            component = getattr(self, name)
            if component is not None and type(component) is not Component:
                _fail(name, "expected Component or None")
        if type(self.tools) not in (list, tuple) or len(self.tools) > MAX_TOOLS:
            _fail("tools", f"expected at most {MAX_TOOLS} tools")
        if any(type(tool) is not ComposedTool for tool in self.tools):
            _fail("tools", "expected ComposedTool entries")
        object.__setattr__(self, "tools", tuple(self.tools))
        if len({tool.name for tool in self.tools}) != len(self.tools):
            _fail("tools", "duplicate tool names")
        _check_json(self.to_dict())

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, primitive_tools: Collection[str] = ()
    ) -> "HarnessBundle":
        _check_json(data)
        _object(data, _BUNDLE_FIELDS, "bundle")
        if type(data["tools"]) is not list:
            _fail("tools", "expected a JSON array")
        tools = tuple(ComposedTool.from_dict(tool) for tool in data["tools"])
        collision = {tool.name for tool in tools} & _allowlist(
            primitive_tools, "primitive_tools"
        )
        if collision:
            _fail(
                "tools",
                f"names collide with primitive tools: {', '.join(sorted(collision))}",
            )
        components = {
            name: None if data[name] is None else Component.from_dict(data[name])
            for name in COMPONENT_MODULES
        }
        return cls(
            interface_version=data["interface_version"],
            instruction=data["instruction"],
            tools=tools,
            **components,
        )

    @classmethod
    def from_json(
        cls, source: str, *, primitive_tools: Collection[str] = ()
    ) -> "HarnessBundle":
        return cls.from_dict(_parse_json(source), primitive_tools=primitive_tools)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "interface_version": self.interface_version,
            "instruction": self.instruction,
            "tools": [tool.to_dict() for tool in self.tools],
        }
        for name in COMPONENT_MODULES:
            component = getattr(self, name)
            result[name] = None if component is None else component.to_dict()
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()


def neutral_bundle() -> HarnessBundle:
    """Return the versioned B0 with no researcher-authored module policies."""
    return HarnessBundle(
        interface_version=INTERFACE_VERSION,
        instruction="Complete the task using the provided tools.",
        memory=None,
        tools=(),
        context=None,
        controller=None,
        verification=None,
        workspace=None,
    )


@dataclass(frozen=True, slots=True)
class SupportSkill:
    """One reusable when/provide/use support need learned from evidence."""

    skill_id: str
    interface_version: str
    when: str
    provide: str
    use: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _version(self.interface_version)
        _text(self.skill_id, "skill_id", 128)
        if not _SKILL_ID.fullmatch(self.skill_id):
            _fail("skill_id", "invalid identifier")
        for name in ("when", "provide", "use"):
            _text(getattr(self, name), name, MAX_SKILL_TEXT_BYTES)
        object.__setattr__(
            self,
            "evidence_ids",
            _strings(
                self.evidence_ids,
                "evidence_ids",
                maximum=MAX_SKILL_EVIDENCE,
                item_bytes=256,
            ),
        )
        _check_json(self.to_dict(), limit=MAX_SKILL_BYTES)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        known_episode_ids: Collection[str] | None = None,
    ) -> "SupportSkill":
        _check_json(data, limit=MAX_SKILL_BYTES)
        _object(data, _SUPPORT_SKILL_FIELDS, "support_skill")
        if type(data["evidence_ids"]) is not list:
            _fail("evidence_ids", "expected a JSON array")
        result = cls(**data)
        if known_episode_ids is not None:
            evidence = _allowlist(known_episode_ids, "known_episode_ids")
            if not set(result.evidence_ids) <= evidence:
                _fail(
                    "evidence_ids",
                    "contains IDs outside the induction evidence allowlist",
                )
        return result

    @classmethod
    def from_json(
        cls,
        source: str,
        *,
        known_episode_ids: Collection[str] | None = None,
    ) -> "SupportSkill":
        return cls.from_dict(
            _parse_json(source, limit=MAX_SKILL_BYTES),
            known_episode_ids=known_episode_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "interface_version": self.interface_version,
            "when": self.when,
            "provide": self.provide,
            "use": self.use,
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def semantic_dict(self) -> dict[str, str]:
        return {"when": self.when, "provide": self.provide, "use": self.use}

    @property
    def semantic_sha256(self) -> str:
        return hashlib.sha256(
            _canonical(self.semantic_dict).encode("utf-8")
        ).hexdigest()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()
