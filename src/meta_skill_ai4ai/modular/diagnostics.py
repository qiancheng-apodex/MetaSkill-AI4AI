"""A value-free, fixed-vocabulary public view of generated-module failures.

This is a schema allowlist, not authentication of arbitrary logs. Runtime binds
real callback failures to this shape; archive readers must still verify their
source and seals. Exception messages, free names, paths and values never enter
the public view. Source columns use AST's zero-based UTF-8 byte offsets.
"""

from __future__ import annotations

from typing import Any


ERROR_CODES = frozenset(
    {
        "policy_execution_error",
        "invalid_indexed_assignment",
        "invalid_augmented_assignment",
        "module_contract_error",
        "policy_validation_error",
    }
)
ERROR_TYPES = frozenset(
    {"PolicyExecutionError", "ModuleContractError", "PolicyValidationError"}
)
JSON_TYPE_NAMES = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "NoneType",
        "slice",
        "other",
    }
)
CALLBACKS = {
    "memory": frozenset({"write", "retrieve"}),
    "context": frozenset({"select_context"}),
    "controller": frozenset({"on_event"}),
    "verification": frozenset({"before_finish", "on_failure"}),
    "workspace": frozenset({"setup"}),
    "tools": frozenset({"next_action"}),
}


def json_type_name(value: Any) -> str:
    """Never expose a user-defined class name or invoke a conversion method."""
    names = {
        dict: "dict",
        list: "list",
        tuple: "tuple",
        str: "str",
        int: "int",
        float: "float",
        bool: "bool",
        type(None): "NoneType",
        slice: "slice",
    }
    return names.get(type(value), "other")


def source_location(value: object) -> dict[str, int] | None:
    if type(value) is not dict:
        return None
    line, column = value.get("line"), value.get("column")
    if (
        type(line) is int
        and 1 <= line <= 100001
        and type(column) is int
        and 0 <= column <= 400000
    ):
        return {"line": line, "column": column}
    return None


def public_module_failure(event: object) -> dict[str, Any] | None:
    """Project a failure event through a strict, finite, JSON-only allowlist.

    Malformed required fields fail closed. Optional malformed fields and all
    unknown fields are discarded, never copied or converted to strings.
    A bounded location is only emitted with a valid fixed callback identity.
    """
    if (
        type(event) is not dict
        or type(event.get("type")) is not str
        or event["type"] != "module_failure"
    ):
        return None
    raw = event.get("public_diagnostic")
    if (
        type(raw) is not dict
        or type(raw.get("schema_version")) is not int
        or raw["schema_version"] != 1
    ):
        return None
    code, error_type = raw.get("error_code"), raw.get("error_type")
    if (
        type(code) is not str
        or code not in ERROR_CODES
        or type(error_type) is not str
        or error_type not in ERROR_TYPES
    ):
        return None
    expected = {
        "module_contract_error": "ModuleContractError",
        "policy_validation_error": "PolicyValidationError",
    }.get(code, "PolicyExecutionError")
    if error_type != expected:
        return None
    result: dict[str, Any] = {
        "schema_version": 1,
        "error_code": code,
        "error_type": error_type,
    }
    module, callback = raw.get("module"), raw.get("callback")
    bound = (
        type(module) is str
        and type(callback) is str
        and callback in CALLBACKS.get(module, ())
    )
    if type(module) is str and module == "tools":
        tool_index = raw.get("tool_index")
        bound = bound and type(tool_index) is int and 0 <= tool_index < 32
    if bound:
        result.update(module=module, callback=callback)
        if module == "tools":
            result["tool_index"] = tool_index
        location = source_location(raw.get("source_location"))
        if location is not None:
            result["source_location"] = location
    operand_types = raw.get("operand_types")
    if (
        code in {"invalid_indexed_assignment", "invalid_augmented_assignment"}
        and type(operand_types) is dict
    ):
        filtered = {
            key: operand_types[key]
            for key in ("container", "index")
            if type(operand_types.get(key)) is str
            and operand_types[key] in JSON_TYPE_NAMES
        }
        if filtered:
            result["operand_types"] = filtered
    return {"type": "module_failure", "public_diagnostic": result}


__all__ = ["public_module_failure"]
