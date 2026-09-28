"""Neutral storage and context boundaries; no learned retention policy lives here."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence


class ModuleContractError(ValueError):
    """A generated policy violated the fixed interface or storage allowance."""


def json_copy(value: Any, *, max_chars: int = 262144) -> Any:
    nodes = 0
    ancestors: set[int] = set()

    def validate(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > 64 or nodes > 20000:
            raise ModuleContractError("JSON value exceeds depth or node limits")
        kind = type(item)
        if item is None or kind is bool:
            return
        if kind is str:
            if len(item) > max_chars:
                raise ModuleContractError("JSON text exceeds interface limit")
            item.encode("utf-8")
            return
        if kind is int and item.bit_length() <= 512:
            return
        if kind is float and math.isfinite(item):
            return
        if kind not in (dict, list) or id(item) in ancestors:
            raise ModuleContractError(
                "interface values must be acyclic exact JSON types"
            )
        ancestors.add(id(item))
        if kind is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ModuleContractError("JSON object keys must be strings")
                validate(key, depth + 1)
                validate(child, depth + 1)
        else:
            for child in item:
                validate(child, depth + 1)
        ancestors.remove(id(item))

    try:
        validate(value, 0)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded) > max_chars:
            raise ModuleContractError("JSON value exceeds the interface size limit")
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError, UnicodeError) as error:
        raise ModuleContractError(
            "interface value must be bounded finite JSON"
        ) from error


@dataclass(frozen=True)
class RuntimeLimits:
    max_memory_records: int = 256
    max_memory_chars: int = 131072
    max_workspace_files: int = 64
    max_workspace_chars: int = 524288
    max_composed_actions: int = 32
    max_policy_calls: int = 10000
    max_state_chars: int = 65536
    max_instruction_chars: int = 8192
    event_tail: int = 24

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if type(value) is not int or value <= 0:
                raise ValueError("runtime limits must be positive integers")


class MemoryStore:
    """Episode-local records, atomically updated by Builder-selected operations."""

    def __init__(self, limits: RuntimeLimits) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._limits = limits
        self.writes = 0
        self.reads = 0

    @property
    def records(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self._records.values()))

    def apply(self, decision: Any) -> None:
        if not isinstance(decision, dict) or set(decision) != {"upsert", "delete"}:
            raise ModuleContractError(
                "memory.write returns {upsert: [records], delete: [ids]}"
            )
        upserts, deletes = decision["upsert"], decision["delete"]
        if not isinstance(upserts, list) or not isinstance(deletes, list):
            raise ModuleContractError("memory operations must be arrays")
        updated = copy.deepcopy(self._records)
        for identifier in deletes:
            if not isinstance(identifier, str) or identifier not in updated:
                raise ModuleContractError(
                    "memory deletion must name an existing record"
                )
            del updated[identifier]
        seen: set[str] = set()
        for record in upserts:
            if not isinstance(record, dict):
                raise ModuleContractError("memory records must be objects")
            identifier = record.get("id")
            if (
                not isinstance(identifier, str)
                or not identifier
                or len(identifier) > 128
                or identifier in seen
            ):
                raise ModuleContractError(
                    "memory upserts need distinct, nonempty string ids"
                )
            seen.add(identifier)
            updated[identifier] = json_copy(
                record, max_chars=self._limits.max_memory_chars
            )
        if len(updated) > self._limits.max_memory_records:
            raise ModuleContractError(
                "memory record allowance exceeded; policy must evict"
            )
        json_copy(list(updated.values()), max_chars=self._limits.max_memory_chars)
        self._records = updated
        self.writes += len(upserts) + len(deletes)

    def retrieve(self, identifiers: Any) -> list[dict[str, Any]]:
        if not isinstance(identifiers, list) or any(
            not isinstance(x, str) for x in identifiers
        ):
            raise ModuleContractError("memory.retrieve returns an array of record ids")
        if len(set(identifiers)) != len(identifiers) or any(
            x not in self._records for x in identifiers
        ):
            raise ModuleContractError("retrieved ids must be distinct existing records")
        self.reads += len(identifiers)
        return [copy.deepcopy(self._records[x]) for x in identifiers]


class Workbench:
    """Private scratch files, never mounted over an official benchmark workspace.

    All methods are offered through counted tools. The interpreter receives only
    returned JSON and cannot open this directory or any host path itself.
    """

    def __init__(self, limits: RuntimeLimits) -> None:
        self._directory = TemporaryDirectory(prefix="metaskill-v2-workbench-")
        self._root = Path(self._directory.name)
        self._limits = limits
        self._files: dict[str, str] = {}

    def _path(self, name: Any) -> Path:
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 240
            or "\\" in name
            or "\x00" in name
        ):
            raise ModuleContractError(
                "scratch path must be a short relative POSIX path"
            )
        lexical = PurePosixPath(name)
        if lexical.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")):
            raise ModuleContractError("scratch path must not escape its task directory")
        return self._root.joinpath(*lexical.parts)

    def write(self, name: Any, content: Any) -> dict[str, Any]:
        path = self._path(name)
        if not isinstance(content, str):
            raise ModuleContractError("scratch file content must be text")
        try:
            content.encode("utf-8")
            name.encode("utf-8")
        except UnicodeError as error:
            raise ModuleContractError(
                "scratch file names and content must be valid UTF-8"
            ) from error
        if any(
            existing.startswith(name + "/") or name.startswith(existing + "/")
            for existing in self._files
        ):
            raise ModuleContractError(
                "scratch file path conflicts with an existing file or directory"
            )
        updated = {**self._files, name: content}
        if (
            len(updated) > self._limits.max_workspace_files
            or sum(map(len, updated.values())) > self._limits.max_workspace_chars
        ):
            raise ModuleContractError("scratch workspace allowance exceeded")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._files = updated
        return {"path": name, "characters": len(content)}

    def read(self, name: Any) -> dict[str, str]:
        path = self._path(name)
        if name not in self._files:
            raise ModuleContractError("scratch file does not exist")
        return {"path": name, "content": path.read_text(encoding="utf-8")}

    def listing(self) -> list[dict[str, Any]]:
        return [
            {"path": name, "characters": len(content)}
            for name, content in sorted(self._files.items())
        ]

    def close(self) -> None:
        self._directory.cleanup()


def context_groups(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first system/task pair outside selection and group closed calls.

    Generated context code selects group ids, never individual tool messages.
    The immutable source transcript is not changed by this projection.
    """
    groups: list[dict[str, Any]] = []
    index = 2
    while index < len(messages):
        current = dict(messages[index])
        group = [current]
        index += 1
        calls = current.get("tool_calls")
        if calls:
            expected = [call["id"] for call in calls]
            observed: list[str] = []
            while index < len(messages) and messages[index].get("role") == "tool":
                observed.append(messages[index].get("tool_call_id"))
                group.append(dict(messages[index]))
                index += 1
            if len(set(expected)) != len(expected) or observed != expected:
                raise ModuleContractError(
                    "context contains an unpaired or reordered tool exchange"
                )
        elif current.get("role") == "tool":
            raise ModuleContractError("context contains an orphan tool response")
        groups.append({"id": len(groups), "messages": group})
    return copy.deepcopy(groups)


def project_context(
    messages: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    keep: Any,
) -> list[dict[str, Any]]:
    if not isinstance(keep, list) or any(type(x) is not int for x in keep):
        raise ModuleContractError(
            "context selection must be an array of integer group ids"
        )
    if keep != sorted(set(keep)) or any(x < 0 or x >= len(groups) for x in keep):
        raise ModuleContractError(
            "context selection must retain original order without duplicates"
        )
    if groups and len(groups) - 1 not in keep:
        raise ModuleContractError(
            "context selection must retain the newest complete exchange"
        )
    selected = copy.deepcopy(list(messages[:2]))
    for index in keep:
        selected.extend(copy.deepcopy(groups[index]["messages"]))
    return selected


def describe_context_groups(
    groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """All group ids stay selectable without copying the whole log into code.

    Fixed-size previews are interface metadata, not a retention heuristic. The
    Builder chooses ids; the host then projects the full selected exchanges.
    """
    preview_limit = min(256, 64000 // max(1, len(groups)))
    descriptions = []
    for group in groups:
        rendered = json.dumps(group["messages"], ensure_ascii=False)
        descriptions.append(
            {
                "id": group["id"],
                "roles": "/".join(message["role"] for message in group["messages"]),
                "characters": len(rendered),
                "preview": rendered[:preview_limit],
            }
        )
    return descriptions
