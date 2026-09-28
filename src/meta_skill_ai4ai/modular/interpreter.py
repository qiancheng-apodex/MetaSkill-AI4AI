"""A bounded interpreter for Builder-written, JSON-only Python policies.

Source is parsed and interpreted, never passed to Python ``eval`` or ``exec``.
Only the syntax, functions and methods explicitly dispatched below exist. This
is a small policy language, not a sandbox for executing arbitrary Python.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any

from .diagnostics import json_type_name, source_location


class PolicyValidationError(ValueError):
    """Source uses unsupported syntax or exceeds construction limits."""

    def __init__(self, message: str, *, issues=None, omitted: int = 0) -> None:
        super().__init__(message)
        self.issues = issues or []
        self.issues_omitted = omitted
        self.source_location = self.issues[0]["source_location"] if self.issues else None


class PolicyExecutionError(ValueError):
    """A policy failed, exceeded its allowance, or produced invalid data."""

    def __init__(
        self,
        *args: Any,
        error_code: str = "policy_execution_error",
        source_location: dict[str, int] | None = None,
        operand_types: dict[str, str] | None = None,
    ) -> None:
        super().__init__(*args)
        self.error_code = error_code
        self.source_location = source_location
        self.operand_types = operand_types


@dataclass(frozen=True, slots=True)
class PolicyLimits:
    max_instructions: int = 50_000
    max_loop_iterations: int = 10_000
    max_call_depth: int = 24
    max_data_nodes: int = 20_000
    max_container_items: int = 4_096
    max_string_chars: int = 65_536
    max_data_chars: int = 262_144
    max_source_chars: int = 100_000
    max_syntax_nodes: int = 20_000
    max_integer_bits: int = 256
    max_data_depth: int = 64

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")


_SCALAR_MATH = {
    "sqrt": (math.sqrt, 1), "exp": (math.exp, 1), "expm1": (math.expm1, 1),
    "log": (math.log, 1), "log10": (math.log10, 1), "log1p": (math.log1p, 1),
    "sin": (math.sin, 1), "cos": (math.cos, 1), "tan": (math.tan, 1),
    "asin": (math.asin, 1), "acos": (math.acos, 1), "atan": (math.atan, 1),
    "atan2": (math.atan2, 2), "sinh": (math.sinh, 1), "cosh": (math.cosh, 1),
    "tanh": (math.tanh, 1), "degrees": (math.degrees, 1), "radians": (math.radians, 1),
}
_MATH_CONSTANTS = {"pi": math.pi, "e": math.e}
_BUILTIN_POSITIONAL_ARITIES: dict[str, tuple[int, int | None]] = {
    "len": (1, 1),
    "range": (1, 3),
    "min": (1, None),
    "max": (1, None),
    "sum": (1, 2),
    "sorted": (1, 1),
    "enumerate": (1, 2),
    "zip": (0, None),
    "str": (0, 1),
    "json_dumps": (1, 1),
    "int": (0, 1),
    "float": (0, 1),
    "abs": (1, 1),
    "round": (1, 2),
    **{name: (arity, arity) for name, (_, arity) in _SCALAR_MATH.items()},
}
_BUILTINS = frozenset(_BUILTIN_POSITIONAL_ARITIES)
_METHODS = frozenset({"get", "items", "keys", "values", "append", "extend"})
_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.Expr,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Load,
    ast.Store,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
)


@dataclass(slots=True)
class _JSONValidation:
    nodes: int = 0
    containers: set[int] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _ValidatedContainer:
    value: Any
    nodes: int
    dependencies: frozenset[int]


def _json(
    value: Any,
    limits: PolicyLimits,
    *,
    clone: bool = False,
    spend=None,
    validation: _JSONValidation | None = None,
) -> Any:
    """Validate exact JSON types, including cycles and aggregate size."""
    active: set[int] = set()
    nodes = chars = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal nodes, chars
        nodes += 1
        if spend is not None:
            spend()
        if nodes > limits.max_data_nodes or depth > limits.max_data_depth:
            raise PolicyExecutionError("JSON data exceeds node/depth limit")
        kind = type(item)
        if item is None or kind is bool:
            return item
        if kind is int:
            if item.bit_length() > limits.max_integer_bits:
                raise PolicyExecutionError("integer exceeds bit limit")
            return item
        if kind is float:
            if not math.isfinite(item):
                raise PolicyExecutionError("non-finite number")
            return item
        if kind is str:
            chars += len(item)
            if len(item) > limits.max_string_chars or chars > limits.max_data_chars:
                raise PolicyExecutionError("JSON text exceeds size limit")
            return item
        if kind not in (list, dict):
            raise PolicyExecutionError("only exact JSON data types are accepted")
        if len(item) > limits.max_container_items:
            raise PolicyExecutionError("container exceeds item limit")
        if id(item) in active:
            raise PolicyExecutionError("cyclic data is forbidden")
        active.add(id(item))
        if kind is list:
            result = [visit(child, depth + 1) for child in item] if clone else None
            if not clone:
                for child in item:
                    visit(child, depth + 1)
        else:
            result = {} if clone else None
            for key, child in item.items():
                if type(key) is not str:
                    raise PolicyExecutionError("JSON object keys must be strings")
                visit(key, depth + 1)
                copied = visit(child, depth + 1)
                if clone:
                    result[key] = copied
        active.remove(id(item))
        if validation is not None:
            validation.containers.add(id(result if clone else item))
        return result if clone else item

    result = visit(value, 0)
    if validation is not None:
        validation.nodes = nodes
    return result


class PolicyProgram:
    """Validated policy functions; each call receives a fresh execution budget."""

    def __init__(self, source: str, *, limits: PolicyLimits | None = None, numeric_syntax: bool = False) -> None:
        if type(numeric_syntax) is not bool:
            raise TypeError("numeric_syntax must be a boolean")
        self.numeric_syntax = numeric_syntax
        self.limits = limits if limits is not None else PolicyLimits()
        if not isinstance(self.limits, PolicyLimits):
            raise TypeError("limits must be PolicyLimits")
        if type(source) is not str or len(source) > self.limits.max_source_chars:
            raise PolicyValidationError("source must be a bounded string")
        try:
            tree = ast.parse(source, mode="exec")
        except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
            raise PolicyValidationError("invalid or excessive policy source") from error
        if numeric_syntax:
            self._validate_numeric_syntax(tree)
        self._functions: dict[str, ast.FunctionDef] = {}
        for statement in tree.body:
            if numeric_syntax and isinstance(statement, ast.Import):
                continue  # Preflight admitted only the fixed, inert `import math`.
            if not isinstance(statement, ast.FunctionDef):
                raise PolicyValidationError(
                    "top level must contain only function definitions"
                )
            if (
                statement.name in self._functions
                or statement.name in _BUILTINS
                or numeric_syntax and statement.name == "math"
                or not self._name(statement.name)
            ):
                raise PolicyValidationError("duplicate or forbidden function name")
            self._functions[statement.name] = statement
        if not self._functions:
            raise PolicyValidationError("policy must define at least one function")
        reserved = _BUILTINS | self._functions.keys() | ({"math"} if numeric_syntax else set())
        called_attributes = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        allowed = _NODES + (ast.Import, ast.alias, ast.ListComp, ast.comprehension) if numeric_syntax else _NODES
        for index, node in enumerate(ast.walk(tree), 1):
            if index > self.limits.max_syntax_nodes or not isinstance(node, allowed):
                raise PolicyValidationError(
                    f"unsupported or excessive syntax: {type(node).__name__}"
                )
            if isinstance(node, ast.FunctionDef):
                if (
                    node not in tree.body
                    or node.decorator_list
                    or node.returns
                    or getattr(node, "type_params", [])
                ):
                    raise PolicyValidationError(
                        "nested/decorated/annotated functions are forbidden"
                    )
                args = node.args
                if (
                    args.defaults
                    or args.kw_defaults
                    or args.kwonlyargs
                    or args.vararg
                    or args.kwarg
                ):
                    raise PolicyValidationError(
                        "functions require fixed positional parameters"
                    )
                names = [arg.arg for arg in args.posonlyargs + args.args]
                if len(set(names)) != len(names):
                    raise PolicyValidationError("duplicate parameter")
            if isinstance(node, ast.arg):
                if node.annotation or not self._name(node.arg) or node.arg in reserved:
                    raise PolicyValidationError(
                        "forbidden parameter name or annotation"
                    )
            if isinstance(node, ast.Name):
                if not self._name(node.id) or (
                    isinstance(node.ctx, ast.Store) and node.id in reserved
                ):
                    raise PolicyValidationError("forbidden name or reserved assignment")
            if isinstance(node, ast.Constant):
                try:
                    _json(node.value, self.limits)
                except PolicyExecutionError as error:
                    raise PolicyValidationError(str(error)) from error
            if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
                raise PolicyValidationError("dictionary unpacking is forbidden")
            if isinstance(node, ast.Attribute):
                if numeric_syntax and isinstance(node.value, ast.Name) and node.value.id == "math":
                    continue  # All attributes and uses were checked in preflight.
                if node.attr not in _METHODS:
                    raise PolicyValidationError(
                        "attribute is not an allowed data method"
                    )
                if id(node) not in called_attributes:
                    raise PolicyValidationError(
                        "method references cannot be stored or returned"
                    )
            if isinstance(node, ast.Call):
                if any(keyword.arg is None for keyword in node.keywords):
                    raise PolicyValidationError("keyword unpacking is forbidden")
                if isinstance(node.func, ast.Name):
                    if node.func.id not in reserved:
                        raise PolicyValidationError(f"unknown function: {node.func.id}")
                    if node.func.id in _BUILTINS:
                        self._validate_builtin_call(node)
                elif not isinstance(node.func, ast.Attribute):
                    raise PolicyValidationError("indirect calls are forbidden")
                elif numeric_syntax and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                    self._validate_builtin_call(node, node.func.attr)
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.For)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    self._validate_target(target)
        self.function_names = tuple(self._functions)

    def _validate_numeric_syntax(self, tree: ast.Module) -> None:
        """Bounded diagnostics for compute only; no source rewriting or imports."""
        issues, total = [], 0

        def reject(node, message):
            nonlocal total
            total += 1
            if len(issues) < 12:
                issues.append({"syntax": type(node).__name__, "message": message,
                    "source_location": {"line": getattr(node, "lineno", 1),
                        "column": getattr(node, "col_offset", 0)}})

        nodes, stack = [], [(tree, False)]
        while stack:
            node, inside_comprehension = stack.pop()
            nodes.append(node)
            if len(nodes) > self.limits.max_syntax_nodes:
                raise PolicyValidationError("syntax node budget exceeded")
            if isinstance(node, ast.ListComp) and (inside_comprehension or len(node.generators) != 1
                or not isinstance(node.generators[0].target, ast.Name) or node.generators[0].is_async):
                reject(node, "Use one non-nested list comprehension with one variable target, or explicit bounded for loops.")
            stack.extend((child, inside_comprehension or isinstance(node, ast.ListComp))
                for child in reversed(list(ast.iter_child_nodes(node))))
        called = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
        top_level = {id(n) for n in tree.body}
        math_roots = {id(n.value) for n in nodes if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "math"}
        hints = {ast.While: "Use a bounded for loop, optionally with break.",
            ast.GeneratorExp: "Build a list with a for loop before reducing it.",
            ast.SetComp: "Build a JSON list with a for loop.",
            ast.DictComp: "Build a JSON dictionary with a for loop.",
            ast.ImportFrom: "Use exactly import math and the documented math functions; from-import is unsupported."}
        allowed = _NODES + (ast.Import, ast.alias, ast.ListComp, ast.comprehension)
        for node in nodes:
            if not isinstance(node, allowed):
                reject(node, hints.get(type(node), "Use documented functions, if and bounded for loops over JSON data."))
            if id(node) in top_level and not isinstance(node, (ast.FunctionDef, ast.Import)):
                reject(node, "Move calculation statements inside calculate(records) or a top-level helper function.")
            if isinstance(node, ast.Import) and (len(node.names) != 1 or node.names[0].name != "math" or node.names[0].asname):
                reject(node, "Only exact import math is supported; no aliases or other modules.")
            if isinstance(node, ast.FunctionDef) and (id(node) not in top_level or node.decorator_list or node.returns):
                reject(node, "Put undecorated, unannotated helper functions at top level and pass values explicitly.")
            if (isinstance(node, ast.Name) and node.id == "math" and
                (not isinstance(node.ctx, ast.Load) or id(node) not in math_roots)
                or isinstance(node, ast.arg) and node.arg == "math"
                or isinstance(node, ast.FunctionDef) and node.name == "math"):
                reject(node, "math is a fixed namespace, not a value or assignable name; use another variable name.")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math":
                valid = (isinstance(node.ctx, ast.Load) and
                    (node.attr in _SCALAR_MATH and id(node) in called
                     or node.attr in _MATH_CONSTANTS and id(node) not in called))
                if not valid:
                    reject(node, "Use only direct calls to documented math functions or read math.pi/math.e; no aliases or assignment.")
        if total:
            raise PolicyValidationError("Unsupported compute syntax; see located issues and rewrite hints.",
                issues=issues, omitted=total - len(issues))

    @staticmethod
    def _name(name: str) -> bool:
        return not name.startswith("_") and name not in {"True", "False", "None"}

    @staticmethod
    def _validate_builtin_call(node: ast.Call, name: str | None = None) -> None:
        """Check call syntax without executing arguments or choosing branches.

        Counts and keyword names are known for every direct call: unpacked call
        arguments are not part of this language. Data-dependent types, values,
        and resource use remain checked by the bounded execution dispatcher.
        """
        name = name if name is not None else node.func.id
        minimum, maximum = _BUILTIN_POSITIONAL_ARITIES[name]
        count = len(node.args)
        if count < minimum or (maximum is not None and count > maximum):
            if maximum is None:
                expected = f"at least {minimum}"
            elif minimum == maximum:
                expected = str(minimum)
            else:
                expected = f"{minimum} to {maximum}"
            raise PolicyValidationError(
                f"{name} expects {expected} positional arguments; received {count}"
            )
        seen: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg in seen:
                raise PolicyValidationError(f"duplicate keyword for {name}")
            seen.add(keyword.arg)
            if name != "sorted" or keyword.arg != "reverse":
                raise PolicyValidationError(f"unsupported keyword for {name}")
            value = keyword.value
            if (
                isinstance(value, ast.Constant) and type(value.value) is not bool
            ) or isinstance(value, (ast.List, ast.Tuple, ast.Dict)):
                raise PolicyValidationError("sorted reverse must be a boolean")

    @classmethod
    def _validate_target(cls, target: ast.AST) -> None:
        if isinstance(target, (ast.Name, ast.Subscript)):
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                cls._validate_target(child)
            return
        raise PolicyValidationError("unsupported assignment target")

    def arity(self, name: str) -> int:
        if type(name) is not str or name not in self._functions:
            raise PolicyValidationError(f"unknown function: {name}")
        args = self._functions[name].args
        return len(args.posonlyargs) + len(args.args)

    def call(
        self, function_name: str, arguments: Sequence[Any], *, accounting: dict[str, int] | None = None
    ) -> Any:
        if type(function_name) is not str:
            raise PolicyExecutionError("function name must be a string")
        if type(arguments) not in (list, tuple):
            raise PolicyExecutionError(
                "arguments must be a list or tuple of JSON values"
            )
        if accounting is not None and type(accounting) is not dict:
            raise TypeError("accounting must be a plain dictionary")
        execution = _Execution(self)
        try:
            copied = _json(
                list(arguments), self.limits, clone=True, spend=execution.spend
            )
            result = execution.invoke(function_name, copied)
            return _json(result, self.limits, clone=True, spend=execution.spend)
        except PolicyExecutionError:
            raise
        except (
            ArithmeticError,
            LookupError,
            TypeError,
            ValueError,
            RecursionError,
            _Break,
            _Continue,
        ) as error:
            raise PolicyExecutionError(
                f"policy operation failed: {type(error).__name__}",
                source_location=execution.failure_location,
            ) from error
        finally:
            if accounting is not None:
                accounting.update(instructions=execution.instructions, loop_iterations=execution.iterations)


class _Return(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Execution:
    def __init__(self, program: PolicyProgram) -> None:
        self.program = program
        self.limits = program.limits
        self.instructions = self.iterations = self.depth = 0
        # A read of an unchanged, already validated JSON container is constant
        # work, not another traversal of every descendant. Each cache receipt
        # depends on ALL reachable containers, including transitive aliases.
        # Strong references prevent object-id reuse while a receipt is live.
        self._validated: dict[int, _ValidatedContainer] = {}
        self._validation_nodes = 0
        self.failure_location: dict[str, int] | None = None

    @contextmanager
    def located(self, node: ast.AST):
        """Annotate the innermost real statement without charging extra work.

        This context manager does not catch interpreter return/break/continue
        signals, change exception types/messages, or recover from failures.
        """
        location = source_location(
            {
                "line": getattr(node, "lineno", None),
                "column": getattr(node, "col_offset", None),
            }
        )
        try:
            yield
        except PolicyExecutionError as error:
            if error.source_location is None:
                error.source_location = location
            raise
        except (ArithmeticError, LookupError, TypeError, ValueError, RecursionError):
            if self.failure_location is None:
                self.failure_location = location
            raise

    def spend(self, amount: int = 1) -> None:
        self.instructions += amount
        if self.instructions > self.limits.max_instructions:
            raise PolicyExecutionError("instruction budget exhausted")

    def checked(self, value: Any) -> Any:
        if type(value) not in (list, dict):
            return _json(value, self.limits, spend=self.spend)
        known = self._validated.get(id(value))
        if known is not None and known.value is value:
            self.spend()
            return value
        receipt = _JSONValidation()
        _json(value, self.limits, spend=self.spend, validation=receipt)
        # Bound retained objects and dependency metadata independently of how
        # many temporary containers the policy creates. Eviction never grants
        # trust: an evicted value must pass a full validation on its next read.
        while self._validated and (
            len(self._validated) >= 128
            or self._validation_nodes + receipt.nodes > self.limits.max_data_nodes
        ):
            self.spend()
            oldest = self._validated.pop(next(iter(self._validated)))
            self._validation_nodes -= oldest.nodes
        self._validated[id(value)] = _ValidatedContainer(
            value, receipt.nodes, frozenset(receipt.containers)
        )
        self._validation_nodes += receipt.nodes
        return value

    def invalidate(self, value: Any) -> None:
        """Revoke all receipts affected by a mutation, before changing data."""
        identity = id(value)
        for key, receipt in tuple(self._validated.items()):
            self.spend()
            if identity in receipt.dependencies:
                self._validation_nodes -= receipt.nodes
                del self._validated[key]

    def data_nodes(self, value: Any) -> int:
        """Bound real recursive operation work even when operand reads cache."""
        if type(value) not in (list, dict):
            return 1
        self.checked(value)
        return self._validated[id(value)].nodes

    def invoke(self, name: str, arguments: list[Any]) -> Any:
        self.spend()
        if name not in self.program._functions:
            raise PolicyExecutionError(f"unknown policy function: {name}")
        function = self.program._functions[name]
        parameters = function.args.posonlyargs + function.args.args
        if len(parameters) != len(arguments):
            raise PolicyExecutionError(f"wrong argument count for {name}")
        self.depth += 1
        if self.depth > self.limits.max_call_depth:
            raise PolicyExecutionError("call depth exhausted")
        environment = {
            parameter.arg: value for parameter, value in zip(parameters, arguments)
        }
        try:
            self.block(function.body, environment)
        except _Return as returned:
            return returned.value
        finally:
            self.depth -= 1
        return None

    def block(self, statements: list[ast.stmt], env: dict[str, Any]) -> None:
        for statement in statements:
            with self.located(statement):
                self.spend()
                if isinstance(statement, ast.Return):
                    raise _Return(
                        self.expression(statement.value, env)
                        if statement.value
                        else None
                    )
                if isinstance(statement, ast.Assign):
                    value = self.expression(statement.value, env)
                    for target in statement.targets:
                        self.assign(target, value, env)
                elif isinstance(statement, ast.AugAssign):
                    target = statement.target
                    if isinstance(target, ast.Subscript):
                        container = self.expression(target.value, env)
                        key = self.index(target.slice, env)
                        if not (
                            type(container) is dict
                            and type(key) is str
                            or type(container) is list
                            and type(key) is int
                        ):
                            raise PolicyExecutionError(
                                "invalid augmented assignment",
                                error_code="invalid_augmented_assignment",
                                source_location={
                                    "line": target.lineno,
                                    "column": target.col_offset,
                                },
                                operand_types={
                                    "container": json_type_name(container),
                                    "index": json_type_name(key),
                                },
                            )
                        updated = self.binary(
                            statement.op,
                            container[key],
                            self.expression(statement.value, env),
                        )
                        self.invalidate(container)
                        container[key] = updated
                        self.checked(container)
                    else:
                        old = self.expression(target, env)
                        self.assign(
                            target,
                            self.binary(
                                statement.op, old, self.expression(statement.value, env)
                            ),
                            env,
                        )
                elif isinstance(statement, ast.Expr):
                    self.expression(statement.value, env)
                elif isinstance(statement, ast.If):
                    self.block(
                        (
                            statement.body
                            if self.expression(statement.test, env)
                            else statement.orelse
                        ),
                        env,
                    )
                elif isinstance(statement, ast.For):
                    values = self.sequence(self.expression(statement.iter, env))
                    broken = False
                    for value in values:
                        self.iterations += 1
                        self.spend()
                        if self.iterations > self.limits.max_loop_iterations:
                            raise PolicyExecutionError("loop budget exhausted")
                        self.assign(statement.target, value, env)
                        try:
                            self.block(statement.body, env)
                        except _Continue:
                            continue
                        except _Break:
                            broken = True
                            break
                    if not broken:
                        self.block(statement.orelse, env)
                elif isinstance(statement, ast.Break):
                    raise _Break()
                elif isinstance(statement, ast.Continue):
                    raise _Continue()
                elif self.program.numeric_syntax and isinstance(statement, ast.Import):
                    pass  # A checked declaration; no host module or environment value.
                elif not isinstance(statement, ast.Pass):
                    raise PolicyExecutionError("unsupported statement")

    def assign(self, target: ast.AST, value: Any, env: dict[str, Any]) -> None:
        self.spend()
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.List, ast.Tuple)):
            if type(value) is not list or len(target.elts) != len(value):
                raise PolicyExecutionError("unpacking requires a matching JSON array")
            for element, item in zip(target.elts, value):
                self.assign(element, item, env)
        elif isinstance(target, ast.Subscript):
            container = self.expression(target.value, env)
            key = self.index(target.slice, env)
            if type(container) is dict and type(key) is str:
                if (
                    key not in container
                    and len(container) >= self.limits.max_container_items
                ):
                    raise PolicyExecutionError("container exceeds item limit")
                self.invalidate(container)
                container[key] = value
            elif type(container) is list and type(key) is int:
                self.invalidate(container)
                container[key] = value
            else:
                raise PolicyExecutionError(
                    "invalid indexed assignment",
                    error_code="invalid_indexed_assignment",
                    source_location={
                        "line": target.lineno,
                        "column": target.col_offset,
                    },
                    operand_types={
                        "container": json_type_name(container),
                        "index": json_type_name(key),
                    },
                )
            self.checked(container)
        else:
            raise PolicyExecutionError("invalid assignment")

    def index(self, node: ast.AST, env: dict[str, Any]) -> Any:
        if isinstance(node, ast.Slice):
            values = [
                self.expression(value, env) if value else None
                for value in (node.lower, node.upper, node.step)
            ]
            if any(value is not None and type(value) is not int for value in values):
                raise PolicyExecutionError("slice indices must be integers")
            return slice(*values)
        return self.expression(node, env)

    def expression(self, node: ast.AST, env: dict[str, Any]) -> Any:
        self.spend()
        if isinstance(node, ast.Constant):
            result = node.value
        elif isinstance(node, ast.Name):
            if node.id not in env:
                raise PolicyExecutionError(f"unbound data name: {node.id}")
            result = env[node.id]
        elif isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > self.limits.max_container_items:
                raise PolicyExecutionError("array literal exceeds item limit")
            result = [self.expression(value, env) for value in node.elts]
        elif self.program.numeric_syntax and isinstance(node, ast.ListComp):
            generator = node.generators[0]
            values = self.sequence(self.expression(generator.iter, env))
            self.spend(len(env))
            local = dict(env)  # The target binding never escapes this expression.
            result = []
            for value in values:
                self.iterations += 1
                self.spend()
                if self.iterations > self.limits.max_loop_iterations:
                    raise PolicyExecutionError("loop budget exhausted")
                self.assign(generator.target, value, local)
                if all(self.expression(condition, local) for condition in generator.ifs):
                    if len(result) >= self.limits.max_container_items:
                        raise PolicyExecutionError("container exceeds item limit")
                    result.append(self.expression(node.elt, local))
        elif self.program.numeric_syntax and isinstance(node, ast.Attribute):
            result = _MATH_CONSTANTS[node.attr]  # Only validated math constants reach here.
        elif isinstance(node, ast.Dict):
            if len(node.keys) > self.limits.max_container_items:
                raise PolicyExecutionError("object literal exceeds item limit")
            result = {}
            for key, value in zip(node.keys, node.values):
                name = self.expression(key, env)
                if type(name) is not str:
                    raise PolicyExecutionError("object keys must be strings")
                result[name] = self.expression(value, env)
        elif isinstance(node, ast.Subscript):
            container = self.expression(node.value, env)
            key = self.index(node.slice, env)
            if type(container) not in (dict, list, str):
                raise PolicyExecutionError(
                    "indexing requires JSON object, array or string"
                )
            if type(container) is dict and type(key) is not str:
                raise PolicyExecutionError("object key must be a string")
            if type(container) in (list, str) and type(key) not in (int, slice):
                raise PolicyExecutionError(
                    "array/string index must be an integer or slice"
                )
            result = container[key]
        elif isinstance(node, ast.BinOp):
            result = self.binary(
                node.op,
                self.expression(node.left, env),
                self.expression(node.right, env),
            )
        elif isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand, env)
            if isinstance(node.op, ast.Not):
                result = not value
            else:
                self.numeric(value)
                result = -value if isinstance(node.op, ast.USub) else value
        elif isinstance(node, ast.BoolOp):
            result = self.expression(node.values[0], env)
            for value in node.values[1:]:
                if (
                    isinstance(node.op, ast.And)
                    and not result
                    or isinstance(node.op, ast.Or)
                    and result
                ):
                    break
                result = self.expression(value, env)
        elif isinstance(node, ast.Compare):
            left = self.expression(node.left, env)
            result = True
            for operation, right_node in zip(node.ops, node.comparators):
                right = self.expression(right_node, env)
                if not self.compare(operation, left, right):
                    result = False
                    break
                left = right
        elif isinstance(node, ast.IfExp):
            result = self.expression(
                node.body if self.expression(node.test, env) else node.orelse, env
            )
        elif isinstance(node, ast.Call):
            math_call = (self.program.numeric_syntax and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "math")
            receiver = (
                self.expression(node.func.value, env)
                if isinstance(node.func, ast.Attribute) and not math_call
                else None
            )
            arguments = [self.expression(value, env) for value in node.args]
            keywords = {
                value.arg: self.expression(value.value, env) for value in node.keywords
            }
            if len(keywords) != len(node.keywords):
                raise PolicyExecutionError("duplicate keyword")
            if math_call:
                result = self.builtin(node.func.attr, arguments, keywords)
            elif isinstance(node.func, ast.Name):
                if node.func.id in _BUILTINS:
                    result = self.builtin(node.func.id, arguments, keywords)
                else:
                    if keywords:
                        raise PolicyExecutionError(
                            "policy calls require positional arguments"
                        )
                    result = self.invoke(node.func.id, arguments)
            else:
                if keywords:
                    raise PolicyExecutionError(
                        "data methods require positional arguments"
                    )
                result = self.method(receiver, node.func.attr, arguments)
        else:
            raise PolicyExecutionError("unsupported expression")
        return self.checked(result)

    @staticmethod
    def numeric(value: Any) -> None:
        if type(value) not in (int, float):
            raise PolicyExecutionError("operation requires numbers")

    def sequence(self, value: Any) -> list[Any]:
        if type(value) not in (list, dict, str):
            raise PolicyExecutionError("operation requires an array, object or string")
        if len(value) > self.limits.max_container_items:
            raise PolicyExecutionError("iteration exceeds item limit")
        self.spend(len(value))
        return list(value)

    def binary(self, op: ast.operator, left: Any, right: Any) -> Any:
        if (
            isinstance(op, ast.Add)
            and type(left) is type(right)
            and type(left) in (str, list)
        ):
            maximum = (
                self.limits.max_string_chars
                if type(left) is str
                else self.limits.max_container_items
            )
            if len(left) + len(right) > maximum:
                raise PolicyExecutionError("concatenation exceeds size limit")
            return self.checked(left + right)
        if isinstance(op, ast.Mult) and (
            type(left) in (str, list) or type(right) in (str, list)
        ):
            sequence, count = (
                (left, right) if type(left) in (str, list) else (right, left)
            )
            if type(count) is not int:
                raise PolicyExecutionError("repetition requires integer count")
            maximum = (
                self.limits.max_string_chars
                if type(sequence) is str
                else self.limits.max_container_items
            )
            if len(sequence) * max(0, count) > maximum:
                raise PolicyExecutionError("repetition exceeds size limit")
            return self.checked(sequence * max(0, count))
        self.numeric(left)
        self.numeric(right)
        if isinstance(op, ast.Pow):
            if abs(right) > self.limits.max_integer_bits:
                raise PolicyExecutionError("exponent exceeds limit")
            if type(left) is int and type(right) is int and right >= 0:
                if (
                    max(0, abs(left).bit_length() - 1) * right
                    > self.limits.max_integer_bits
                ):
                    raise PolicyExecutionError("power exceeds integer limit")
            result = left**right
        elif isinstance(op, ast.Add):
            result = left + right
        elif isinstance(op, ast.Sub):
            result = left - right
        elif isinstance(op, ast.Mult):
            result = left * right
        elif isinstance(op, ast.Div):
            result = left / right
        elif isinstance(op, ast.FloorDiv):
            result = left // right
        elif isinstance(op, ast.Mod):
            result = left % right
        else:
            raise PolicyExecutionError("unsupported arithmetic")
        return self.checked(result)

    def compare(self, op: ast.cmpop, left: Any, right: Any) -> bool:
        if not isinstance(op, (ast.Is, ast.IsNot)):
            # Equality, ordering and membership can recursively traverse JSON.
            # Cache hits must not make repeated large comparisons unmetered.
            self.spend(self.data_nodes(left) + self.data_nodes(right))
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        if isinstance(op, (ast.Is, ast.IsNot)):
            if right is not None and type(right) is not bool:
                raise PolicyExecutionError(
                    "identity checks only support None and booleans"
                )
            return (left is right) if isinstance(op, ast.Is) else (left is not right)
        raise PolicyExecutionError("unsupported comparison")

    def json_dumps(self, value: Any) -> str:
        """Encode only validated JSON, charging before each bounded operation.

        This is deliberately not a host serializer with custom-object hooks.
        Output fragments are at most one escaped character or one bounded
        numeric literal. No full string or sorted key array is allocated before
        its work and output allowance have been checked.
        """
        _json(value, self.limits, spend=self.spend)
        fragments: list[str] = []
        output_chars = output_bytes = 0
        maximum = min(self.limits.max_string_chars, self.limits.max_data_chars)
        escapes = {
            '"': '\\"',
            "\\": "\\\\",
            "\b": "\\b",
            "\f": "\\f",
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
        }

        def emit(fragment: str) -> None:
            nonlocal output_chars, output_bytes
            if output_chars + len(fragment) > maximum:
                raise PolicyExecutionError("json_dumps output exceeds text size limit")
            size = len(fragment.encode("utf-8"))
            self.spend(size)
            output_chars += len(fragment)
            output_bytes += size
            fragments.append(fragment)

        def quote(text: str) -> None:
            emit('"')
            for character in text:
                self.spend()
                point = ord(character)
                if 0xD800 <= point <= 0xDFFF:
                    raise PolicyExecutionError("json_dumps requires valid UTF-8 text")
                if character in escapes:
                    emit(escapes[character])
                elif point < 32:
                    emit("\\u" + format(point, "04x"))
                else:
                    emit(character)
            emit('"')

        def encode(item: Any) -> None:
            self.spend()
            kind = type(item)
            if item is None:
                emit("null")
            elif kind is bool:
                emit("true" if item else "false")
            elif kind in (int, float):
                # Validation has already bounded integer bits and rejected
                # non-finite floats. Exact types cannot run user methods.
                if kind is int:
                    digits = max(1, (item.bit_length() * 30103) // 100000 + 1)
                    digits += item < 0
                    self.spend(digits)
                    if digits > maximum - output_chars + 1:
                        raise PolicyExecutionError(
                            "json_dumps output exceeds text size limit"
                        )
                else:
                    self.spend(24)  # Finite binary64 repr has bounded length.
                emit(str(item) if kind is int else repr(item))
            elif kind is str:
                quote(item)
            elif kind is list:
                emit("[")
                for index, child in enumerate(item):
                    if index:
                        emit(",")
                    encode(child)
                emit("]")
            elif kind is dict:
                # Bound string-comparison work before the actual key sort,
                # including shared prefixes of long keys and cached inputs.
                self.spend(
                    (len(item) + sum(len(key) for key in item))
                    * max(1, len(item).bit_length())
                )
                emit("{")
                for index, key in enumerate(sorted(item)):
                    if index:
                        emit(",")
                    self.spend()
                    quote(key)
                    emit(":")
                    encode(item[key])
                emit("}")
            else:
                raise PolicyExecutionError("json_dumps requires exact JSON data")

        encode(value)
        self.spend(output_bytes)
        return "".join(fragments)

    def builtin(self, name: str, args: list[Any], keywords: dict[str, Any]) -> Any:
        self.spend()
        if keywords and (
            name != "sorted"
            or set(keywords) != {"reverse"}
            or type(keywords["reverse"]) is not bool
        ):
            raise PolicyExecutionError("unsupported builtin keyword")
        if name in _SCALAR_MATH:
            function, arity = _SCALAR_MATH[name]
            if len(args) != arity or any(type(value) not in (int, float) for value in args):
                raise PolicyExecutionError(f"{name} requires {arity} finite real scalar arguments")
            if not all(math.isfinite(value) for value in args):
                raise PolicyExecutionError(f"{name} requires finite real arguments")
            try:
                result = function(*args)
            except (ValueError, OverflowError) as error:
                raise PolicyExecutionError(f"{name}: domain or overflow error", error_code="numeric_domain_or_overflow") from error
            if not math.isfinite(result):
                raise PolicyExecutionError(f"{name}: non-finite result", error_code="numeric_domain_or_overflow")
            return result
        if name == "range":
            if not 1 <= len(args) <= 3 or any(type(value) is not int for value in args):
                raise PolicyExecutionError("range requires one to three integers")
            values = range(*args)
            try:
                size = len(values)
            except OverflowError as error:
                raise PolicyExecutionError("range exceeds size limit") from error
            if size > self.limits.max_container_items:
                raise PolicyExecutionError("range exceeds size limit")
            self.spend(size)
            return list(values)
        if name == "len" and len(args) == 1:
            if type(args[0]) not in (str, list, dict):
                raise PolicyExecutionError("len requires JSON container")
            return len(args[0])
        if name in {"min", "max"} and args:
            values = self.sequence(args[0]) if len(args) == 1 else args
            self.spend(sum(self.data_nodes(value) for value in values))
            return min(values) if name == "min" else max(values)
        if name == "sum" and 1 <= len(args) <= 2:
            result = args[1] if len(args) == 2 else 0
            self.numeric(result)
            for value in self.sequence(args[0]):
                result = self.binary(ast.Add(), result, value)
            return result
        if name == "sorted" and len(args) == 1:
            values = self.sequence(args[0])
            self.spend(
                max(self.data_nodes(args[0]), len(values))
                * max(1, len(values).bit_length())
            )
            return sorted(values, reverse=keywords.get("reverse", False))
        if name == "enumerate" and 1 <= len(args) <= 2:
            start = args[1] if len(args) == 2 else 0
            if type(start) is not int:
                raise PolicyExecutionError("enumerate start must be an integer")
            return [
                [index, value]
                for index, value in enumerate(self.sequence(args[0]), start)
            ]
        if name == "zip":
            values = [self.sequence(value) for value in args]
            return [list(row) for row in zip(*values)]
        if name == "json_dumps" and len(args) == 1:
            return self.json_dumps(args[0])
        if name in {"str", "int", "float"} and len(args) <= 1:
            value = args[0] if args else ("" if name == "str" else 0)
            if type(value) not in (str, int, float, bool) and value is not None:
                raise PolicyExecutionError("conversion requires scalar JSON data")
            if name == "str":
                return str(value)
            if type(value) is str and len(value) > self.limits.max_integer_bits + 4:
                raise PolicyExecutionError("numeric conversion text exceeds limit")
            return int(value) if name == "int" else float(value)
        if name == "abs" and len(args) == 1:
            self.numeric(args[0])
            return abs(args[0])
        if name == "round" and 1 <= len(args) <= 2:
            self.numeric(args[0])
            if len(args) == 2 and (
                type(args[1]) is not int or abs(args[1]) > self.limits.max_integer_bits
            ):
                raise PolicyExecutionError("round precision exceeds limit")
            return round(*args)
        raise PolicyExecutionError(f"unsupported argument count for {name}")

    def method(self, value: Any, name: str, args: list[Any]) -> Any:
        self.spend()
        if type(value) is dict:
            if name == "get" and 1 <= len(args) <= 2 and type(args[0]) is str:
                return value.get(args[0], args[1] if len(args) == 2 else None)
            if name in {"items", "keys", "values"} and not args:
                self.spend(len(value))
                if name == "items":
                    return [[key, child] for key, child in value.items()]
                if name == "keys":
                    return list(value)
                return list(value.values())
        if type(value) is list and len(args) == 1:
            if name == "append":
                if len(value) >= self.limits.max_container_items:
                    raise PolicyExecutionError("append exceeds size limit")
                self.invalidate(value)
                value.append(args[0])
            elif name == "extend":
                other = self.sequence(args[0])
                if len(value) + len(other) > self.limits.max_container_items:
                    raise PolicyExecutionError("extend exceeds size limit")
                self.invalidate(value)
                value.extend(other)
            else:
                raise PolicyExecutionError("unsupported array method")
            self.checked(value)
            return None
        raise PolicyExecutionError("method is unavailable for this JSON value")


__all__ = [
    "PolicyProgram",
    "PolicyLimits",
    "PolicyValidationError",
    "PolicyExecutionError",
]
